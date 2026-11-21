# -*- coding: utf-8 -*-
"""
llm_worker.py —— 在【服务端 ComfyUI 的 Python】里跑的 LLM 小进程。

为什么单独开进程：
    服务端 exe 是 PyInstaller 打包的 python，装不了 llama-cpp-python；而服务端 ComfyUI 的
    .venv / python_embeded 里本来就有 torch + CUDA，只要补一个 llama-cpp-python 就能跑 GGUF。
    所以 exe 用「ComfyUI 的 python + 本文件」拉起子进程，用 stdin/stdout 一行一个 JSON 通信。

协议（每行一个 JSON）：
    收：{"id":1,"cmd":"ping"}
        {"id":2,"cmd":"load","model":"D:/.../Qwen.gguf","mmproj":"D:/.../mmproj.gguf","n_ctx":4096,"n_gpu_layers":-1}
        {"id":3,"cmd":"chat","system":"...","text":"...","images":["<base64 png>"],"max_tokens":1024,
         "temperature":0.7,"seed":123,"stop":[]}
        {"id":4,"cmd":"unload"}  {"id":5,"cmd":"info"}
    回：{"id":N,"ok":true,"data":{...}} / {"id":N,"ok":false,"error":"..."}
        流式进度：{"event":"progress","id":N,"tokens":k,"tps":1.2,"elapsed":3.4}
    启动后先发一行 {"event":"ready",...}（含 llama_cpp 版本，装没装一眼看出）。
"""

from __future__ import annotations

import base64
import gc
import json
import os
import sys
import time
import traceback


def emit(obj: dict) -> None:
    try:
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


STATE = {
    "llama": None,
    "model_path": "",
    "mmproj_path": "",
    "n_ctx": 0,
    "n_gpu_layers": 0,
    "load_seconds": 0.0,
    "handler": "",
    "last": {},
    "note": "",
    "dev_name": "",
    "offloaded": "",
}


def _run_capture(fn):
    """跑 fn 的时候把 C 层打到 stderr 的内容抓下来：ggml 会在这儿打显卡名和上卡层数。"""
    import tempfile
    try:
        old = os.dup(2)
    except Exception:
        return fn(), ""
    try:
        with tempfile.TemporaryFile(mode="w+b") as tf:
            sys.stderr.flush()
            os.dup2(tf.fileno(), 2)
            try:
                out = fn()
            finally:
                try:
                    sys.stderr.flush()
                except Exception:
                    pass
                os.dup2(old, 2)
                try:
                    os.close(old)
                except Exception:
                    pass
            tf.seek(0)
            return out, tf.read().decode("utf-8", "replace")
    except Exception:
        try:
            os.dup2(old, 2)
        except Exception:
            pass
        return fn(), ""


def _parse_load_log(text: str) -> tuple:
    """从 ggml 的原生日志里摘出「显卡名」和「多少层上了卡」。"""
    dev, layers = "", ""
    for ln in (text or "").splitlines():
        if " = " in ln and any(k in ln for k in ("ggml_vulkan", "ggml_cuda", "ggml_hip", "ggml_sycl")):
            tail = ln.split(" = ", 1)[1].split(" | ")[0].strip()
            if tail and not dev:
                dev = tail
        low = ln.lower()
        if "layers" in low and ("offloaded" in low or "offloading" in low) and not layers:
            layers = ln.strip().split(" ", 1)[-1][:60]
    return dev, layers


def _parse_buffers(text: str) -> tuple:
    """摘出显存/内存缓冲：('Vulkan0 compute buffer size = 285.29 MiB', 'CPU compute buffer size = 4484.10 MiB')
    → (gpu_mib, cpu_mib)。CPU 那份很大就说明还有大量张量没上卡（视觉编码器是主要来源）。"""
    gpu_mib = cpu_mib = 0.0
    for ln in (text or "").splitlines():
        low = ln.lower()
        if "buffer size" not in low:
            continue
        try:
            num = float(ln.split("=")[-1].strip().split(" ")[0])
        except Exception:
            continue
        if "cpu compute buffer" in low:
            cpu_mib = max(cpu_mib, num)
        elif "compute buffer" in low or "kv buffer" in low:
            gpu_mib = max(gpu_mib, num)
        elif low.strip().startswith("cpu") and "buffer" in low:
            cpu_mib = max(cpu_mib, num)
        elif "buffer" in low:
            gpu_mib = max(gpu_mib, num)
    return gpu_mib, cpu_mib


def _llama_cpp():
    import llama_cpp  # noqa: F401
    return llama_cpp


def _make_vision_handler(mmproj: str):
    """按版本挑可用的多模态 chat handler（Qwen2.5-VL 优先，其次 Qwen2-VL，最后 LLaVA）。"""
    tried = []
    for name in ("Qwen25VLChatHandler", "Qwen2VLChatHandler", "Llava15ChatHandler", "LlavaChatHandler"):
        try:
            mod = __import__("llama_cpp.llama_chat_format", fromlist=[name])
            cls = getattr(mod, name, None)
            if cls is None:
                tried.append(f"{name}(无)")
                continue
            return cls(clip_model_path=mmproj), name
        except Exception as e:
            tried.append(f"{name}({type(e).__name__})")
    raise RuntimeError("这个版本的 llama-cpp-python 没有可用的多模态 handler：" + ", ".join(tried)
                       + "。请升级 llama-cpp-python（>=0.3.2）后再试")


def cmd_load(p: dict) -> dict:
    model = str(p.get("model") or "")
    mmproj = str(p.get("mmproj") or "")
    if not model or not os.path.exists(model):
        raise RuntimeError(f"GGUF 文件不存在：{model}")
    n_ctx = int(p.get("n_ctx") or 4096)
    n_gpu_layers = int(p.get("n_gpu_layers", -1) or -1)
    if STATE["llama"] is not None and STATE["model_path"] == model and STATE["mmproj_path"] == mmproj:
        return cmd_info()

    import llama_cpp
    from llama_cpp import Llama

    cmd_unload()
    handler, handler_name = (None, "纯文本")
    if mmproj:
        if not os.path.exists(mmproj):
            raise RuntimeError(f"视觉投影（mmproj）文件不存在：{mmproj}")
        handler, handler_name = _make_vision_handler(mmproj)

    t0 = time.time()
    kw = dict(model_path=model, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False,
              seed=-1, logits_all=False, embedding=False)
    if handler is not None:
        kw["chat_handler"] = handler
    kw["verbose"] = True      # 只在加载这一下开：为的是拿到「显卡名 / 多少层上卡」那几行日志
    import io
    buf = io.StringIO()
    old_err = sys.stderr
    sys.stderr = buf
    try:
        llm, log_text = _run_capture(lambda: Llama(**kw))
    finally:
        sys.stderr = old_err
    dev_name, offloaded = _parse_load_log((log_text or "") + "\n" + buf.getvalue())
    gpu_buf, cpu_buf = _parse_buffers((log_text or "") + "\n" + buf.getvalue())
    note = ""
    if _backend_tag() == "cpu" and n_gpu_layers != 0:
        note = ("这个 llama-cpp-python 是 CPU 版（没编 Vulkan/CUDA），层全在内存里跑。"
                "AMD 显卡点服务端面板的「装 Vulkan 版（AMD·免SDK）」，装完就上卡")
    elif offloaded and ("0/" in offloaded or " 0/" in offloaded):
        note = f"没有任何层上卡（{offloaded}），看驱动/显存"
    elif cpu_buf > 1000 or "unsupported operators by the backend" in (log_text or ""):
        extra = ("日志里还有「mmproj graph uses unsupported operators by the backend」——"
                 "视觉编码器有一部分算子这个 Vulkan 后端不支持，会整段退回 CPU 算"
                 if "unsupported operators by the backend" in (log_text or "") else "")
        note = (f"模型主体在显卡，但还有 {cpu_buf/1024:.1f} GB 张量在内存里算（显存缓冲 {gpu_buf/1024:.1f} GB）。"
                + extra + "。带图做提示词优化时这一步要几十秒到两分钟，任务管理器里 GPU 利用率自然很低。"
                "缓解：① 节点上「参考图最大像素」设 0.4（百万像素，默认值，发过来之前就缩小了）；"
                "② 优化时不接图片，只用文生图 GGUF；③ 换 f16 版 mmproj（bf16 在 Vulkan 上同样容易回落 CPU）")
    STATE.update({"llama": llm, "model_path": model, "mmproj_path": mmproj, "n_ctx": n_ctx,
                  "n_gpu_layers": n_gpu_layers, "load_seconds": round(time.time() - t0, 2),
                  "handler": handler_name, "note": note, "dev_name": dev_name,
                  "offloaded": offloaded,
                  "gpu_buf_mib": round(gpu_buf, 1), "cpu_buf_mib": round(cpu_buf, 1)})
    try:
        n_ctx_real = int(llm.n_ctx())
    except Exception:
        n_ctx_real = n_ctx
    STATE["n_ctx"] = n_ctx_real
    info = cmd_info()
    emit({"event": "loaded", **(info.get("data") or {})})
    return info


def cmd_unload(p: dict | None = None) -> dict:
    had = STATE["llama"] is not None
    STATE.update({"llama": None, "model_path": "", "mmproj_path": "", "n_ctx": 0, "load_seconds": 0.0,
                  "handler": "", "last": {}, "note": ""})
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    return {"data": {"unloaded": had}}


def _backend_tag() -> str:
    """这个 llama-cpp-python 是拿什么后端编译的：vulkan / cuda / hip / metal / cpu。"""
    try:
        import llama_cpp
        lib = os.path.join(os.path.dirname(llama_cpp.__file__), "lib")
        names = [n.lower() for n in os.listdir(lib)] if os.path.isdir(lib) else []
        has = lambda k: any(k in n for n in names)  # noqa: E731
        if has("vulkan"):
            return "vulkan"
        if has("ggml-hip") or has("rocblas"):
            return "hip"
        if has("ggml-cuda") or has("cublas"):
            return "cuda"
        if has("metal"):
            return "metal"
        return "cpu"
    except Exception:
        return "?"


def cmd_info() -> dict:
    ver = ""
    try:
        ver = __import__("llama_cpp").__version__
    except Exception:
        ver = "未安装"
    backend = _backend_tag()
    gpu = "—"
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    if gpu == "—" and STATE.get("dev_name"):
        gpu = STATE["dev_name"]
    if gpu == "—" and backend in ("vulkan", "hip", "metal"):
        gpu = {"vulkan": "Vulkan 显卡（AMD/Intel）", "hip": "ROCm 显卡", "metal": "Apple Metal"}[backend]
    return {"data": {
        "loaded": STATE["llama"] is not None,
        "file": os.path.basename(STATE["model_path"]) if STATE["model_path"] else "",
        "model_path": STATE["model_path"],
        "mmproj": os.path.basename(STATE["mmproj_path"]) if STATE["mmproj_path"] else "",
        "handler": STATE["handler"],
        "n_ctx": STATE["n_ctx"],
        "n_gpu_layers": STATE["n_gpu_layers"],
        "note": STATE["note"],
        "load_seconds": STATE["load_seconds"],
        "llama_cpp": ver,
        "backend": backend,
        "device_name": gpu,
        "offloaded": STATE.get("offloaded") or "",
        "gpu_buf_mib": STATE.get("gpu_buf_mib"),
        "cpu_buf_mib": STATE.get("cpu_buf_mib"),
        "last": STATE["last"],
        "python": sys.executable,
    }}


def _shrink_b64(b64: str, max_side: int) -> tuple:
    """把客户端的原图按长边缩到 max_side 再编码（PNG/JPEG 都吃）。
    AMD + Vulkan 上视觉编码器的 f32 注意力算子不被后端支持，会整段回落 CPU，
    耗时会随图片 token 数平方增长（1408×1920 实测 148 秒），缩到 768 长边后只剩几秒。
    返回 (新 base64, 说明文本)；PIL 不可用或本来就够小时原样返回。"""
    if max_side <= 0 or not b64:
        return b64, ""
    try:
        import io
        import base64 as _b64
        from PIL import Image
        raw = _b64.b64decode(b64)
        im = Image.open(io.BytesIO(raw))
        w, h = im.size
        if max(w, h) <= max_side:
            return b64, f"{w}×{h} 未超上限"
        k = max_side / float(max(w, h))
        nw, nh = max(1, int(w * k)), max(1, int(h * k))
        im = im.convert("RGB") if im.mode not in ("RGB", "RGBA") else im
        im = im.resize((nw, nh), Image.LANCZOS)
        out = io.BytesIO()
        im.save(out, "PNG")
        nb = _b64.b64encode(out.getvalue()).decode("ascii")
        return nb, f"{w}×{h} → {nw}×{nh}（{len(raw)//1024}KB → {len(out.getvalue())//1024}KB）"
    except Exception as e:
        return b64, f"缩图失败，用原图：{type(e).__name__}: {e}"


def cmd_chat(p: dict) -> dict:
    if STATE["llama"] is None:
        raise RuntimeError("还没有加载 GGUF 模型（服务端面板点「加载 GGUF 到显存」，或让节点首次调用自动加载）")
    text = str(p.get("text") or "")
    system = str(p.get("system") or "")
    images = p.get("images") or []
    max_tokens = int(p.get("max_tokens") or 1024)
    temperature = float(p.get("temperature") or 0.7)
    seed = p.get("seed")

    content = text
    if images:
        if not STATE["mmproj_path"]:
            raise RuntimeError("这次带了参考图，但这个模型没有配「视觉投影 mmproj」，只能做纯文本。"
                               "在服务端面板把 mmproj 也选上，或在节点里取消图片连接。")
        content = []
        if text:
            content.append({"type": "text", "text": text})
        max_side = int(p.get("image_max_side", 768) or 0)
        for b in images:
            b = str(b or "")
            if not b:
                continue
            b, how = _shrink_b64(b, max_side)
            if how:
                emit({"event": "image", "msg": f"参考图 {how}"})
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + b}})

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    t0 = time.time()
    chunks = []
    n = 0
    tps = 0.0
    req_id = p.get("_id")
    kw = dict(messages=messages, stream=True, max_tokens=max_tokens, temperature=temperature)
    if seed is not None:
        kw["seed"] = int(seed)
    # Qwen3 这类带思考的模板：能关就关（关不掉就靠节点那边抠 JSON）
    if p.get("no_thinking", True):
        kw["chat_template_kwargs"] = {"enable_thinking": False}
    # 能上 JSON 语法约束就上：模型就不会写思考段、也不会吐半截不是 JSON 的东西
    if p.get("json_mode", True):
        kw["response_format"] = {"type": "json_object"}
    while True:
        try:
            it = STATE["llama"].create_chat_completion(**kw)
            break
        except TypeError as e:
            if "chat_template_kwargs" in kw:
                kw.pop("chat_template_kwargs", None)
                continue
            if "response_format" in kw:
                kw.pop("response_format", None)
                continue
            raise e
        except ValueError as e:
            # 部分版本不支持 response_format 的 grammar
            if "response_format" in kw and ("response_format" in str(e) or "grammar" in str(e).lower()):
                kw.pop("response_format", None)
                continue
            raise e
    for part in it:
        try:
            delta = (part.get("choices") or [{}])[0].get("delta") or {}
            piece = delta.get("content") or ""
        except Exception:
            piece = ""
        if piece:
            chunks.append(piece)
            n += 1
            if n % 8 == 0:
                el = max(1e-6, time.time() - t0)
                tps = n / el
                emit({"event": "progress", "id": req_id, "tokens": n,
                      "tps": round(tps, 2), "elapsed": round(el, 1)})
    out = "".join(chunks)
    elapsed = round(time.time() - t0, 2)
    tps = round(n / elapsed, 2) if elapsed > 0 else 0.0
    STATE["last"] = {"tokens": n, "seconds": elapsed, "tps": tps,
                     "images": len(images), "at": time.strftime("%H:%M:%S")}
    return {"data": {"text": out, "tokens": n, "seconds": elapsed, "tps": tps,
                     "images": len(images)}}


def _perf_tps(text: str) -> tuple:
    """从 llama.cpp 的 llama_perf_context_print 行里抠出「预填 tokens/s」和「生成 tokens/s」。
    行长这样：prompt eval time = 381.66 ms / 600 tokens ( 0.64 ms per token, 1572.10 tokens per second)"""
    import re
    pp = 0.0
    gen = 0.0
    for ln in (text or "").splitlines():
        low = ln.lower()
        if "tokens per second" not in low:
            continue
        m = re.search(r"(\d+(?:\.\d+)?)\s*tokens per second", low)
        if not m:
            continue
        num = float(m.group(1))
        if "prompt eval time" in low:
            pp = num
        elif "eval time" in low:
            gen = num
    return pp, gen


def cmd_bench(p: dict) -> dict:
    """跑一次短测试，把「预填(PP) / 生成(gen)」的速度量出来。
    这两个数字一出来就能判定到底是谁在算：CPU 上生成通常 <10 tokens/s，
    上了卡通常 30+；Vulkan 在缺 cooperative matrix 的卡上则是「生成快、预填极慢」。"""
    if STATE["llama"] is None:
        raise RuntimeError("先加载 GGUF 模型再跑分（面板点「加载 GGUF 到显存」）")
    pt = max(64, int(p.get("prompt_tokens") or 512))
    gt = max(16, int(p.get("gen_tokens") or 64))
    filler = ("The quick brown fox jumps over the lazy dog. " * (pt // 9 + 4))[: pt * 6]
    t0 = time.time()
    out, log = _run_capture(lambda: STATE["llama"](filler, max_tokens=gt, temperature=0.0))
    sec = max(0.001, time.time() - t0)
    pp, gen = _perf_tps(log)
    dev, offloaded = _parse_load_log(log)
    backend = _backend_tag()
    if backend == "cpu":
        verdict = "CPU 版 llama：所有层都在内存里算，必须换成 Vulkan/HIP 版"
    elif pp and gen and gen >= 20 and pp < 40:
        verdict = "显卡在算，但预填(PP)很慢：这张卡缺 cooperative matrix，长提示/带图会很拖"
    elif gen >= 20:
        verdict = "显卡在算（生成速度正常）"
    elif gen:
        verdict = "生成速度偏低：可能只有部分层上卡，或显存不够被挤回内存"
    else:
        verdict = "没拿到 perf 行，无法判定（看下面的原始日志）"
    return {"data": {
        "pp_tps": round(pp, 2), "gen_tps": round(gen, 2),
        "prompt_chars": len(filler), "gen_tokens": gt, "seconds": round(sec, 2),
        "backend": backend, "device_name": dev or STATE.get("dev_name") or "",
        "offloaded": offloaded or STATE.get("offloaded") or "",
        "model": os.path.basename(STATE["model_path"]), "verdict": verdict,
        "log": [x for x in (log or "").splitlines() if "perf" in x.lower() or "offload" in x.lower()][-12:],
        "tail": (out or {}).get("choices", [{}])[0].get("text", "")[:80] if isinstance(out, dict) else "",
    }}


HANDLERS = {"ping": lambda p: {"data": {"pong": True}}, "load": cmd_load, "chat": cmd_chat,
            "unload": cmd_unload, "info": cmd_info, "bench": cmd_bench}


def main() -> int:
    try:
        emit({"event": "ready", "pid": os.getpid(), "python": sys.executable,
              "llama_cpp": __import__("llama_cpp").__version__})
    except Exception as e:
        emit({"event": "ready", "pid": os.getpid(), "python": sys.executable,
              "llama_cpp": "未安装", "error": f"{type(e).__name__}: {e}"})
        # 不退出：仍可响应 info，让面板显示「未安装 llama-cpp-python」
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        rid = req.get("id")
        cmd = str(req.get("cmd") or "")
        params = dict(req)
        params["_id"] = rid
        emit({"event": "busy", "id": rid, "cmd": cmd})
        try:
            fn = HANDLERS.get(cmd)
            if fn is None:
                raise RuntimeError(f"未知命令 {cmd}")
            res = fn(params)
            emit({"id": rid, "ok": True, "data": (res or {}).get("data", res)})
        except Exception as e:
            tb = traceback.format_exc()
            emit({"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}", "trace": tb[-800:]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
