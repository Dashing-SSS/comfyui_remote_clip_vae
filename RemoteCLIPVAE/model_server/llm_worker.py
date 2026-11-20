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
}


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
    llm = Llama(**kw)
    STATE.update({"llama": llm, "model_path": model, "mmproj_path": mmproj, "n_ctx": n_ctx,
                  "n_gpu_layers": n_gpu_layers, "load_seconds": round(time.time() - t0, 2),
                  "handler": handler_name})
    try:
        n_ctx_real = int(llm.n_ctx())
    except Exception:
        n_ctx_real = n_ctx
    STATE["n_ctx"] = n_ctx_real
    info = cmd_info()
    emit({"event": "loaded", **(info.get("data") or {})})
    return info


def cmd_unload() -> dict:
    had = STATE["llama"] is not None
    STATE.update({"llama": None, "model_path": "", "mmproj_path": "", "n_ctx": 0, "load_seconds": 0.0,
                  "handler": "", "last": {}})
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    return {"data": {"unloaded": had}}


def cmd_info() -> dict:
    ver = ""
    try:
        ver = __import__("llama_cpp").__version__
    except Exception:
        ver = "未安装"
    gpu = "—"
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return {"data": {
        "loaded": STATE["llama"] is not None,
        "file": os.path.basename(STATE["model_path"]) if STATE["model_path"] else "",
        "model_path": STATE["model_path"],
        "mmproj": os.path.basename(STATE["mmproj_path"]) if STATE["mmproj_path"] else "",
        "handler": STATE["handler"],
        "n_ctx": STATE["n_ctx"],
        "n_gpu_layers": STATE["n_gpu_layers"],
        "load_seconds": STATE["load_seconds"],
        "llama_cpp": ver,
        "device_name": gpu,
        "last": STATE["last"],
        "python": sys.executable,
    }}


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
        for b in images:
            b = str(b or "")
            if not b:
                continue
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


HANDLERS = {"ping": lambda p: {"data": {"pong": True}}, "load": cmd_load, "chat": cmd_chat,
            "unload": cmd_unload, "info": cmd_info}


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
