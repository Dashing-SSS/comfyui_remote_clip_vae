# -*- coding: utf-8 -*-
"""
llm.py —— 服务端侧的「远端 ComfyUI LLM」引擎。

做的事：
  · 找 GGUF：config.json 的 llm_model_dir，没配就用 <ComfyUI>/models/LLM、models/llm、models/llm_gguf
  · 找 python：config.json 的 llm_python → model_server/_python.txt → <ComfyUI>/.venv/Scripts/python.exe
               → <ComfyUI>/python_embeded/python.exe（必须是装了 torch 的那个，和 CLIP 同一个）
  · 拉起 llm_worker.py 子进程（llama-cpp-python 装在那个 python 里），stdin/stdout 一行一个 JSON
  · 缓存「最近一次推理」的实时指标，给 /health、/llm_status、面板显示

对外（backend 调）：
    llm.list_models()           -> {"llm": {...}, "llm_vision": {...}}
    llm.chat(...)               -> {"text","tokens","tps",...}
    llm.load(model, mmproj)     -> 状态
    llm.unload()                -> 状态
    llm.status()                -> 给面板/节点的实时信息
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

GGUF_DIRS = ("LLM", "llm", "llm_gguf", "LLM_gguf", "gguf", "text_encoders_llm")
_HERE = os.path.dirname(os.path.abspath(__file__))


def _meipass_dir() -> str:
    """PyInstaller 打包后 llm_worker.py 被解到 sys._MEIPASS/model_server/。"""
    base = getattr(sys, "_MEIPASS", "")
    if base:
        cand = os.path.join(base, "model_server", "llm_worker.py")
        if os.path.exists(cand):
            return base
    return ""


def worker_script() -> str:
    cands = [os.path.join(_HERE, "llm_worker.py")]
    mp = _meipass_dir()
    if mp:
        cands.append(os.path.join(mp, "model_server", "llm_worker.py"))
    for c in cands:
        if os.path.exists(c):
            return c
    return cands[0]


def _has_torch(py: str) -> bool:
    """轻量判断：这个 python 里能不能 import torch（别把 Microsoft Store 的 python 当 ComfyUI 的用）。"""
    try:
        r = subprocess.run([py, "-c", "import torch,sys;print(torch.__version__)"],
                           capture_output=True, timeout=40, text=True,
                           creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
        return r.returncode == 0
    except Exception:
        return False


class LLMEngine:
    def __init__(self, backend):
        self.backend = backend
        self._lock = threading.RLock()
        self.proc = None
        self._reader = None
        self._pending: dict[int, dict] = {}
        self._seq = 0
        self._stderr_tail: list[str] = []
        self._ready: dict = {}
        self.state = {"state": "未启动", "detail": "", "model": "", "mmproj": "", "n_ctx": 0,
                      "handler": "", "load_seconds": 0.0, "python": "", "llama_cpp": "",
                      "device_name": "", "offloaded": "", "backend": "", "note": "",
                      "gpu_buf_mib": None, "cpu_buf_mib": None,
                      "progress": None, "last": {}, "error": "", "pid": None}
        self._loaded_key = ("", "")

    # ------------------------------------------------------------------ #
    # 路径解析
    # ------------------------------------------------------------------ #
    def comfy_dir(self) -> str:
        return str(getattr(self.backend, "comfy_dir", "") or "")

    def _model_roots(self) -> list[str]:
        """模型根目录：ComfyUI 自己的 models，加上它旁边那些 *Shared* 的 models 目录
        （比如 F:\\Comfy-Desktop\\ComfyUI\\ComfyUI + F:\\Comfy-Desktop\\ComfyUI-Shared\\models，
        CLIP/VAE 就是从 Shared 里扫到的，GGUF 通常也在那）。向上找两层。"""
        cd = self.comfy_dir()
        roots = []
        # 面板里配的「模型根目录」（可多个）：这些 models 目录下的 LLM / gguf 文件夹都算
        cfg = getattr(self.backend, "cfg", {}) or {}
        for r in (cfg.get("model_roots") or []):
            r = str(r or "").strip()
            if r and os.path.isdir(r):
                roots.append(os.path.abspath(os.path.expanduser(r)))
        if cd:
            roots.append(os.path.join(cd, "models"))
            parent = cd.rstrip("\\/")
            for _lv in range(2):
                parent = os.path.dirname(parent)
                if not parent or parent == os.path.dirname(parent):
                    break
                try:
                    for name in sorted(os.listdir(parent)):
                        p = os.path.join(parent, name)
                        if os.path.isdir(p) and os.path.isdir(os.path.join(p, "models")):
                            roots.append(os.path.join(p, "models"))
                except Exception:
                    pass
        uniq, seen = [], set()
        for r in roots:
            r = os.path.abspath(r)
            k = os.path.normcase(r)
            if k not in seen:
                seen.add(k)
                uniq.append(r)
        return uniq

    def llm_dirs(self) -> list[str]:
        cfg = getattr(self.backend, "cfg", {}) or {}
        d = str(cfg.get("llm_model_dir") or "").strip()
        if d:
            return [os.path.abspath(os.path.expanduser(d))]
        dirs = []
        seen = set()
        for root in self._model_roots():
            for sub in GGUF_DIRS:
                p = os.path.join(root, sub)
                if os.path.isdir(p) and os.path.normcase(p) not in seen:
                    seen.add(os.path.normcase(p))
                    dirs.append(p)
        return dirs

    def list_models(self, ttl: float = 3.0) -> dict:
        hit = getattr(self, "_models_cache", None)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
        out = {"llm": {}, "llm_vision": {}}
        for d in self.llm_dirs():
            if not os.path.isdir(d):
                continue
            for dirpath, _dirs, files in os.walk(d):
                if dirpath[len(d):].count(os.sep) > 2:
                    continue
                for fn in files:
                    if not fn.lower().endswith(".gguf") or fn.startswith("put_"):
                        continue
                    low = fn.lower()
                    rel = os.path.relpath(os.path.join(dirpath, fn), d).replace(os.sep, "/")
                    kind = "llm_vision" if ("mmproj" in low or "vision" in low or "vit" in low) else "llm"
                    out[kind][rel] = {"files": [rel], "path": os.path.join(dirpath, fn),
                                      "dir": d, "exists": True,
                                      "size_gb": round(os.path.getsize(os.path.join(dirpath, fn)) / 2**30, 2)}
        self._models_cache = (time.time(), out)
        return out

    def resolve(self, name: str, kind: str = "llm") -> str:
        """把节点里填的名字变成绝对路径：支持相对路径、纯文件名、唯一子串。"""
        name = str(name or "").strip().replace("\\", "/")
        if not name or name.startswith("（") or name.startswith("("):
            raise RuntimeError("没有指定 GGUF 模型名")
        if os.path.isabs(name) and os.path.exists(name):
            return name
        models = self.list_models().get(kind) or {}
        if name in models:
            return models[name]["path"]
        base = os.path.basename(name).lower()
        hits = [v["path"] for k, v in models.items() if os.path.basename(k).lower() == base]
        if not hits:
            hits = [v["path"] for k, v in models.items() if base and base in k.lower()]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise RuntimeError(f"模型名「{name}」匹配到多个 GGUF，请写全相对路径："
                               + ", ".join(os.path.basename(h) for h in hits[:5]))
        dirs = ", ".join(self.llm_dirs())
        raise RuntimeError(f"在服务端找不到 GGUF「{name}」。服务端已扫描的目录：{dirs}；"
                           f"可用模型：{', '.join(sorted(models.keys()))[:400] or '（空）'}")

    def python(self) -> str:
        cfg = getattr(self.backend, "cfg", {}) or {}
        cands = []
        if cfg.get("llm_python"):
            cands.append(str(cfg["llm_python"]))
        rec = os.path.join(os.path.dirname(os.path.abspath(getattr(self.backend, "config_dir", _HERE))),
                           "_python.txt")
        for path in (rec, os.path.join(_HERE, "_python.txt")):
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    cands.append(f.read().strip())
            except Exception:
                pass
        cd = self.comfy_dir()
        if cd:
            cands += [os.path.join(cd, ".venv", "Scripts", "python.exe"),
                      os.path.join(cd, ".venv", "bin", "python"),
                      os.path.join(cd, "python_embeded", "python.exe"),
                      os.path.join(os.path.dirname(cd), "python_embeded", "python.exe")]
        frozen = bool(getattr(sys, "frozen", False))
        if not frozen:
            cands.append(sys.executable)
        for py in cands:
            py = py.strip().strip('"')
            if not py or not os.path.exists(py):
                continue
            if "WindowsApps" in py:  # Microsoft Store 的 python 3.13 没 torch
                continue
            if not self._torch_ok(py):
                continue
            return py
        raise RuntimeError(
            "找不到能用的服务端 Python。LLM 必须跑在【服务端 ComfyUI 的 Python】里（它自带 torch/CUDA）。\n"
            f"COMfyUI 目录：{cd or '（未配置）'}\n"
            "请在服务端面板的「本机 ComfyUI 的 Python」里选好（或 config.json 里加 llm_python）。"
        )

    def _torch_ok(self, py: str) -> bool:
        key = ("torchok", py)
        hit = self.state.get("_torch_cache", {}).get(key)
        if hit is not None:
            return hit
        ok = _has_torch(py)
        self.state.setdefault("_torch_cache", {})[key] = ok
        return ok

    # ------------------------------------------------------------------ #
    # 子进程
    # ------------------------------------------------------------------ #
    def _spawn(self):
        py = self.python()
        script = worker_script()
        if not os.path.exists(script):
            raise RuntimeError(f"找不到 llm_worker.py：{script}")
        kw = {}
        if os.name == "nt":
            kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)
        proc = subprocess.Popen([py, "-u", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                                bufsize=1, cwd=os.path.dirname(script), env=env, **kw)
        self.proc = proc
        self.state["python"] = py
        self.state["pid"] = proc.pid
        self.state["state"] = "进程已启动"
        self._reader = threading.Thread(target=self._read_loop, args=(proc,), daemon=True, name="llm-reader")
        self._reader.start()
        threading.Thread(target=self._err_loop, args=(proc,), daemon=True, name="llm-stderr").start()
        # 等 ready
        t0 = time.time()
        while time.time() - t0 < 20:
            if self._ready:
                break
            if proc.poll() is not None:
                break
            time.sleep(0.1)
        if self._ready.get("llama_cpp") and self._ready.get("llama_cpp") != "未安装":
            self.state["llama_cpp"] = str(self._ready.get("llama_cpp"))
        elif self._ready:
            self.state["llama_cpp"] = "未安装"
            self.state["detail"] = ("服务端 Python 里没有 llama-cpp-python。点面板「安装 llama-cpp-python」"
                                    "（装到 " + py + "）")
        return proc

    def _err_loop(self, proc):
        try:
            for line in proc.stderr:
                line = line.rstrip()
                if not line:
                    continue
                self._stderr_tail.append(line)
                del self._stderr_tail[:-40]
                if len(self._stderr_tail) > 40:
                    self._stderr_tail = self._stderr_tail[-40:]
                print("[LLM worker]", line, flush=True)
        except Exception:
            pass

    def _read_loop(self, proc):
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except Exception:
                    continue
                if msg.get("event") == "ready":
                    self._ready.update(msg)
                    continue
                if msg.get("event") == "loaded":
                    self.state.update({"model": os.path.basename(str(msg.get("model_path") or "")),
                                       "mmproj": msg.get("mmproj") or "",
                                       "n_ctx": msg.get("n_ctx") or 0,
                                       "handler": msg.get("handler") or "",
                                       "load_seconds": msg.get("load_seconds") or 0.0,
                                       "device_name": msg.get("device_name") or "",
                                       "offloaded": msg.get("offloaded") or "",
                                       "gpu_buf_mib": msg.get("gpu_buf_mib"),
                                       "cpu_buf_mib": msg.get("cpu_buf_mib"),
                                       "backend": msg.get("backend") or self.state.get("backend") or "",
                                       "note": msg.get("note") or "",
                                       "state": "已加载", "progress": None})
                    continue
                if msg.get("event") == "progress":
                    rid = msg.get("id")
                    if rid is not None and rid == self._seq:
                        self.state["progress"] = {"tokens": msg.get("tokens"), "tps": msg.get("tps"),
                                                  "elapsed": msg.get("elapsed")}
                    continue
                if msg.get("event") == "busy":
                    continue
                if msg.get("event") == "image":
                    self.state["image_note"] = msg.get("msg") or ""
                    continue
                rid = msg.get("id")
                slot = self._pending.get(rid)
                if slot is not None:
                    slot["resp"] = msg
                    slot["ev"].set()
        except Exception:
            pass
        finally:
            for slot in list(self._pending.values()):
                if slot.get("resp") is None:
                    slot["resp"] = {"ok": False, "error": "LLM 子进程已退出"}
                slot["ev"].set()
            self.state["state"] = "未启动"

    def _call(self, cmd: str, params: dict | None = None, timeout: float | None = None) -> dict:
        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                self._ready.clear()
                self._stderr_tail.clear()
                self._spawn()
            self._seq += 1
            rid = self._seq
            slot = {"ev": threading.Event(), "resp": None}
            self._pending[rid] = slot
            payload = {"id": rid, "cmd": cmd}
            payload.update(params or {})
            try:
                self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                self.proc.stdin.flush()
            except Exception as e:
                self._pending.pop(rid, None)
                raise RuntimeError(f"LLM 子进程写不进去（可能已退出）：{e}") from None
            got = slot["ev"].wait(timeout)  # timeout=None → 一直等，不砍生成
            self._pending.pop(rid, None)
            if not got:
                self.state["progress"] = None
                raise RuntimeError(f"LLM 子进程 {cmd} 超时（{timeout or 0:.0f}s）。面板「服务端日志」里能看到它卡在哪。")
            resp = slot["resp"] or {}
            if not resp.get("ok"):
                msg = str(resp.get("error") or "未知错误")
                if "No module named 'llama_cpp'" in msg or "llama_cpp" in msg and "ModuleNotFound" in msg:
                    msg = ("服务端 Python 里没有 llama-cpp-python。面板点「安装 llama-cpp-python」"
                           "（装到 " + self.state.get("python", "ComfyUI 的 python") + "）后重试。")
                self.state["error"] = msg
                raise RuntimeError(msg)
            self.state["error"] = ""
            return resp.get("data") or {}

    # ------------------------------------------------------------------ #
    # 对外
    # ------------------------------------------------------------------ #
    def load(self, model: str, mmproj: str = "", n_ctx: int = 4096, n_gpu_layers: int = -1) -> dict:
        path = self.resolve(model, "llm")
        mpath = self.resolve(mmproj, "llm_vision") if mmproj else ""
        self.state.update({"state": "加载中", "progress": None,
                           "model": os.path.basename(path), "mmproj": os.path.basename(mpath)})
        t0 = time.time()
        data = self._call("load", {"model": path, "mmproj": mpath,
                                   "n_ctx": int(n_ctx), "n_gpu_layers": int(n_gpu_layers)}, timeout=1800)
        self._loaded_key = (path, mpath)
        self.state.update({"state": "已加载", "model": os.path.basename(path),
                           "mmproj": os.path.basename(mpath),
                           "n_ctx": data.get("n_ctx") or 0,
                           "handler": data.get("handler") or "",
                           "load_seconds": data.get("load_seconds") or round(time.time() - t0, 2),
                           "llama_cpp": data.get("llama_cpp") or self.state.get("llama_cpp"),
                           "device_name": data.get("device_name") or self.state.get("device_name"),
                           "backend": data.get("backend") or self.state.get("backend") or "",
                           "note": data.get("note") or "",
                           "progress": None})
        return self.status()

    def chat(self, model: str, text: str, system: str = "", images: list | None = None,
             max_tokens: int = 1024, temperature: float = 0.7, seed=None,
             mmproj: str = "", n_ctx: int = 4096, timeout: float | None = None,
             auto_load: bool = True) -> dict:
        if auto_load and model:
            path = self.resolve(model, "llm")
            mpath = self.resolve(mmproj, "llm_vision") if mmproj else ""
            if (path, mpath) != self._loaded_key:
                self.load(model, mmproj, n_ctx=n_ctx)
        self.state.update({"state": "生成中", "progress": {"tokens": 0, "tps": 0, "elapsed": 0}})
        params = {"system": system, "text": text, "images": images or [],
                  "image_max_side": int((self.backend.cfg or {}).get("llm_image_max_side", 768) or 0),
                  "max_tokens": int(max_tokens), "temperature": float(temperature)}
        if seed is not None:
            params["seed"] = int(seed)
        t0 = time.time()
        data = self._call("chat", params, timeout=timeout)
        self.state.update({"state": "空闲", "progress": None, "last": data})
        data["engine_seconds"] = round(time.time() - t0, 2)
        return data

    def _kill(self, why: str = ""):
        """把 worker 进程直接干掉：光让 worker 自己 del 掉模型，进程里的 CUDA 上下文
        仍然占着一块显存（nvidia-smi 会显示没退），杀进程才彻底还回去。"""
        p = self.proc
        self.proc = None
        self._ready.clear()
        if p is None:
            return
        try:
            if p.poll() is None:
                p.kill()
            p.wait(timeout=10)
        except Exception:
            pass
        try:
            for s in (p.stdin, p.stdout, p.stderr):
                if s:
                    s.close()
        except Exception:
            pass
        self._pending.clear()
        self.state.update({"state": "未加载", "progress": None, "pid": None})
        if why:
            print(f"[LLM] 已结束 worker 进程（{why}），显存随之释放", flush=True)

    def unload(self) -> dict:
        try:
            self._call("unload", {}, timeout=60)
        except Exception as e:
            self.state["error"] = str(e)
        try:
            self._kill("卸载模型")
        except Exception:
            pass
        self._loaded_key = ("", "")
        self.state.update({"state": "未加载", "model": "", "mmproj": "", "n_ctx": 0,
                           "handler": "", "last": {}, "progress": None, "note": ""})
        return self.status()

    def info(self) -> dict:
        if self.proc is None or self.proc.poll() is not None:
            return {"loaded": False, "llama_cpp": "未启动"}
        try:
            return self._call("info", {}, timeout=30)
        except Exception as e:
            return {"loaded": False, "error": str(e)}

    def env(self) -> dict:
        """面板「环境诊断」：看清这个 python 里装的是哪版 llama、有没有 CPU 版残留。"""
        return llama_env(self.python())

    def uninstall(self) -> dict:
        """面板「卸载 llama-cpp-python」：先杀掉 worker（否则 pyd 被占用删不掉），再卸干净。"""
        try:
            self.unload()
        except Exception:
            pass
        self._kill("卸载 llama-cpp-python")
        r = uninstall_llama(self.python(), _null_log)
        self.state.update({"llama_cpp": "" if r.get("ok") else self.state.get("llama_cpp", "")})
        return r

    def bench(self, prompt_tokens: int = 512, gen_tokens: int = 64, timeout: float = 900) -> dict:
        """真跑一次：把「预填 PP」和「生成」两段的速度量出来，判断到底谁在算。"""
        return self._call("bench", {"prompt_tokens": int(prompt_tokens),
                                    "gen_tokens": int(gen_tokens)}, timeout=timeout)

    def status(self, light: bool = False) -> dict:
        st = dict(self.state)
        st.pop("_torch_cache", None)
        if light:
            # 面板每 3 秒轮询一次：不扫盘（共享盘上 stat 上千个文件会拖到几秒，直接把 /health 拖超时）
            st["models"] = {"llm": {}, "llm_vision": {}}
            st["dirs"] = []
            st["running"] = self.proc is not None and self.proc.poll() is None
            st["stderr"] = []
            return st
        try:
            st["models"] = self.list_models()
            st["dirs"] = self.llm_dirs()
        except Exception as e:
            st["models"] = {"llm": {}, "llm_vision": {}}
            st["dirs"] = self.llm_dirs()
            st["models_error"] = str(e)
        st["running"] = self.proc is not None and self.proc.poll() is None
        st["stderr"] = list(self._stderr_tail)[-8:]
        return st


class MockLLM(LLMEngine):
    """--mock 联调模式用：不拉子进程、不加载 GGUF，固定返回一段 JSON。"""

    def __init__(self, backend):
        super().__init__(backend)
        self.state.update({"state": "MOCK", "llama_cpp": "MOCK", "model": "mock_llm.gguf",
                           "handler": "MOCK", "n_ctx": 4096, "device_name": "MOCK"})

    def list_models(self, ttl: float = 3.0) -> dict:
        return {"llm": {"mock_llm.gguf": {"files": ["mock_llm.gguf"], "exists": True, "size_gb": 0.0}},
                "llm_vision": {"mock_mmproj.gguf": {"files": ["mock_mmproj.gguf"], "exists": True,
                                                    "size_gb": 0.0}}}

    def resolve(self, name: str, kind: str = "llm") -> str:
        return f"/mock/{name or 'mock_llm.gguf'}"

    def env(self) -> dict:
        return {"what": "MOCK（不读真实环境）", "gpu_capable": True, "version": "MOCK", "dists": [],
                "libs": [], "leftover": [], "import_file": "/mock/llama_cpp"}

    def uninstall(self) -> dict:
        return {"ok": False, "error": "MOCK 模式下不会动真实环境"}

    def bench(self, prompt_tokens: int = 512, gen_tokens: int = 64, timeout: float = 900) -> dict:
        return {"pp_tps": 300.0, "gen_tps": 300.0, "backend": "MOCK", "device_name": "MOCK",
                "verdict": "MOCK 跑分", "log": [], "model": "mock_llm.gguf"}

    def load(self, model, mmproj="", n_ctx=4096, n_gpu_layers=-1):
        self.state.update({"state": "已加载", "model": os.path.basename(str(model) or "mock_llm.gguf"),
                           "mmproj": os.path.basename(str(mmproj)) if mmproj else "",
                           "load_seconds": 0.0})
        return self.status()

    def chat(self, model, text, system="", images=None, max_tokens=1024, temperature=0.7,
             seed=None, mmproj="", n_ctx=4096, timeout=600.0, auto_load=True):
        out = json.dumps({"prompt": f"MOCK 远端提示词（{len(images or [])} 张图）：{text[:40]}",
                          "negative_prompt": "模糊, 水印"}, ensure_ascii=False)
        self.state.update({"state": "MOCK", "last": {"tokens": 30, "seconds": 0.1, "tps": 300,
                                                     "images": len(images or []), "at": "MOCK"}})
        return {"text": out, "tokens": 30, "seconds": 0.1, "tps": 300, "images": len(images or [])}

    def unload(self):
        self.state.update({"state": "MOCK", "model": "", "mmproj": ""})
        return self.status()

    def info(self):
        return {"loaded": True, "llama_cpp": "MOCK"}

    def status(self, light: bool = False) -> dict:
        st = dict(self.state)
        st.pop("_torch_cache", None)
        st.update({"models": ({"llm": {}, "llm_vision": {}} if light else self.list_models()),
                   "dirs": ["<mock>"], "running": True, "stderr": []})
        return st


# --------------------------------------------------------------------- 安装 llama-cpp-python
# 思路：优先用 JamePeng 预编译 wheel（就是孤海「Qwen image 2.1 提示词优化」那套，Win + CUDA 直接可装，
# 不会走源码编译），失败再退回 abetlen 官方 index，最后退回 PyPI。
JAMEPENG_API = "https://api.github.com/repos/JamePeng/llama-cpp-python/releases?per_page=100"
JAMEPENG_DL = "https://github.com/JamePeng/llama-cpp-python/releases/download/"


def _no_window() -> int:
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _null_log(*a, **k):
    return None


def py_env_info(py: str) -> dict:
    """问目标 python：版本 tag（cp313）、torch 的 CUDA 版本。"""
    code = (
        "import sys,json\n"
        "d={'tag':'cp%d%d'%(sys.version_info[0],sys.version_info[1]),"
        "'ver':sys.version.split()[0],'cuda':'','torch':'','hip':'','exe':sys.executable}\n"
        "try:\n"
        "    import torch\n"
        "    d['cuda']=str(torch.version.cuda or '')\n"
        "    d['torch']=torch.__version__\n"
        "    d['hip']=str(getattr(torch.version,'hip',None) or '')\n"
        "    try:\n"
        "        d['devices']=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []\n"
        "    except Exception:\n"
        "        d['devices']=[]\n"
        "except Exception as e:\n"
        "    d['torch_error']=str(e)\n"
        "print(json.dumps(d,ensure_ascii=False))\n"
    )
    try:
        p = subprocess.run([py, "-c", code], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=120, creationflags=_no_window())
    except Exception as e:
        return {"tag": "", "error": f"探测 python 失败：{e}"}
    out = (p.stdout or "").strip().splitlines()
    try:
        return json.loads(out[-1]) if out else {"error": (p.stderr or "无输出")[:400]}
    except Exception:
        return {"tag": "", "error": ((p.stdout or "") + (p.stderr or ""))[:400]}


def llama_version(py: str) -> str:
    ok, _v, _e = llama_import_check(py)
    return _v if ok else ""


def llama_import_check(py: str) -> tuple:
    """返回 (ok, version, error_text)。import 失败时带上真实原因（缺 numpy / 缺 DLL …）。"""
    code = ("import llama_cpp\n"
            "print('VERSION', getattr(llama_cpp,'__version__','?'))\n")
    try:
        p = subprocess.run([py, "-c", code], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=120, creationflags=_no_window())
    except Exception as e:
        return (False, "", f"运行 import 检查失败：{e}")
    out = p.stdout or ""
    ver = ""
    for line in out.splitlines():
        if line.startswith("VERSION "):
            ver = line.split(" ", 1)[1].strip()
    if p.returncode == 0 and ver:
        return (True, ver, "")
    err = (p.stderr or out or "").strip()
    tail = err.splitlines()[-3:] if err else []
    return (False, "", " / ".join(tail)[:300] or f"退出码 {p.returncode}")


def find_jamepeng_wheel(py: str) -> dict:
    """按 <python 版本 tag> + <torch 的 cuXXX> 去 JamePeng 的 release 里找 win_amd64 wheel。"""
    import re
    import urllib.request

    info = py_env_info(py)
    tag = str(info.get("tag") or "")
    cuda = str(info.get("cuda") or "")
    hip = str(info.get("hip") or "")
    m = re.match(r"(\d+)\.(\d+)", cuda)
    if not tag:
        raise RuntimeError(f"探测不到目标 python 的版本（{info.get('error') or info}）")
    if hip and not m:
        raise RuntimeError(
            f"这个 python 的 torch 是 ROCm/HIP 版（torch {info.get('torch')}，hip {hip}），"
            "JamePeng 只发 CUDA 预编译 wheel，没有 HIP 版。AMD 卡请点「装 Vulkan 版（AMD·免SDK）」"
            "或「装 HIP 版（AMD·免SDK）」，走官方预编译 wheel，不用编译")
    if not m:
        raise RuntimeError("这个 python 里没有可用的 PyTorch CUDA（torch.version.cuda 为空）。"
                           "说明它不是跑 ComfyUI 的那个 python，或者装的是 CPU 版 torch。")
    backend = f"cu{m.group(1)}{m.group(2)}"
    want = f"-{backend}-win-".lower()
    suffix = f"-{tag}-{tag}-win_amd64.whl".lower()
    req = urllib.request.Request(JAMEPENG_API, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "RemoteCLIPVAE-server"})
    with urllib.request.urlopen(req, timeout=40) as r:
        releases = json.load(r)

    def ver_key(txt: str):
        nums = re.findall(r"\d+", txt.split("-")[0])
        return tuple(int(x) for x in nums) if nums else (0,)

    cands = []
    for rel in releases:
        if rel.get("draft") or rel.get("prerelease"):
            continue
        tag_name = str(rel.get("tag_name") or "")
        if want not in tag_name.lower():
            continue
        for a in rel.get("assets") or []:
            name = str(a.get("name") or "")
            url = str(a.get("browser_download_url") or "")
            if name.lower().endswith(suffix) and url.startswith(JAMEPENG_DL):
                cands.append((ver_key(tag_name), tag_name, name, url))
    if not cands:
        raise RuntimeError(f"JamePeng 发布页里没有匹配 {backend} + {tag} 的 Windows wheel")
    cands.sort(reverse=True)
    best = cands[0]
    return {"url": best[3], "name": best[2], "tag": best[1], "backend": backend,
            "python": tag, "cuda": cuda, "python_ver": info.get("ver")}


def _pip(py: str, args: list, log, timeout: int = 1500) -> int:
    return _pip_env(py, args, log, None, timeout)


def _pip_env(py: str, args: list, log, extra_env: dict | None = None, timeout: int = 1500) -> int:
    log("$ " + " ".join([os.path.basename(py)] + args))
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    for k, v in (extra_env or {}).items():
        env[k] = str(v)
    try:
        p = subprocess.Popen([py] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", bufsize=1,
                             env=env, creationflags=_no_window())
    except Exception as e:
        log(f"启动 pip 失败：{e}")
        return 1
    t0 = time.time()
    for line in p.stdout:  # type: ignore[union-attr]
        s = line.rstrip("\n")
        if s.strip():
            log(s)
        if time.time() - t0 > timeout:
            p.kill()
            log("pip 超时，已中止")
            return 1
    return p.wait()


def install_llama(py: str, log=print, wheel_url: str = "") -> dict:
    """把 llama-cpp-python 装进目标 python。返回 {ok, how, version, wheel, log}。"""
    lines: list[str] = []

    def say(s):
        lines.append(str(s))
        try:
            log(str(s))
        except Exception:
            pass

    old = llama_version(py)
    if old:
        say(f"这个 python 里已经有 llama-cpp-python {old}；下面强制重装一遍，装完再看版本")
    info = py_env_info(py)
    say(f"目标 python：{py}")
    say(f"  Python {info.get('ver') or '?'}（{info.get('tag') or '?'}）  torch {info.get('torch') or '?'}  "
        f"CUDA {info.get('cuda') or '（无）'}" + (f"  HIP {info.get('hip')}" if info.get("hip") else ""))
    hip = str(info.get("hip") or "")
    py_tag = str(info.get("tag") or "")

    wheels: list[tuple[str, str]] = []   # [(how, url)]
    if wheel_url:
        # 先核对 wheel 是不是给这个 python 的（cp314 的 wheel 装不上 cp313 的 python）
        import re as _re
        mt = _re.search(r"-(cp\d{2,3})-", os.path.basename(wheel_url))
        if mt and py_tag and mt.group(1).lower() != py_tag.lower():
            say(f"⚠ 这个 wheel 是 {mt.group(1)} 的，目标 python 是 {py_tag} —— 装不上，去下 {py_tag} 那一版")
            return {"ok": False, "version": "", "how": "", "wheel": os.path.basename(wheel_url),
                    "error": f"wheel 是 {mt.group(1)}，python 是 {py_tag}",
                    "python": py, "python_ver": info.get("ver"), "cuda": info.get("cuda"),
                    "filtered": [], "log": lines[-60:]}
        wheels.append(("本地/手动指定的 wheel", wheel_url))
    if hip:
        say("这个 python 的 torch 是 ROCm/HIP 版（AMD 卡）：CUDA 的 wheel 在这台机上只能拿 CPU 算。"
            "下面直接给 AMD 用官方预编译的 Vulkan / HIP-Radeon wheel（自带 GPU 后端，免 SDK、免编译）。")
        wheel_info = {}
        wheels.append(("官方预编译 Vulkan wheel（免 SDK）", "__preset__vulkan"))
        wheels.append(("官方预编译 HIP-Radeon wheel（免 SDK）", "__preset__hip-radeon"))
    else:
        try:
            w = find_jamepeng_wheel(py)
            say(f"匹配到预编译 wheel：{w['name']}  （{w['backend']} / {w['python']} / {w['tag']}）")
            wheels.append(("JamePeng 预编译 wheel", w["url"]))
            wheel_info = w
        except Exception as e:
            say(f"JamePeng 预编译 wheel 没匹配上：{e}")
            wheel_info = {}
    # 官方 index 兜底（按 torch 的 CUDA 版本猜最近的一档）
    import re
    m = re.match(r"(\d+)\.(\d+)", str(info.get("cuda") or ""))
    idx_tags = []
    if m:
        base = m.group(1) + m.group(2)
        idx_tags = [f"cu{base}", "cu124", "cu123", "cu122", "cu126", "cu125", "cu121"]
    elif not hip:
        if _is_amd(info):
            say("这台机是 AMD 显卡 + CPU 版 torch：优先试官方 Vulkan 预编译 wheel（不用装 SDK）")
            idx_tags = ["vulkan", "cu124", "cpu"]
            wheels.append(("官方预编译 Vulkan wheel（免 SDK）", "__preset__vulkan"))
        else:
            idx_tags = ["cu124", "cu122", "cpu"]
    for t in idx_tags:
        wheels.append((f"abetlen 官方 index（{t}）", f"__index__{t}"))

    # 先装运行依赖（numpy/jinja2/diskcache/typing-extensions）：已装过就是 no-op
    rc = _pip(py, ["-m", "pip", "install", "-U", "--no-input",
                   "numpy>=1.20.0", "typing-extensions>=4.5.0", "diskcache>=5.6.1", "jinja2>=2.11.3"],
              say, timeout=600)
    if rc != 0:
        say("运行依赖装失败（继续试 wheel，装完再查 import 错误）")

    ok_how = ""
    last_err = ""
    for how, target in wheels:
        say("=" * 60)
        say(f"尝试：{how}")
        if target.startswith("__preset__"):
            kind = target[len("__preset__"):]
            try:
                w = pick_preset_wheel(kind, py_tag)
                say(f"  选中 {w['name']}")
                local = fetch_wheel(w["url"], w["name"], _wheel_cache_dir(), say)
            except Exception as e:
                say(f"  这条不行：{e}")
                last_err = last_err or str(e)
                continue
            rc = _pip(py, ["-m", "pip", "install", "-U", "--force-reinstall", "--no-input", local], say)
        elif target.startswith("__index__"):
            tag = target[len("__index__"):]
            extra = ["--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/" + tag]
            # 只接受预编译 wheel；没有就快速失败，绝不走源码编译（Windows 上几乎必然失败）
            rc = _pip(py, ["-m", "pip", "install", "-U", "--force-reinstall", "--no-cache-dir",
                           "--no-input", "--only-binary=llama-cpp-python", "llama-cpp-python"] + extra, say)
            if tag == "cpu" and rc != 0:
                rc = _pip(py, ["-m", "pip", "install", "-U", "--force-reinstall",
                               "--prefer-binary", "--only-binary=llama-cpp-python", "--no-input",
                               "llama-cpp-python"], say)
        else:
            rc = _pip(py, ["-m", "pip", "install", "-U", "--force-reinstall", "--no-deps",
                           "--no-input", target], say)
        ok_imp, v, err = llama_import_check(py)
        if rc == 0 and ok_imp:
            be = llama_backend(py) if target.startswith("__preset__") else {}
            tail = ""
            if be:
                tail = f"，后端 {be.get('backend')}" + (f"，设备 {be.get('devices')}" if be.get("devices") else "")
            say(f"安装成功：llama-cpp-python {v}（{how}）{tail}")
            ok_how = how
            break
        if rc == 0:
            last_err = err
            say(f"pip 装完了，但 import llama_cpp 失败：{err}")
        else:
            last_err = last_err or f"pip 退出码 {rc}"
    ok_imp, v, err = llama_import_check(py)
    if not ok_imp and last_err:
        say(f"最后一次 import 失败原因：{last_err}")
    return {"ok": bool(ok_imp), "version": v, "how": ok_how,
            "wheel": (wheel_info or {}).get("name", ""), "error": ("" if ok_imp else last_err),
            "python": py, "python_ver": info.get("ver"), "cuda": info.get("cuda"),
            "log": lines[-80:]}


def llama_backend(py: str) -> dict:
    """看这个 python 里的 llama-cpp-python 到底是用什么后端编译的（cuda / vulkan / hip / cpu …）。
    做法：直接查 llama_cpp/lib 里带的原生 dll —— ggml-vulkan.dll 就说明是 Vulkan 版，
    再去 worker 里让它报一下实际用的设备名。"""
    code = r"""
import json, os, glob
d = {"version": "", "libs": [], "backend": "cpu", "devices": [], "error": ""}
try:
    import llama_cpp
    d["version"] = getattr(llama_cpp, "__version__", "")
    lib = os.path.join(os.path.dirname(llama_cpp.__file__), "lib")
    names = [os.path.basename(p).lower() for p in glob.glob(os.path.join(lib, "*"))]
    d["libs"] = sorted(set(n for n in names if n.endswith((".dll", ".so", ".dylib"))))
    has = lambda k: any(k in n for n in names)
    if has("vulkan"): d["backend"] = "vulkan"
    elif has("ggml-cuda") or has("cublas") or has("ggml-hip") or has("rocblas"): d["backend"] = "hip" if has("ggml-hip") else "cuda"
    elif has("metal"): d["backend"] = "metal"
except Exception as e:
    d["error"] = str(e)
try:
    import llama_cpp as L
    if d["backend"] in ("vulkan", "cuda", "hip", "metal"):
        if hasattr(L, "llama_max_devices"):
            n = int(L.llama_max_devices())
            d["devices"] = [str(i) for i in range(n)] if n else []
    d["offload"] = bool(L.llama_supports_gpu_offload()) if hasattr(L, "llama_supports_gpu_offload") else None
except Exception as e:
    if not d["error"]:
        d["error"] = str(e)
print(json.dumps(d))
"""
    try:
        p = subprocess.run([py, "-c", code], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=120, creationflags=_no_window())
    except Exception as e:
        return {"backend": "unknown", "error": f"探测失败：{e}"}
    out = (p.stdout or "").strip().splitlines()
    try:
        d = json.loads(out[-1]) if out else {"backend": "unknown", "error": (p.stderr or "")[:300]}
    except Exception:
        return {"backend": "unknown", "error": ((p.stdout or "") + (p.stderr or ""))[:300]}
    # 设备名是 ggml 在原生层打到 stderr 的（"ggml_vulkan: 0 = AMD Radeon RX 5700 XT (AMD) | ..."）
    devs = _parse_dev_names(p.stderr or "")
    if devs:
        d["devices"] = devs
    return d


def _parse_dev_names(text: str) -> list:
    names = []
    for ln in (text or "").splitlines():
        if " = " not in ln:
            continue
        if not any(k in ln for k in ("ggml_vulkan", "ggml_cuda", "ggml_hip", "ggml_sycl")):
            continue
        head = ln.split(": ", 1)[0] if ": " in ln else ln
        if "Found" in head:
            continue
        tail = ln.split(" = ", 1)[1].split(" | ")[0].strip()
        if tail and tail not in names:
            names.append(tail)
    return names


def llama_env(py: str, log=print) -> dict:
    """把「这个 python 里跟 llama 有关的真实状态」一次捞干净：
    装了几份、import 到的是哪份、lib 里带了哪些后端 dll、Vulkan/HIP 运行时在不在、
    有没有 CPU 版残留目录盖在上面。面板「环境诊断」就是它。"""
    code = r"""
import json, os, sys, glob, site
import importlib.metadata as md
d = {"python": sys.executable, "dists": [], "site": [], "import_file": "", "version": "",
     "libs": [], "leftover": [], "vulkan_runtime": "", "torch_lib_gpu": [], "env": {},
     "toolkit": "", "procs": []}
try:
    d["site"] = [p for p in site.getsitepackages() if os.path.isdir(p)]
except Exception:
    pass
for dist in md.distributions():
    nm = str(dist.metadata["Name"] or "")
    if nm.lower().replace("-", "_") in ("llama_cpp_python", "llama_cpp"):
        try:
            d["dists"].append({"name": nm, "version": str(dist.version),
                               "loc": str(dist.locate_file(""))})
        except Exception:
            d["dists"].append({"name": nm})
try:
    import llama_cpp
    d["import_file"] = llama_cpp.__file__
    d["version"] = getattr(llama_cpp, "__version__", "")
    lib = os.path.join(os.path.dirname(llama_cpp.__file__), "lib")
    d["libs"] = sorted(os.path.basename(p) for p in glob.glob(os.path.join(lib, "*"))
                       if os.path.isfile(p)) if os.path.isdir(lib) else []
except Exception as e:
    d["import_error"] = "%s: %s" % (type(e).__name__, e)
for sp in d["site"]:
    for p in glob.glob(os.path.join(sp, "llama_cpp*")) + glob.glob(os.path.join(sp, "llama-cpp*")):
        d["leftover"].append(p)
vk = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "vulkan-1.dll")
d["vulkan_runtime"] = vk if os.path.exists(vk) else ""
try:
    import torch
    tl = os.path.join(os.path.dirname(torch.__file__), "lib")
    names = [os.path.basename(p) for p in glob.glob(os.path.join(tl, "*"))]
    d["torch_lib_gpu"] = sorted(n for n in names
                                if any(k in n.lower() for k in ("hip", "rocblas", "amd", "cublas", "cudart")))
except Exception:
    pass
for k in ("GGML_VK_VISIBLE_DEVICES", "GGML_VULKAN_DEVICE", "VK_ICD_FILENAMES", "VK_LOADER_DEBUG",
          "HIP_VISIBLE_DEVICES", "AMD_VULKAN_ICD", "GGML_CUDA_ENABLE_UNIFIED_MEMORY"):
    if os.environ.get(k):
        d["env"][k] = os.environ[k]
try:
    d["toolkit"] = "" if os.name == "nt" else ""
except Exception:
    pass
cur = os.path.normcase(os.path.dirname(d["import_file"])) if d.get("import_file") else ""
d["stray"] = [p for p in d["leftover"]
              if os.path.basename(p).lower() in ("llama_cpp", "llama-cpp")
              and os.path.normcase(p) != cur]
d["dup"] = len(d["dists"]) > 1
print(json.dumps(d, ensure_ascii=False))
"""
    try:
        p = subprocess.run([py, "-c", code], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=180, creationflags=_no_window())
    except Exception as e:
        return {"error": f"探测失败：{e}"}
    out = (p.stdout or "").strip().splitlines()
    try:
        d = json.loads(out[-1]) if out else {"error": (p.stderr or "无输出")[:600]}
    except Exception:
        return {"error": ((p.stdout or "") + (p.stderr or ""))[:600]}
    # 顺手把结论也说清楚，面板直接显示人话
    libs = [x.lower() for x in (d.get("libs") or [])]
    has_vk = any("vulkan" in x for x in libs)
    has_hip = any("ggml-hip" in x or "rocblas" in x for x in libs)
    has_cuda = any("ggml-cuda" in x or "cublas" in x for x in libs)
    d["what"] = ("vulkan 版（走显卡）" if has_vk else
                 ("HIP/ROCm 版（走显卡）" if has_hip else
                  ("CUDA 版（走显卡）" if has_cuda else "CPU 版（只在内存里算）")))
    d["gpu_capable"] = bool(has_vk or has_hip or has_cuda)
    if not d.get("import_error"):
        devs = _parse_dev_names(p.stderr or "")
        if devs:
            d["devices"] = devs
    if log:
        try:
            log(f"[诊断] {d.get('what')}  version={d.get('version') or '?'}  "
                f"file={d.get('import_file') or '?'}")
        except Exception:
            pass
    return d


def uninstall_llama(py: str, log=print) -> dict:
    """把这个 python 里的 llama-cpp-python 卸干净：
    1) pip uninstall -y（cuda/cpu/vulkan 哪版都卸）
    2) 再扫 site-packages，把 pip 删不掉的残留目录（源码编译装过留下的 llama_cpp\、
       *.dist-info、*.egg-info）统统删掉——两份 llama_cpp 并存时，导入的可能是那份 CPU 版。"""
    lines: list[str] = []

    def say(s):
        lines.append(str(s))
        try:
            log(str(s))
        except Exception:
            pass

    say(f"目标 python：{py}")
    for name in ("llama-cpp-python", "llama_cpp_python"):
        rc = _pip(py, ["-m", "pip", "uninstall", "-y", "--no-input", name], say, timeout=300)
        say(f"  pip uninstall {name} → 退出码 {rc}")
    code = r"""
import glob, json, os, shutil, site
removed, failed, kept = [], [], []
site_dirs = []
try:
    site_dirs = [p for p in site.getsitepackages() if os.path.isdir(p)]
except Exception:
    pass
for sp in site_dirs:
    for p in glob.glob(os.path.join(sp, "llama_cpp*")) + glob.glob(os.path.join(sp, "llama-cpp*")):
        b = os.path.basename(p)
        try:
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
            removed.append(p)
        except Exception as e:
            failed.append("%s (%s)" % (p, e))
try:
    import llama_cpp
    kept.append(getattr(llama_cpp, "__file__", "?"))
except Exception:
    pass
print(json.dumps({"removed": removed, "failed": failed, "still_importable": kept}, ensure_ascii=False))
"""
    try:
        p = subprocess.run([py, "-c", code], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=180, creationflags=_no_window())
    except Exception as e:
        say(f"清残留失败：{e}")
        return {"ok": False, "error": str(e), "log": lines[-40:]}
    out = (p.stdout or "").strip().splitlines()
    info = {}
    try:
        info = json.loads(out[-1]) if out else {}
    except Exception:
        info = {"error": ((p.stdout or "") + (p.stderr or ""))[:300]}
    for r in info.get("removed") or []:
        say(f"  已删除残留：{r}")
    for r in info.get("failed") or []:
        say(f"  删不掉（可能还被占用）：{r}")
    still = info.get("still_importable") or []
    ok = not still
    if still:
        say(f"  卸载后仍然 import 得到：{still[0]} —— 说明还有一份没清掉，把服务停了再点一次")
    else:
        say("  卸载完成：这个 python 里已经没有 llama_cpp 了")
    return {"ok": ok, "removed": info.get("removed") or [], "failed": info.get("failed") or [],
            "still_importable": still, "log": lines[-40:]}


def _vk_sdk() -> str:
    """Vulkan SDK 在不在（编译 Vulkan 版 llama.cpp 必须有它）。"""
    for k in ("VULKAN_SDK", "VK_SDK_PATH"):
        v = os.environ.get(k)
        if v and os.path.isdir(v):
            return v
    cands = [os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "VulkanSDK")]
    for c in cands:
        if os.path.isdir(c):
            subs = sorted(os.listdir(c))
            for s in subs:
                p = os.path.join(c, s)
                if os.path.isdir(p):
                    return p
    return ""


def install_llama_vulkan(py: str, log=print) -> dict:
    """把 llama-cpp-python 换成 Vulkan 版（AMD 显卡在 Windows 上的主路）。
    先拿官方预编译 wheel（py3-none-win_amd64，自带 ggml-vulkan.dll）——不用装 Vulkan SDK、
    不用 MSVC/CMake；只有下载失败且本机确实有 SDK 时才退回去现场编译。"""
    lines: list[str] = []

    def say(s):
        lines.append(str(s))
        try:
            log(str(s))
        except Exception:
            pass

    say("=" * 60)
    say("第一步：官方预编译 Vulkan wheel（免 SDK、免编译）")
    r = install_llama_preset(py, "vulkan", say)
    if r.get("ok") and str(r.get("backend")) == "vulkan":
        return {"ok": True, "version": r.get("version"), "backend": "vulkan",
                "devices": r.get("devices"), "wheel": r.get("wheel"), "error": "", "log": lines[-80:]}
    say(f"预编译 wheel 这条路没成（{r.get('error') or '未知原因'}）")

    vk = _vk_sdk()
    if not vk:
        say("本机也没有 Vulkan SDK，现场编译走不了。")
        say("要 GPU 跑就再点一次「装 Vulkan 版」——下载中断的话会接着续传；"
            "或者手动下 wheel（点日志里那条链接）后用「装本地 wheel…」。")
        return {"ok": False, "backend": r.get("backend", ""), "error": r.get("error") or "没装上",
                "wheel": r.get("wheel", ""), "log": lines[-60:]}
    say(f"第二步：找到 Vulkan SDK（{vk}），改为现场编译")
    say("CMAKE_ARGS=-DGGML_VULKAN=ON，现场编译 5~20 分钟")
    old = llama_version(py)
    if old:
        say(f"当前版本：{old}（会被覆盖）")
    rc = _pip(py, ["-m", "pip", "install", "-U", "--no-input",
                   "numpy>=1.20.0", "typing-extensions>=4.5.0", "diskcache>=5.6.1", "jinja2>=2.11.3"],
              say, timeout=900)
    if rc != 0:
        say("运行依赖装失败（继续试主包）")
    env_extra = {"CMAKE_ARGS": "-DGGML_VULKAN=ON", "FORCE_CMAKE": "1"}
    rc = _pip_env(py, ["-m", "pip", "install", "-U", "--force-reinstall", "--no-cache-dir",
                       "--no-input", "llama-cpp-python"], say, env_extra, timeout=3600)
    ok_imp, v, err = llama_import_check(py)
    be = llama_backend(py)
    if rc != 0 or not ok_imp:
        say(f"编译/安装失败：rc={rc} / import={ok_imp} / {err}")
        say("常见原因：① 没装 Visual Studio 的「使用 C++ 的桌面开发」；② 没装 CMake；③ Vulkan SDK 没配 glslc。")
        return {"ok": False, "backend": be.get("backend", ""), "error": err or f"pip 退出码 {rc}",
                "log": lines[-60:]}
    if be.get("backend") != "vulkan":
        say(f"装上了，但探测到的后端是 {be.get('backend')}（libs={be.get('libs')}）——没编进 Vulkan，"
            f"这样跑还是会用 CPU。")
        return {"ok": False, "backend": be.get("backend", ""), "error": "编译产物里没有 ggml-vulkan",
                "version": v, "log": lines[-60:]}
    say(f"安装成功：llama-cpp-python {v}，后端 = Vulkan，GPU 可以用起来了")
    return {"ok": True, "version": v, "backend": "vulkan", "devices": be.get("devices"),
            "error": "", "log": lines[-60:]}


def install_llama_hip(py: str, log=print, targets: str = "gfx1010") -> dict:
    """把 llama-cpp-python 换成 HIP 版。同样先试官方预编译 hip-radeon wheel（免 SDK），
    只有下载失败且本机有 ROCm/HIP SDK 时才现场编译。"""
    lines: list[str] = []

    def say(s):
        lines.append(str(s))
        try:
            log(str(s))
        except Exception:
            pass

    say("=" * 60)
    say("第一步：官方预编译 HIP-Radeon wheel（免 SDK、免编译）")
    r = install_llama_preset(py, "hip-radeon", say)
    if r.get("ok") and str(r.get("backend")) in ("hip", "cuda", "vulkan"):
        return {"ok": True, "version": r.get("version"), "backend": r.get("backend"),
                "devices": r.get("devices"), "wheel": r.get("wheel"), "error": "", "log": lines[-80:]}
    say(f"预编译 wheel 这条路没成（{r.get('error') or '未知原因'}）")

    sdk = _hip_sdk()
    if not sdk:
        say("本机也没有 AMD ROCm / HIP SDK，现场编译走不了。")
        say("AMD 卡建议改用「装 Vulkan 版」——那条路只要显卡驱动，跟 ROCm 无关。")
        return {"ok": False, "backend": r.get("backend", ""), "error": r.get("error") or "没装上",
                "wheel": r.get("wheel", ""), "log": lines[-60:]}
    say(f"第二步：找到 HIP SDK（{sdk}），改为现场编译")
    say(f"CMAKE_ARGS=-DGGML_HIP=ON -DAMDGPU_TARGETS={targets}（5~20 分钟）")
    rc = _pip(py, ["-m", "pip", "install", "-U", "--no-input",
                   "numpy>=1.20.0", "typing-extensions>=4.5.0", "diskcache>=5.6.1", "jinja2>=2.11.3"],
              say, timeout=900)
    if rc != 0:
        say("运行依赖装失败（继续试主包）")
    env_extra = {"CMAKE_ARGS": f"-DGGML_HIP=ON -DAMDGPU_TARGETS={targets}", "FORCE_CMAKE": "1"}
    rc = _pip_env(py, ["-m", "pip", "install", "-U", "--force-reinstall", "--no-cache-dir",
                       "--no-input", "llama-cpp-python"], say, env_extra, timeout=3600)
    ok_imp, v, err = llama_import_check(py)
    be = llama_backend(py)
    if rc != 0 or not ok_imp:
        say(f"编译/安装失败：rc={rc} / import={ok_imp} / {err}")
        say("常见原因：① ROCm 的 hipcc 不在 PATH；② 没装 Visual Studio C++ 工具链；③ 没装 CMake。")
        return {"ok": False, "backend": be.get("backend", ""), "error": err or f"pip 退出码 {rc}",
                "log": lines[-60:]}
    if be.get("backend") not in ("hip", "cuda"):
        say(f"装上了但后端是 {be.get('backend')}（libs={be.get('libs')}）——HIP 没编进去，还是 CPU。")
        return {"ok": False, "backend": be.get("backend", ""), "error": "没有 ggml-hip",
                "version": v, "log": lines[-60:]}
    say(f"安装成功：llama-cpp-python {v}，后端 = {be.get('backend')}，GPU 可以用起来了")
    return {"ok": True, "version": v, "backend": be.get("backend"), "devices": be.get("devices"),
            "error": "", "log": lines[-60:]}


def _hip_sdk() -> str:
    """ROCm / HIP SDK 在不在。"""
    for k in ("HIP_PATH", "ROCM_PATH", "HIP_PATH64"):
        v = os.environ.get(k)
        if v and os.path.isdir(v):
            return v
    roots = [os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "AMD", "ROCm")]
    for r in roots:
        if os.path.isdir(r):
            subs = sorted(os.listdir(r), reverse=True)
            for s in subs:
                p = os.path.join(r, s)
                if os.path.isdir(p) and (os.path.isfile(os.path.join(p, "bin", "hipcc.exe"))
                                        or os.path.isdir(os.path.join(p, "bin"))):
                    return p
    return ""


def install_llama_wheel_file(py: str, whl_path: str, log=print) -> dict:
    """用户自己下好 .whl 时，直接装本地文件。"""
    return install_llama(py, log=log, wheel_url=os.path.abspath(whl_path))


# ---------------------------------------------- 官方预编译 GPU wheel（不用装任何 SDK / 不用编译器）
# abetlen 的 index 里除了 cuXXX，还有 vulkan / hip-radeon 两条通道，都是 py3-none-win_amd64
# （CPython 3.8+ 通用，跟 cp313 完全兼容），里面自带 ggml-vulkan.dll / ggml-hip.dll。
# AMD 的 RX 5700 XT（gfx1010）走这一条，显卡直接用上，不用装 Vulkan SDK、ROCm、MSVC、CMake。
ABETLEN_WHL_IDX = "https://abetlen.github.io/llama-cpp-python/whl/"
WHEEL_MIRRORS = ("", "https://gh-proxy.com/", "https://ghfast.top/", "https://ghproxy.net/")


def _wheel_cache_dir() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Local")
    d = os.path.join(base, "RemoteCLIPVAE", "wheels")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        d = os.path.join(os.environ.get("TEMP") or ".", "RemoteCLIPVAE_wheels")
        os.makedirs(d, exist_ok=True)
    return d


def _wheel_ver_key(name: str) -> tuple:
    import re
    m = re.search(r"llama_cpp_python-([0-9][^/]*?)-(?:py3|cp\d|abi3)", name)
    if not m:
        return (0,)
    nums = re.findall(r"\d+", m.group(1).split("+")[0])
    return tuple(int(x) for x in nums) if nums else (0,)


def index_wheels(tag: str, timeout: float = 30.0) -> list:
    """抓官方预编译 index 页面，返回 Windows wheel 列表（版本从高到低）。"""
    import re
    import urllib.request
    url = ABETLEN_WHL_IDX + tag.strip("/") + "/llama-cpp-python/"
    req = urllib.request.Request(url, headers={"User-Agent": "RemoteCLIPVAE"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        html = r.read().decode("utf-8", "replace")
    items = []
    for h in re.findall(r'href="([^"]+\.whl[^"]*)"', html):
        clean = h.split("#")[0].split("?")[0]
        name = clean.rstrip("/").split("/")[-1]
        if not name.lower().endswith(".whl") or "win_amd64" not in name.lower():
            continue
        items.append({"name": name, "url": clean})
    items.sort(key=lambda it: _wheel_ver_key(it["name"]), reverse=True)
    return items


def pick_preset_wheel(tag: str, py_tag: str = "") -> dict:
    """从 index 里挑一个能装进这个 python 的 wheel：py3-none / abi3 通用，或 cp 版本对得上。"""
    want = (py_tag or "").lower()
    items = index_wheels(tag)
    if not items:
        raise RuntimeError(f"{tag} 这条通道里没有 Windows wheel")
    for it in items:
        n = it["name"].lower()
        if "-py3-none-" in n or "-abi3-" in n:
            return it
        if want and f"-{want}-" in n:
            return it
    raise RuntimeError(f"{tag} 通道里没有适配 {py_tag or '这个 python'} 的 wheel（最新是 {items[0]['name']}）")


def _head_size(url: str) -> int:
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "RemoteCLIPVAE", "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=45) as r:
        cr = r.headers.get("Content-Range") or ""
        if "/" in cr:
            try:
                return int(cr.rsplit("/", 1)[1])
            except Exception:
                pass
        return int(r.headers.get("Content-Length") or 0)


def _fetch_part(url: str, part: str, max_bytes: int, say) -> dict:
    """拉一小段（默认 8MB）就收手回上层重连——国内线路长连接必断，短连接最稳。"""
    import urllib.request
    have = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"User-Agent": "RemoteCLIPVAE"}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        partial = getattr(r, "status", 200) == 206
        if not partial and have:
            have = 0
            open(part, "wb").close()
        mode = "ab" if have else "wb"
        mark = have // (4 * 1048576)
        got = 0
        f = open(part, mode)
        try:
            while got < max_bytes:
                chunk = r.read(128 * 1024)
                if not chunk:
                    return {"have": have, "eof": True}
                f.write(chunk)
                got += len(chunk)
                have += len(chunk)
                if have // (4 * 1048576) != mark:
                    mark = have // (4 * 1048576)
                    say(f"  {have // 1048576} MB")
        finally:
            f.close()
    return {"have": have, "eof": False}


def _whl_ok(path: str) -> tuple:
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
            names = z.namelist()
        if bad:
            return (False, f"压缩包损坏（{bad}）")
        if not any(n.startswith("llama_cpp/") for n in names):
            return (False, "包内容不像 llama_cpp_python")
        return (True, "")
    except Exception as e:
        return (False, str(e)[:120])


def fetch_wheel(url: str, name: str, dirpath: str, log=print) -> str:
    """多线路 + 断点续传下载 wheel，下完校验，返回本地路径。"""
    lines = []

    def say(s):
        lines.append(str(s))
        try:
            log(str(s))
        except Exception:
            pass

    os.makedirs(dirpath, exist_ok=True)
    dst = os.path.join(dirpath, name)
    if os.path.isfile(dst):
        ok, why = _whl_ok(dst)
        if ok:
            say(f"本地已经有 {name}，直接拿来装")
            return dst
        say(f"本地那个 {name} 坏了（{why}），删掉重下")
        try:
            os.remove(dst)
        except Exception:
            pass
    part = dst + ".part"
    total = 0
    for u in [m + url for m in WHEEL_MIRRORS]:
        try:
            total = _head_size(u)
            if total:
                break
        except Exception:
            continue
    if not os.path.exists(part):
        open(part, "wb").close()
    have0 = os.path.getsize(part)
    say(f"下载 {name}" + (f"（{total // 1048576} MB）" if total else "") +
        (f"，接着上次的 {have0 // 1048576} MB 继续" if have0 else "") + "…")
    urls = [m + url for m in WHEEL_MIRRORS]
    done = False
    tries = 0
    while tries < len(urls) * 6:
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if total and have >= total:
            done = True
            break
        u = urls[tries % len(urls)]
        tries += 1
        try:
            info = _fetch_part(u, part, 8 * 1048576, say)
        except Exception as e:
            host = u.split("//")[-1].split("/")[0]
            say(f"  {host} 断了：{str(e)[:80]}")
            time.sleep(1)
            continue
        if info.get("eof"):
            size_now = os.path.getsize(part) if os.path.exists(part) else 0
            if not total or size_now >= total:
                done = True
                break
    size_now = os.path.getsize(part) if os.path.exists(part) else 0
    if not done and (not total or size_now < total):
        say(f"下载没下完（{size_now // 1048576} MB" + (f" / {total // 1048576} MB" if total else "") +
            "）；已经下的部分留在 .part 里，再点一次会接着下。")
        raise RuntimeError("下载没完成（网络慢/被墙），再点一次会续传")
    try:
        os.replace(part, dst)
    except Exception as e:
        raise RuntimeError(f"落盘失败：{e}")
    ok, why = _whl_ok(dst)
    if not ok:
        try:
            os.remove(dst)
        except Exception:
            pass
        raise RuntimeError(f"下下来的 wheel 校验不过：{why}")
    say(f"下好了：{dst}")
    return dst


def install_llama_preset(py: str, kind: str = "vulkan", log=print) -> dict:
    """装官方预编译的 Vulkan / HIP-Radeon wheel —— 不装 SDK、不编译，AMD 显卡直接能用。"""
    lines = []

    def say(s):
        lines.append(str(s))
        try:
            log(str(s))
        except Exception:
            pass

    tag = "hip-radeon" if str(kind).lower().startswith("hip") else "vulkan"
    info = py_env_info(py)
    say(f"走官方预编译 {tag} wheel（py3-none-win_amd64，自带 GPU 后端，不用 SDK / 不用编译器）")
    try:
        w = pick_preset_wheel(tag, str(info.get("tag") or ""))
        say(f"选中：{w['name']}")
    except Exception as e:
        say(f"取 wheel 列表失败：{e}")
        return {"ok": False, "backend": "", "wheel": "", "error": str(e), "log": lines[-60:]}
    try:
        local = fetch_wheel(w["url"], w["name"], _wheel_cache_dir(), say)
    except Exception as e:
        say(f"下载失败：{e}")
        say("兜底：手动打开 " + w["url"] + " 下这个 wheel，然后点「装本地 wheel…」选它")
        return {"ok": False, "backend": "", "wheel": w["name"], "error": str(e), "log": lines[-60:]}
    res = install_llama(py, log=say, wheel_url=local)
    be = llama_backend(py)
    if not res.get("ok"):
        say(f"装完 import 不过：{res.get('error') or ''}")
        return {"ok": False, "backend": be.get("backend", ""), "wheel": w["name"],
                "error": res.get("error") or "import 失败", "log": lines[-80:]}
    if tag == "vulkan" and be.get("backend") != "vulkan":
        say(f"装上了，但后端探测是 {be.get('backend')}（libs={be.get('libs')}）——没进 Vulkan，还是 CPU。")
        return {"ok": False, "backend": be.get("backend", ""), "wheel": w["name"],
                "error": "wheel 里没有 ggml-vulkan", "log": lines[-80:]}
    say(f"安装成功：llama-cpp-python {res.get('version')}，后端 = {be.get('backend')}"
        + (f"，设备 {be.get('devices')}" if be.get("devices") else ""))
    return {"ok": True, "version": res.get("version"), "backend": be.get("backend", ""),
            "wheel": w["name"], "devices": be.get("devices"), "error": "", "log": lines[-80:]}


def _is_amd(info: dict) -> bool:
    devs = " ".join([str(d) for d in (info.get("devices") or [])]).lower()
    return ("radeon" in devs) or ("amd" in devs) or ("gfx" in devs)


# --------------------------------------------------------------------- llama-server（Vulkan 版二进制）
LLAMA_SERVER_PORT = 8300
_LLAMA_SRV = {"proc": None, "port": LLAMA_SERVER_PORT, "exe": "", "cmd": ""}


def _gh_json(url: str, timeout: float = 30.0):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "RemoteCLIPVAE",
                                               "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def download_llama_server_vulkan(dirpath: str, tag: str = "", log=print) -> dict:
    """从 llama.cpp 官方 release 里下 Vulkan 版 Windows 包，解压到 dirpath。
    这样 AMD 卡不用编 llama-cpp-python，直接用 llama-server.exe 就能吃显卡。"""
    import urllib.request
    import zipfile

    lines: list[str] = []

    def say(s):
        lines.append(str(s))
        try:
            log(str(s))
        except Exception:
            pass

    os.makedirs(dirpath, exist_ok=True)
    asset = None
    cands: list = []
    tag_name = ""
    rels: list = []

    def _api(url: str, tries: int = 3):
        last = None
        for _i in range(tries):
            try:
                return _gh_json(url)
            except Exception as e:
                last = e
                import time as _t
                _t.sleep(2)
        raise last if last else RuntimeError("取 release 失败")

    try:
        if tag:
            rels = [_api(f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}")]
        else:
            rels = _api("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15")
    except Exception as e:
        say(f"取 release 列表失败（{str(e)[:100]}）——改用内置的最新版本号兜底")
        rels = []
    # 有的版本（如 v1.0.0）只是占位包，没有 exe；往后翻，找真带 Windows Vulkan 二进制的
    for rel in rels:
        for a in (rel.get("assets") or []):
            n = (a.get("name") or "").lower()
            if n.endswith(".zip") and "win" in n and "x64" in n and "vulkan" in n:
                cands.append((rel.get("tag_name") or "", a))
        if cands:
            break
    if not cands:
        # API 被墙/超时：直接按已知版本号拼地址试（这几个版本都有 win-vulkan-x64 包）
        for t in ("b12606", "b12600", "b12596"):
            cands.append((t, {"name": f"llama-{t}-bin-win-vulkan-x64.zip",
                              "browser_download_url": f"https://github.com/ggml-org/llama.cpp/releases/"
                                                      f"download/{t}/llama-{t}-bin-win-vulkan-x64.zip"}))
        say("走内置版本号兜底（down 不到就手动下）")
    tag_name, asset = cands[0]
    name = asset.get("name") or "llama-vulkan.zip"
    url = asset.get("browser_download_url") or ""
    gh_url = f"https://github.com/ggml-org/llama.cpp/releases/download/{tag_name}/{name}"
    # GitHub 直连在国内经常被掐（SSL EOF / 连接重置），依次试镜像；中途断了能续传
    urls = [url or gh_url]
    for m in ("https://gh-proxy.com/", "https://ghproxy.net/", "https://ghfast.top/"):
        urls.append(m + gh_url)
    say(f"下载 {name}（{tag_name}，约 30~40MB）…")
    zpath = os.path.join(dirpath, name)
    part = zpath + ".part"
    if os.path.exists(part) and os.path.getsize(part) > 1024:
        say(f"接着上次没下完的部分继续（{os.path.getsize(part) // 1048576} MB）")
    else:
        try:
            os.remove(part)
        except Exception:
            pass

    def _fetch(u: str) -> int:
        import urllib.request
        have = os.path.getsize(part) if os.path.exists(part) else 0
        headers = {"User-Agent": "RemoteCLIPVAE"}
        if have:
            headers["Range"] = f"bytes={have}-"
        req = urllib.request.Request(u, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as r:
            partial = (getattr(r, "status", 200) == 206)
            total = int(r.headers.get("Content-Length") or 0) + (have if partial else 0)
            if not partial:
                have = 0
            with open(part, "ab" if partial else "wb") as f:
                mark = have // (8 * 1048576)
                while True:
                    chunk = r.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
                    have += len(chunk)
                    if have // (8 * 1048576) != mark:
                        mark = have // (8 * 1048576)
                        say(f"  {have // 1048576}" + (f" / {total // 1048576}" if total else "") + " MB")
        return have

    got = 0
    for i, u in enumerate(urls):
        try:
            host = u.split("/")[2]
        except Exception:
            host = u
        for attempt in (1, 2):
            try:
                say(f"线路 {i + 1}/{len(urls)}（{host}）第 {attempt} 次…")
                got = _fetch(u)
            except Exception as e:
                say(f"  断了：{str(e)[:110]}")
                got = os.path.getsize(part) if os.path.exists(part) else 0
                import time as _t
                _t.sleep(1)
                continue
            if got >= 20 * 1048576:
                break
            say(f"  这条线路只拿到 {got // 1048576} MB，换线路")
        if got >= 20 * 1048576:
            break
    if got < 20 * 1048576:
        say("下载失败：GitHub 直连和镜像都不通。")
        say(f"手动兜底：浏览器打开 https://github.com/ggml-org/llama.cpp/releases ，"
            f"下 {name}，解压到 {dirpath}（要和 llama-server.exe 同级放好 dll）")
        return {"ok": False, "error": "下载失败（网络不通）", "log": lines[-40:]}
    try:
        os.replace(part, zpath)
        say("解压中…")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(dirpath)
        os.remove(zpath)
    except Exception as e:
        say(f"解压失败：{e}")
        return {"ok": False, "error": str(e), "log": lines[-40:]}
    exe = find_llama_server(dirpath)
    if not exe:
        say("解压完了但没找到 llama-server.exe")
        return {"ok": False, "error": "解压后没有 llama-server.exe", "log": lines[-30:]}
    say(f"好了：{exe}")
    return {"ok": True, "exe": exe, "tag": tag_name, "asset": name, "error": "", "log": lines[-40:]}


def find_llama_server(root: str) -> str:
    for base, _dirs, files in os.walk(root):
        for f in files:
            if f.lower() == "llama-server.exe":
                return os.path.join(base, f)
    return ""


def import_llama_server_zip(zip_path: str, dirpath: str, log=print) -> dict:
    """用户自己下好 llama-*-bin-win-vulkan-x64.zip 时，直接解压到 dirpath。
    （GitHub 在墙内经常下不动，这是最稳的兜底。）"""
    import zipfile
    lines: list[str] = []

    def say(s):
        lines.append(str(s))
        try:
            log(str(s))
        except Exception:
            pass

    if not os.path.isfile(zip_path):
        return {"ok": False, "error": f"文件不存在：{zip_path}", "log": lines}
    os.makedirs(dirpath, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dirpath)
    except Exception as e:
        say(f"解压失败：{e}")
        return {"ok": False, "error": str(e), "log": lines}
    exe = find_llama_server(dirpath)
    if not exe:
        say("解压完了没找到 llama-server.exe —— 解压的可能不是 Vulkan 包（要 llama-*-bin-win-vulkan-x64.zip）")
        return {"ok": False, "error": "压缩包里没有 llama-server.exe", "log": lines}
    say(f"好了：{exe}")
    return {"ok": True, "exe": exe, "tag": "(本地 zip)", "asset": os.path.basename(zip_path),
            "error": "", "log": lines}


def start_llama_server(exe: str, model: str, mmproj: str = "", port: int = LLAMA_SERVER_PORT,
                       ngl: int = 99, n_ctx: int = 4096, log=print) -> dict:
    """拉起 llama-server（Vulkan 版）并等它就绪。model 必须是 .gguf 绝对路径。"""
    lines: list[str] = []

    def say(s):
        lines.append(str(s))
        try:
            log(str(s))
        except Exception:
            pass

    stop_llama_server()
    if not os.path.exists(exe):
        return {"ok": False, "error": f"llama-server.exe 不存在：{exe}", "log": lines}
    if not os.path.exists(model):
        return {"ok": False, "error": f"GGUF 不存在：{model}", "log": lines}
    cmd = [exe, "-m", model, "-ngl", str(int(ngl)), "-c", str(int(n_ctx)),
           "--host", "127.0.0.1", "--port", str(int(port))]
    if mmproj and os.path.exists(mmproj):
        cmd += ["--mmproj", mmproj]
    say("启动：" + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    try:
        import subprocess
        proc = subprocess.Popen(cmd, cwd=os.path.dirname(exe), stdout=subprocess.DEVNULL,
                                stderr=subprocess.STDOUT, creationflags=_no_window())
    except Exception as e:
        say(f"启动失败：{e}")
        return {"ok": False, "error": str(e), "log": lines}
    _LLAMA_SRV.update({"proc": proc, "port": int(port), "exe": exe,
                       "cmd": " ".join(f'"{c}"' if " " in c else c for c in cmd)})
    import time
    t0 = time.time()
    while time.time() - t0 < 180:
        if proc.poll() is not None:
            say("llama-server 退出了（多半是显存不够或参数不对，看它自己的日志窗口）")
            return {"ok": False, "error": f"退出码 {proc.returncode}", "log": lines}
        if llama_server_health(port).get("ok"):
            say(f"就绪：http://127.0.0.1:{port}（{round(time.time() - t0, 1)}s）")
            return {"ok": True, "port": int(port), "pid": proc.pid, "cmd": _LLAMA_SRV["cmd"],
                    "base_url": f"http://127.0.0.1:{port}", "error": "", "log": lines}
        time.sleep(1.0)
    say("等了 180 秒还没就绪（首次加载大模型可能很慢，日志里看进度）")
    return {"ok": False, "error": "启动超时", "port": int(port), "log": lines}


def llama_server_health(port: int = LLAMA_SERVER_PORT, timeout: float = 3.0) -> dict:
    import urllib.request
    last = ""
    for path in ("/health", "/v1/models"):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{int(port)}{path}", timeout=timeout) as r:
                body = r.read().decode("utf-8", "replace")[:300]
            return {"ok": True, "path": path, "body": body}
        except Exception as e:
            last = str(e)
    return {"ok": False, "error": last or "连不上"}


def stop_llama_server() -> dict:
    p = _LLAMA_SRV.get("proc")
    if p is None:
        return {"ok": True, "stopped": False}
    try:
        p.terminate()
        try:
            p.wait(timeout=8)
        except Exception:
            p.kill()
    except Exception:
        pass
    _LLAMA_SRV["proc"] = None
    return {"ok": True, "stopped": True}


def llama_server_state() -> dict:
    p = _LLAMA_SRV.get("proc")
    alive = bool(p is not None and p.poll() is None)
    return {"running": alive, "pid": (p.pid if alive else 0), "port": _LLAMA_SRV.get("port"),
            "exe": _LLAMA_SRV.get("exe") or "", "cmd": _LLAMA_SRV.get("cmd") or ""}



class ExternalLLM:
    """外部 llama.cpp 服务（llama-server）。用于「本机 python 装不出 GPU 后端」的显卡：
    比如 RX 5700 XT（gfx1010）在 Windows 上 ROCm/HIP 不支持、也没有 CUDA，
    但 llama.cpp 的 Vulkan 版二进制直接就能用显卡。

    配置：config.json 里加一行 `"llm_external": "http://127.0.0.1:8300"`（llama-server 的地址，
    带不带 /v1 都行）。填了以后服务端就不再拉 llm_worker 子进程，聊天全部转发给这个服务。
    """

    kind = "external"

    def __init__(self, backend, base_url: str):
        self.backend = backend
        self.base = self._norm(base_url)
        self.state = {"state": "外部服务", "base_url": self.base, "model": "", "mmproj": "",
                      "n_ctx": 0, "handler": "", "device_name": "", "last": {},
                      "progress": None, "error": "", "backend": "vulkan"}

    @staticmethod
    def _norm(u: str) -> str:
        u = str(u or "").strip().rstrip("/")
        if not u:
            return ""
        if not u.startswith("http://") and not u.startswith("https://"):
            u = "http://" + u
        if not u.endswith("/v1"):
            u = u + "/v1"
        return u

    # ---- HTTP ---- #
    def _req(self, path: str, payload: dict | None = None, timeout: float | None = None):
        import urllib.request
        import urllib.error

        url = self.base + path
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers,
                                     method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            detail = (e.read() or b"").decode("utf-8", "replace")[:500]
            raise RuntimeError(f"外部 LLM 服务返回 HTTP {e.code}：{detail}") from None
        except Exception as e:
            raise RuntimeError(
                f"连不上外部 LLM 服务 {self.base}：{e}\n"
                "先在服务端这台机上把 llama-server.exe 跑起来（例如 "
                "`llama-server.exe -m 模型.gguf --mmproj mmproj.gguf -ngl 99 --port 8300`），"
                "再把面板里「外部 LLM 服务地址」填成 http://127.0.0.1:8300"
            ) from None

    def list_models(self, ttl: float = 3.0) -> dict:
        try:
            d = self._req("/models", timeout=6.0)
            ids = [str(m.get("id") or "") for m in (d.get("data") or []) if m.get("id")]
        except Exception as e:
            self.state["error"] = str(e)
            return {"llm": {}, "llm_vision": {},
                    "error": str(e),
                    "dirs": [self.base]}
        out = {"llm": {i: {"files": [i], "exists": True, "size_gb": 0.0} for i in ids},
               "llm_vision": {}, "dirs": [self.base]}
        if ids:
            self.state["model"] = ids[0]
        return out

    def status(self, light: bool = False) -> dict:
        st = dict(self.state)
        try:
            st["models"] = {"llm": {}, "llm_vision": {}} if light else self.list_models(ttl=0.0)
            st["running"] = "error" not in st["models"]
        except Exception:
            st["running"] = False
        st["external"] = True
        return st

    def load(self, model: str, mmproj: str = "", n_ctx: int = 4096, n_gpu_layers: int = -1) -> dict:
        # 外部服务的模型由 llama-server 自己的 -m / --mmproj 决定，这里只记个状态
        self.state.update({"model": str(model or ""), "mmproj": str(mmproj or ""),
                           "n_ctx": int(n_ctx or 0), "state": "外部服务"})
        return self.status()

    def unload(self) -> dict:
        self.state.update({"state": "外部服务", "progress": None})
        return self.status()

    def chat(self, model: str = "", text: str = "", system: str = "", images=None,
             max_tokens: int = 2048, temperature: float = 0.7, seed=None,
             mmproj: str = "", n_ctx: int = 4096, timeout: float | None = None, auto_load: bool = True) -> dict:
        parts = [{"type": "text", "text": text}]
        for b64 in (images or []):
            parts.append({"type": "image_url",
                          "image_url": {"url": "data:image/png;base64," + b64}})
        payload = {
            "model": model or self.state.get("model") or "",
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": parts if images else text}],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": False,
            "cache_prompt": True,
        }
        if seed is not None:
            payload["seed"] = int(seed)
        t0 = time.time()
        d = self._req("/chat/completions", payload, timeout=timeout)
        content = (((d.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        usage = d.get("usage") or {}
        tokens = int(usage.get("completion_tokens") or 0)
        secs = max(0.001, time.time() - t0)
        tps = round(tokens / secs, 2) if tokens else 0
        self.state.update({"state": "已加载", "last": {"tokens": tokens, "tps": tps},
                           "progress": None})
        return {"text": content, "tokens": tokens, "tps": tps,
                "model": payload["model"], "engine_seconds": round(secs, 2)}


