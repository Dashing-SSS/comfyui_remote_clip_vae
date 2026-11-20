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
                      "device_name": "", "progress": None, "last": {}, "error": "", "pid": None}
        self._loaded_key = ("", "")

    # ------------------------------------------------------------------ #
    # 路径解析
    # ------------------------------------------------------------------ #
    def comfy_dir(self) -> str:
        return str(getattr(self.backend, "comfy_dir", "") or "")

    def llm_dirs(self) -> list[str]:
        cfg = getattr(self.backend, "cfg", {}) or {}
        d = str(cfg.get("llm_model_dir") or "").strip()
        dirs = [d] if d else []
        if not dirs:
            cd = self.comfy_dir()
            for sub in GGUF_DIRS:
                dirs.append(os.path.join(cd, "models", sub))
        return [os.path.abspath(os.path.expanduser(x)) for x in dirs if x]

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

    def _call(self, cmd: str, params: dict | None = None, timeout: float = 600.0) -> dict:
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
            got = slot["ev"].wait(timeout)
            self._pending.pop(rid, None)
            if not got:
                self.state["progress"] = None
                raise RuntimeError(f"LLM 子进程 {cmd} 超时（{timeout:.0f}s）。面板「服务端日志」里能看到它卡在哪。")
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
                           "progress": None})
        return self.status()

    def chat(self, model: str, text: str, system: str = "", images: list | None = None,
             max_tokens: int = 1024, temperature: float = 0.7, seed=None,
             mmproj: str = "", n_ctx: int = 4096, timeout: float = 600.0,
             auto_load: bool = True) -> dict:
        if auto_load and model:
            path = self.resolve(model, "llm")
            mpath = self.resolve(mmproj, "llm_vision") if mmproj else ""
            if (path, mpath) != self._loaded_key:
                self.load(model, mmproj, n_ctx=n_ctx)
        self.state.update({"state": "生成中", "progress": {"tokens": 0, "tps": 0, "elapsed": 0}})
        params = {"system": system, "text": text, "images": images or [],
                  "max_tokens": int(max_tokens), "temperature": float(temperature)}
        if seed is not None:
            params["seed"] = int(seed)
        t0 = time.time()
        data = self._call("chat", params, timeout=timeout)
        self.state.update({"state": "空闲", "progress": None, "last": data})
        data["engine_seconds"] = round(time.time() - t0, 2)
        return data

    def unload(self) -> dict:
        try:
            self._call("unload", {}, timeout=60)
        except Exception as e:
            self.state["error"] = str(e)
        self._loaded_key = ("", "")
        self.state.update({"state": "未加载", "model": "", "mmproj": "", "n_ctx": 0,
                           "handler": "", "last": {}, "progress": None})
        return self.status()

    def info(self) -> dict:
        if self.proc is None or self.proc.poll() is not None:
            return {"loaded": False, "llama_cpp": "未启动"}
        try:
            return self._call("info", {}, timeout=30)
        except Exception as e:
            return {"loaded": False, "error": str(e)}

    def status(self) -> dict:
        st = dict(self.state)
        st.pop("_torch_cache", None)
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

    def status(self) -> dict:
        st = dict(self.state)
        st.pop("_torch_cache", None)
        st.update({"models": self.list_models(), "dirs": ["<mock>"], "running": True, "stderr": []})
        return st


# --------------------------------------------------------------------- 安装 llama-cpp-python
# 思路：优先用 JamePeng 预编译 wheel（就是孤海「Qwen image 2.1 提示词优化」那套，Win + CUDA 直接可装，
# 不会走源码编译），失败再退回 abetlen 官方 index，最后退回 PyPI。
JAMEPENG_API = "https://api.github.com/repos/JamePeng/llama-cpp-python/releases?per_page=100"
JAMEPENG_DL = "https://github.com/JamePeng/llama-cpp-python/releases/download/"


def _no_window() -> int:
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def py_env_info(py: str) -> dict:
    """问目标 python：版本 tag（cp313）、torch 的 CUDA 版本。"""
    code = (
        "import sys,json\n"
        "d={'tag':'cp%d%d'%(sys.version_info[0],sys.version_info[1]),"
        "'ver':sys.version.split()[0],'cuda':'','torch':'','exe':sys.executable}\n"
        "try:\n"
        "    import torch\n"
        "    d['cuda']=str(torch.version.cuda or '')\n"
        "    d['torch']=torch.__version__\n"
        "except Exception as e:\n"
        "    d['torch_error']=str(e)\n"
        "print(json.dumps(d))\n"
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
    m = re.match(r"(\d+)\.(\d+)", cuda)
    if not tag:
        raise RuntimeError(f"探测不到目标 python 的版本（{info.get('error') or info}）")
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
    log("$ " + " ".join([os.path.basename(py)] + args))
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
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
        f"CUDA {info.get('cuda') or '（无）'}")

    wheels: list[tuple[str, str]] = []   # [(how, url)]
    if wheel_url:
        wheels.append(("本地/手动指定的 wheel", wheel_url))
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
        if target.startswith("__index__"):
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
            say(f"安装成功：llama-cpp-python {v}（{how}）")
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


def install_llama_wheel_file(py: str, whl_path: str, log=print) -> dict:
    """用户自己下好 .whl 时，直接装本地文件。"""
    return install_llama(py, log=log, wheel_url=os.path.abspath(whl_path))

