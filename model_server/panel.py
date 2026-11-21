# -*- coding: utf-8 -*-
"""
panel.py —— 服务端控制面板（本地网页版，只用标准库，任何 python 都能跑）。

为什么有它：ComfyUI 的整合包 python_embeded 通常**没有 tkinter**（连 Tcl 运行时都没有），
所以除了 gui.py（tkinter 桌面窗口）之外，再提供一个"长得一样"的本地控制面板：
panel.py 起一个 127.0.0.1 的小 HTTP 服务，用浏览器以「应用窗口」方式打开，效果跟桌面程序一致。

功能与 gui.py 完全对齐：
  · 可视化编辑 config.json（监听地址 / 端口 / 计算设备 / ComfyUI 目录 / python / 模型目录）
  · 启动、停止服务端子进程，stdout 实时抓进「服务端日志（实时）」
  · 显存 / GPU 利用率 / 已加载模型实时刷新
  · 预加载模型、清空显存、扫描目录加条目、删条目、打开目录

用法：
    0_启动图形界面.bat              ← 推荐（有 tkinter 就跑 gui.py，没有就跑这个）
    python panel.py                 ← 也可以直接跑，会自动开浏览器
    python panel.py --no-browser    ← 只起服务不开浏览器
"""

from __future__ import annotations

import glob
import json
import os
import platform
import queue
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

def _base_dir() -> tuple:
    """返回 (脚本目录, 包根目录)。打包成 exe 后仍以 exe 旁边的 model_server 为准。"""
    if getattr(sys, "frozen", False):
        root = os.path.dirname(os.path.abspath(sys.executable))
        inner = os.path.join(root, "model_server")
        if os.path.isdir(inner):
            return inner, root
        return root, root
    here = os.path.dirname(os.path.abspath(__file__))
    return here, os.path.dirname(here)


HERE, PKG_ROOT = _base_dir()
CFG_PATH = os.path.join(HERE, "config.json")
PY_RECORD = os.path.join(HERE, "_python.txt")
COMFY_RECORD = os.path.join(HERE, "_comfy.txt")
GUI_PY_RECORD = os.path.join(HERE, "_gui_python.txt")

PANEL_PORT = 8190
APP_TITLE = "远程 CLIP / VAE 模型服务端"
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
MODEL_EXTS = (".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".gguf", ".bin")
# clip 目录里混着的视觉编码器不能当文本编码器加载
VISION_HINTS = ("eva02", "eva-clip", "sigclip", "clip_vision", "clipvision", "vision")

LOCAL_LOG: deque = deque(maxlen=1500)
SRV_LOG: deque = deque(maxlen=4000)
LOG_Q: queue.Queue = queue.Queue()


def now() -> str:
    return time.strftime("%H:%M:%S")


def add_local(msg: str, level: str = "info"):
    LOCAL_LOG.append({"ts": now(), "level": level, "text": msg})


def load_json(path: str, default: dict) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return json.loads(json.dumps(default))


def save_json(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def write_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def lan_ips() -> list[str]:
    ips = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.3)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip and ip not in ips:
            ips.insert(0, ip)
    except Exception:
        pass
    ips = [ip for ip in ips if not ip.startswith("127.")]

    def rank(ip: str) -> int:
        # 192.168.x / 10.x 这种普通局域网优先；tailscale 的 100.64~100.127 往后排
        try:
            first = int(ip.split(".")[0])
            second = int(ip.split(".")[1])
        except Exception:
            return 9
        if first == 100 and 64 <= second <= 127:
            return 5
        if ip.startswith("192.168."):
            return 0
        if first == 10:
            return 1
        if ip.startswith("172."):
            return 2
        return 3

    return sorted(ips, key=rank)


def gpu_stats():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4, creationflags=CREATE_NO_WINDOW,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        used, total, util = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")[:3]]
        return {"used_mb": float(used), "total_mb": float(total), "util": float(util)}
    except Exception:
        return None


def ram_gb():
    if os.name != "nt":
        return 0.0, 0.0
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = MS()
        st.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        total = st.ullTotalPhys / 2**30
        return round(total - st.ullAvailPhys / 2**30, 1), round(total, 1)
    except Exception:
        return 0.0, 0.0


def pythons_around(comfy_dir: str) -> list[str]:
    """只按当前 ComfyUI 目录找它自带的 python（整合包 python_embeded / Desktop .venv\\Scripts）。"""
    if not comfy_dir:
        return []
    base = os.path.abspath(clean_path(comfy_dir))
    layouts = [("python_embeded", "python.exe"), ("python_embedded", "python.exe"),
               ("python", "python.exe"),
               (".venv", os.path.join("Scripts", "python.exe")),
               ("venv", os.path.join("Scripts", "python.exe")),
               ("env", os.path.join("Scripts", "python.exe")),
               (".venv", "python.exe"), ("venv", "python.exe")]
    roots = [base, os.path.abspath(os.path.join(base, os.pardir)),
             os.path.abspath(os.path.join(base, os.pardir, os.pardir))]
    out, seen = [], set()
    for root in roots:
        for sub, exe in layouts:
            p = os.path.abspath(os.path.join(root, sub, exe))
            if p.lower() in seen or not os.path.isfile(p):
                continue
            seen.add(p.lower())
            out.append(p)
    return out


def python_has_torch(py: str) -> bool:
    py = os.path.abspath(py)
    d = os.path.dirname(py)
    for up in (d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))):
        for sub in (os.path.join("Lib", "site-packages", "torch", "__init__.py"),
                    os.path.join("lib", "site-packages", "torch", "__init__.py"),
                    os.path.join("site-packages", "torch", "__init__.py")):
            if os.path.isfile(os.path.join(up, sub)):
                return True
    return False


def best_python(comfy_dir: str) -> tuple[str, str]:
    cands = pythons_around(comfy_dir)
    good = [p for p in cands if python_has_torch(p)]
    if good:
        return good[0], "ok"
    if cands:
        return cands[0], "no-torch"
    return "", "none"


def clean_path(v) -> str:
    """把拖进输入框的 "D:\\...\\python.exe" 变成干净的路径（Windows 拖拽会自动加引号）。"""
    s = str(v or "").strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    s = s.strip().strip("'").strip()
    if s.startswith("\\\\?\\"):
        s = s[4:]
    return s


def probe_python(py: str, timeout: int = 45) -> tuple[bool, str]:
    """服务端要 import torch / numpy / comfy，随便一个 python 是不行的，启动前先探一下。"""
    try:
        r = subprocess.run([py, "-c", "import numpy, torch"],
                           capture_output=True, text=True, timeout=timeout,
                           creationflags=CREATE_NO_WINDOW)
    except Exception as e:
        return False, f"探测失败：{e}"
    if r.returncode == 0:
        return True, "ok"
    lines = ((r.stderr or "") + (r.stdout or "")).strip().splitlines()
    return False, (lines[-1].strip() if lines else f"退出码 {r.returncode}")


def find_comfy_dirs() -> list[str]:
    found, seen = [], set()

    def add(p):
        p = os.path.abspath(p)
        if p in seen or not os.path.isdir(p):
            return
        if os.path.isfile(os.path.join(p, "comfy", "sd.py")) and os.path.isfile(os.path.join(p, "main.py")):
            seen.add(p)
            found.append(p)

    remembered = read_text(COMFY_RECORD)
    if remembered:
        add(remembered)
    roots = [f"{d}:\\" for d in "CDEFGH" if os.path.isdir(f"{d}:\\")] if os.name == "nt" else [os.path.expanduser("~")]
    for r in roots:
        for pat in ("ComfyUI", "ComfyUI*", os.path.join("*", "ComfyUI*"),
                    os.path.join("*", "*", "ComfyUI*"), os.path.join("*", "ComfyUI")):
            try:
                for p in glob.glob(os.path.join(r, pat)):
                    add(p)
            except Exception:
                continue
    return found


# --------------------------------------------------------------------------- #
# 服务端进程管理
# --------------------------------------------------------------------------- #
class Runner:
    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.health: dict | None = None
        self.gpu_cache = {}
        self.gpu_n = 0
        self.cfg = load_json(CFG_PATH, {})

    # ---- 进程 ---- #
    def running(self) -> bool:
        return bool(self.proc and self.proc.poll() is None)

    def reload_cfg(self):
        self.cfg = load_json(CFG_PATH, self.cfg or {})

    def start(self, mock: bool = False) -> tuple[bool, str]:
        self.reload_cfg()
        if self.running():
            return True, "服务已经在运行"
        comfy = clean_path(self.cfg.get("comfyui_dir")).replace("\\", "/")
        py = clean_path(read_text(PY_RECORD))
        if not py or not os.path.isfile(py):
            cands = pythons_around(comfy)
            py = cands[0] if cands else ""
        if not py or not os.path.isfile(py):
            add_local("还没指定 ComfyUI 的 python，启动被拦下", "err")
            return False, ("还没指定 ComfyUI 的 python：\n"
                           "① 在「ComfyUI 路径」里选（或把目录拖进来）然后点『自动找』；\n"
                           "② 或者直接在「本机 ComfyUI 的 Python」里填 python_embeded\\python.exe 的路径。\n"
                           "服务端必须用 ComfyUI 自带的那份 python，面板自己的 python 没有 torch/comfy。")
        ok, why = probe_python(py)
        if not ok:
            add_local(f"这个 python 不能用来跑服务端：{py}（{why}）", "err")
            return False, (f"这个 python 跑不了服务端：{why}\n"
                           "它要有 torch / numpy / comfy —— 也就是 ComfyUI 自带的 python_embeded\\python.exe。\n"
                           "点『自动找』或手动填一个。")
        args = [py, os.path.join(HERE, "app.py"), "--config", CFG_PATH]
        dev = str(self.cfg.get("device") or "auto").lower()
        if dev in ("cuda", "cpu"):
            args += ["--device", dev]
        if mock:
            args.append("--mock")
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self.proc = subprocess.Popen(args, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, encoding="utf-8", errors="replace", bufsize=1,
                                         env=env, creationflags=CREATE_NO_WINDOW)
        except Exception as e:
            return False, f"启动失败：{e}"
        write_text(PY_RECORD, py)
        if comfy:
            write_text(COMFY_RECORD, os.path.abspath(comfy))
        add_local(f"已启动服务端子进程 PID {self.proc.pid}（python: {py}）", "ok")
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        return True, "已启动"

    def _reader(self, proc):
        try:
            for line in proc.stdout:
                if line:
                    LOG_Q.put(line.rstrip("\n"))
        except Exception:
            pass
        finally:
            LOG_Q.put(("__EXIT__", None))

    def stop(self) -> tuple[bool, str]:
        if not self.running():
            self._cleanup()
            return True, "服务没在运行"
        try:
            assert self.proc is not None
            self.proc.terminate()
            try:
                self.proc.wait(timeout=6)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            add_local("已停止服务端", "warn")
        except Exception as e:
            add_local(f"停止失败：{e}", "err")
        self._cleanup()
        return True, "已停止"

    def _cleanup(self):
        self.proc = None
        self.health = None

    def python_path(self) -> str:
        """面板记过 / 按 ComfyUI 目录识别出来的那个 ComfyUI python（LLM 要和它同一个）。"""
        self.reload_cfg()
        py = clean_path(read_text(PY_RECORD))
        if py and os.path.isfile(py):
            return py
        comfy = clean_path(self.cfg.get("comfyui_dir")).replace("\\", "/")
        found, _why = best_python(comfy)
        return found or ""

    # ---- HTTP ---- #
    def base_url(self) -> str:
        port = int((self.cfg.get("server") or {}).get("port", 8199) or 8199)
        return f"http://127.0.0.1:{port}"

    def _headers(self):
        tok = str((self.cfg.get("server") or {}).get("token") or "")
        return {"X-Auth-Token": tok} if tok else {}

    def get_health(self):
        # 轻量 health：不带模型清单（共享盘上 stat 上千个文件会把轮询拖到超时，状态就停在「—」）
        try:
            req = urllib.request.Request(self.base_url() + "/health?light=1", headers=self._headers())
            with urllib.request.urlopen(req, timeout=8) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            return None

    def get_models_cached(self, ttl: float = 20.0):
        """模型清单单独取，并缓存 ttl 秒——轮询只走轻量 health，不会每条都扫盘。"""
        import time as _t
        c = getattr(self, "_models_cache", None)
        if c and (c[0] + ttl) > _t.time():
            return c[1]
        m = self.get_models()
        if m:
            self._models_cache = (_t.time(), m)
        return m

    def get_models(self):
        try:
            req = urllib.request.Request(self.base_url() + "/models", headers=self._headers())
            with urllib.request.urlopen(req, timeout=5) as r:
                j = json.loads(r.read().decode("utf-8"))
                return j.get("data") if j.get("ok") else None
        except Exception:
            return None

    def reload_models(self):
        """让服务端重读 config.json（扫描/改目录后不用重启）。"""
        try:
            req = urllib.request.Request(self.base_url() + "/reload", headers=self._headers())
            with urllib.request.urlopen(req, timeout=8) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def rpc(self, op: str, params: dict | None = None, timeout: float = 480):
        body = json.dumps({"op": op, "params": params or {}}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        headers.update(self._headers())
        req = urllib.request.Request(self.base_url() + "/rpc", data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            return {"ok": False, "error": str(e)}


RUNNER = Runner()


def _drain_logs():
    while True:
        try:
            item = LOG_Q.get_nowait()
        except queue.Empty:
            return
        if isinstance(item, tuple):
            if item[0] == "__EXIT__":
                SRV_LOG.append({"ts": now(), "level": "info", "text": "服务端日志流结束"})
                add_local("服务端进程已退出", "warn")
            continue
        line = str(item)
        low = line.lower()
        if "!!" in line or "error" in low or "traceback" in low or "失败" in line:
            lvl = "err"
        elif "warn" in low or "警告" in line or "注意" in line:
            lvl = "warn"
        elif "已启动" in line or "[ok]" in low or "完成" in line:
            lvl = "ok"
        else:
            lvl = "info"
        SRV_LOG.append({"ts": now(), "level": lvl, "text": line})


# --------------------------------------------------------------------------- #
# HTTP 服务
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, obj, code=200, ctype="application/json; charset=utf-8"):
        body = obj if isinstance(obj, bytes) else json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except Exception:
            pass

    def _body(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n <= 0:
                return {}
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    # ---- GET ---- #
    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(HTML.encode("utf-8"), ctype="text/html; charset=utf-8")
            return
        if path == "/api/state":
            self._send(self.state())
            return
        self._send({"ok": False, "error": "未知路径"}, 404)

    def state(self) -> dict:
        _drain_logs()
        r = RUNNER
        r.reload_cfg()
        running = r.running()
        health = r.get_health() if running else None
        r.gpu_n += 1
        if r.gpu_n % 3 == 1:
            r.gpu_cache = gpu_stats() or {}
        gpu = r.gpu_cache or {}
        ru, rt = ram_gb()
        ip = (lan_ips() or ["127.0.0.1"])[0]
        port = int((r.cfg.get("server") or {}).get("port", 8199) or 8199)
        # 条目以 config.json 为准（刚扫描出来的不会被服务端旧快照盖掉），exists 用服务端判定优先
        cfg_models = r.cfg.get("models") or {}
        models = {k: {n: {"exists": None} for n in (cfg_models.get(k) or {})} for k in ("clip", "vae")}
        srv_models = ((health or {}).get("data", {}) or {}).get("models") if health else None
        if (not srv_models) and running:
            srv_models = r.get_models_cached()
        if isinstance(srv_models, dict):
            for k in ("clip", "vae"):
                for name, info in (srv_models.get(k) or {}).items():
                    if name in models.get(k, {}):
                        if isinstance(info, dict) and "exists" in info:
                            models[k][name]["exists"] = info["exists"]
                        models[k][name].update({x: y for x, y in (info or {}).items() if x != "exists"})
                    else:
                        models.setdefault(k, {})[name] = info
        return {
            "ok": True,
            "cfg": r.cfg,
            "running": running,
            "pid": r.proc.pid if r.proc else None,
            "health": (health or {}).get("data"),
            "llm": ((health or {}).get("data", {}) or {}).get("llm") or {},
            "llm_models": {"llm": (srv_models or {}).get("llm") or {},
                           "llm_vision": (srv_models or {}).get("llm_vision") or {}},
            "gpu": gpu,
            "ram": {"used": ru, "total": rt},
            "ip": ip,
            "url": f"http://{ip}:{port}",
            "models": models,
            "cpu": (platform.processor() or platform.machine() or "CPU").strip(),
            "logs": {"local": list(LOCAL_LOG)[-250:], "srv": list(SRV_LOG)[-250:]},
            "python": clean_path(read_text(PY_RECORD)),
            "platform": platform.platform(),
        }

    # ---- POST ---- #
    def do_POST(self):
        path = self.path.split("?")[0]
        body = self._body()
        try:
            if path == "/api/config":
                cfg = self._save_cfg(body)
                return self._send({"ok": True, "cfg": cfg})
            if path == "/api/start":
                ok, msg = RUNNER.start(bool(body.get("mock")))
                add_local(msg, "ok" if ok else "err")
                return self._send({"ok": ok, "message": msg})
            if path == "/api/stop":
                ok, msg = RUNNER.stop()
                return self._send({"ok": ok, "message": msg})
            if path == "/api/unload":
                if not RUNNER.running():
                    return self._send({"ok": False, "error": "服务没在运行"})
                r = RUNNER.rpc("unload", {})
                ok = bool(r.get("ok"))
                add_local("已请求卸载全部模型" if ok else f"卸载失败：{r.get('error')}", "ok" if ok else "err")
                return self._send({"ok": ok, "result": r})
            # ---------------- LLM（远端 ComfyUI 的 GGUF） ---------------- #
            if path == "/api/llm_load":
                if not RUNNER.running():
                    return self._send({"ok": False, "error": "服务没在运行，先点「启动服务」"})
                model = str(body.get("model") or "")
                mmproj = str(body.get("mmproj") or "")
                n_ctx = int(body.get("n_ctx") or 4096)
                if not model:
                    return self._send({"ok": False, "error": "先选一个 GGUF 模型"})

                def work():
                    add_local(f"正在加载 GGUF：{model} → 服务端 ComfyUI 的显存里……", "info")
                    r = RUNNER.rpc("llm_load", {"model": model, "mmproj": mmproj, "n_ctx": n_ctx}, timeout=1800)
                    if r.get("ok"):
                        d = (r.get("data") or {})
                        add_local(f"GGUF 加载完成：{d.get('model')} 上下文 {d.get('n_ctx')}，"
                                  f"耗时 {d.get('load_seconds')}s，handler={d.get('handler')}", "ok")
                    else:
                        add_local(f"GGUF 加载失败：{r.get('error')}", "err")

                threading.Thread(target=work, daemon=True).start()
                return self._send({"ok": True, "message": "已提交加载请求（日志里看进度）"})
            if path == "/api/llm_unload":
                if not RUNNER.running():
                    return self._send({"ok": False, "error": "服务没在运行"})
                r = RUNNER.rpc("llm_unload", {}, timeout=120)
                ok = bool(r.get("ok"))
                add_local("已卸载 GGUF（显存已释放）" if ok else f"卸载 GGUF 失败：{r.get('error')}",
                          "ok" if ok else "err")
                return self._send(r)
            if path == "/api/llm_install":
                ry = RUNNER.python_path() or best_python(clean_path(load_json(CFG_PATH, {}).get("comfyui_dir")))[0]
                if not ry:
                    return self._send({"ok": False, "error": "先选好「本机 ComfyUI 的 Python」"})

                def pip_work():
                    add_local(f"正在给 {ry} 安装 llama-cpp-python（自动挑预编译 wheel，可能要几分钟）……", "info")
                    try:
                        import llm as llm_mod
                        res = llm_mod.install_llama(ry, log=lambda s: add_local(str(s), "info"))
                    except Exception as e:
                        add_local(f"安装流程异常：{e}", "err")
                        res = {"ok": False, "error": str(e)}
                    if res.get("ok"):
                        add_local(f"llama-cpp-python {res.get('version')} 装好了（{res.get('how')}），可以加载 GGUF 了",
                                  "ok")
                        try:
                            RUNNER.rpc("llm_status", {}, timeout=60)
                        except Exception:
                            pass
                    else:
                        add_local("llama-cpp-python 没装成功。兜底：去 https://github.com/JamePeng/"
                                  "llama-cpp-python/releases 下对应 cp版本+cu版本 的 win_amd64.whl，"
                                  "再点「装本地 wheel」", "err")

                threading.Thread(target=pip_work, daemon=True).start()
                return self._send({"ok": True, "message": "已开始安装 llama-cpp-python（看日志）"})
            if path == "/api/llm_install_file":
                ry = RUNNER.python_path() or best_python(clean_path(load_json(CFG_PATH, {}).get("comfyui_dir")))[0]
                whl = str(body.get("path") or "").strip().strip('"')
                if not ry:
                    return self._send({"ok": False, "error": "先选好「本机 ComfyUI 的 Python」"})
                if not whl or not os.path.isfile(whl):
                    return self._send({"ok": False, "error": "这个 .whl 路径不存在"})

                def wheel_work():
                    add_local(f"用本地 wheel 安装：{whl}", "info")
                    try:
                        import llm as llm_mod
                        res = llm_mod.install_llama(ry, log=lambda s: add_local(str(s), "info"), wheel_url=whl)
                    except Exception as e:
                        res = {"ok": False, "error": str(e)}
                    add_local(("装好了：" + str(res.get("version"))) if res.get("ok")
                              else ("没装上：" + str(res.get("error") or "看上面的日志")),
                              "ok" if res.get("ok") else "err")

                threading.Thread(target=wheel_work, daemon=True).start()
                return self._send({"ok": True, "message": "已开始用本地 wheel 安装（看日志）"})
            if path == "/api/preload":
                if not RUNNER.running():
                    return self._send({"ok": False, "error": "服务没在运行"})
                kind = body.get("kind")
                names = body.get("models") or []
                if not kind or not names:
                    return self._send({"ok": False, "error": "没选模型"})

                def work():
                    for n in names:
                        add_local(f"正在预加载 {kind}/{n} ……", "info")
                        r = RUNNER.rpc("load", {"kind": kind, "model": n}, timeout=900)
                        if r.get("ok"):
                            add_local(f"预加载完成 {kind}/{n}", "ok")
                        else:
                            add_local(f"预加载失败 {kind}/{n}：{r.get('error')}", "err")

                threading.Thread(target=work, daemon=True).start()
                return self._send({"ok": True, "message": "已提交预加载请求"})
            if path == "/api/scan":
                added = self._scan_add()
                return self._send({"ok": True, "message": f"新增 {added} 个条目"})
            if path == "/api/delete":
                kind, name = body.get("kind"), body.get("name")
                cfg = load_json(CFG_PATH, RUNNER.cfg)
                (cfg.get("models", {}).get(kind) or {}).pop(name, None)
                save_json(CFG_PATH, cfg)
                add_local(f"已删除条目 {kind}/{name}", "warn")
                return self._send({"ok": True})
            if path == "/api/find":
                what = body.get("what")
                comfy = str(load_json(CFG_PATH, {}).get("comfyui_dir") or "")
                if what == "comfy":
                    found = find_comfy_dirs()
                    if found:
                        self._save_cfg({**load_json(CFG_PATH, {}), "comfyui_dir": found[0]})
                        py, why = best_python(found[0])
                        if py:
                            write_text(PY_RECORD, py)
                        add_local(f"自动找到 ComfyUI：{found[0]}", "ok")
                        return self._send({"ok": True, "value": found[0], "python": py})
                    return self._send({"ok": False, "error": "没在常见位置找到 ComfyUI"})
                if not comfy:
                    return self._send({"ok": False, "error": "先选 ComfyUI 路径，python 按这个目录识别"})
                py, why = best_python(comfy)
                if py:
                    write_text(PY_RECORD, py)
                    add_local(f"按 ComfyUI 路径识别到 python{'（自带 torch）' if why == 'ok' else '（没看到 torch）'}：{py}", "ok")
                    return self._send({"ok": True, "value": py})
                return self._send({"ok": False, "error": "这个 ComfyUI 目录里没有 python_embeded\\python.exe 或 .venv\\Scripts\\python.exe"})
            if path == "/api/open":
                target = str(load_json(CFG_PATH, {}).get("comfyui_dir") or HERE)
                try:
                    if os.name == "nt":
                        os.startfile(target)  # type: ignore[attr-defined]
                    else:
                        subprocess.Popen(["xdg-open", target])
                except Exception as e:
                    return self._send({"ok": False, "error": str(e)})
                return self._send({"ok": True})
            if path == "/api/clearlog":
                LOCAL_LOG.clear()
                SRV_LOG.clear()
                return self._send({"ok": True})
            self._send({"ok": False, "error": "未知接口"}, 404)
        except Exception as e:
            add_local(f"接口异常：{type(e).__name__}: {e}", "err")
            self._send({"ok": False, "error": str(e)}, 500)

    # ---- 配置 ---- #
    def _save_cfg(self, patch: dict):
        cfg = load_json(CFG_PATH, {})
        srv = cfg.setdefault("server", {})
        for k in ("host", "port", "token", "max_request_mb"):
            if k in patch:
                srv[k] = patch[k]
        if "port" in patch:
            try:
                srv["port"] = int(patch["port"])
            except Exception:
                srv["port"] = 8199
        for k in ("comfyui_dir", "device", "idle_unload_seconds", "clip_model_dir", "vae_model_dir",
                  "llm_model_dir", "llm_python"):
            if k in patch:
                cfg[k] = clean_path(patch[k])
        if "comfyui_dir" in patch:
            cfg["comfyui_dir"] = clean_path(patch["comfyui_dir"]).replace("\\", "/")
        if patch.get("python"):
            write_text(PY_RECORD, clean_path(patch["python"]))
        save_json(CFG_PATH, cfg)
        RUNNER.cfg = cfg
        add_local("配置已保存到 config.json", "ok")
        return cfg

    def _scan_add(self) -> int:
        cfg = load_json(CFG_PATH, {})
        comfy = clean_path(cfg.get("comfyui_dir")).replace("\\", "/")
        models_root = os.path.join(comfy, "models") if comfy else ""
        added = 0

        def rel(p):
            if models_root and os.path.abspath(p).startswith(os.path.abspath(models_root)):
                return os.path.relpath(p, models_root).replace(os.sep, "/")
            return p.replace("\\", "/")

        for kind, sub in (("clip", "clip_model_dir"), ("vae", "vae_model_dir")):
            d = clean_path(cfg.get(sub))
            if not d or not os.path.isdir(d):
                continue
            entries = cfg.setdefault("models", {}).setdefault(kind, {})
            existing = set()
            for e in entries.values():
                files = e.get("files") if isinstance(e, dict) else ([e] if isinstance(e, str) else [])
                for f in files or []:
                    existing.add(os.path.basename(str(f)).lower())
            for dirpath, _dirs, files in os.walk(d):
                if dirpath[len(d):].count(os.sep) > 2:
                    continue
                for fn in files:
                    low = fn.lower()
                    if not low.endswith(MODEL_EXTS) or low.startswith("put_"):
                        continue
                    if kind == "clip" and any(h in low for h in VISION_HINTS):
                        continue
                    if low in existing:
                        continue
                    key = "".join(ch if ch.isalnum() or ch in "._-" else "_"
                                  for ch in os.path.splitext(fn)[0]).lower()
                    entry = {"files": [rel(os.path.join(dirpath, fn))], "note": "面板扫描加入"}
                    # 不写死 clip_type：服务端加载时会按权重/文件名自动识别（Qwen-VL 必须走 qwen_image）
                    entries[key] = entry
                    existing.add(low)
                    added += 1
        save_json(CFG_PATH, cfg)
        RUNNER.cfg = cfg
        if RUNNER.running():
            r = RUNNER.reload_models()
            if r.get("ok"):
                d = (r.get("data") or {}).get("models") or {}
                add_local(f"服务端已重读配置：CLIP {d.get('clip', 0)} 条，VAE {d.get('vae', 0)} 条", "ok")
            else:
                add_local(f"服务端重读配置失败：{r.get('error')}（可停服务再启动一次）", "warn")
        add_local(f"扫描完成，新增 {added} 个条目", "ok" if added else "warn")
        return added


# --------------------------------------------------------------------------- #
# 前端
# --------------------------------------------------------------------------- #
HTML = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>远程 CLIP / VAE 模型服务端</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#f2f2f2;color:#1a1a1a;font:13px/1.5 "Segoe UI","Microsoft YaHei UI",sans-serif}
#wrap{display:flex;gap:10px;padding:10px 12px 6px;height:calc(100vh - 34px)}
#left{width:660px;min-width:640px;display:flex;flex-direction:column;gap:8px}
#right{flex:1;min-width:380px;display:flex;flex-direction:column;gap:6px}
h1{margin:0 0 2px;font-size:19px}
.sub{color:#666;margin:0 0 4px}
.url{color:#127a12;font-weight:700}
.dev{color:#666;margin-left:14px}
fieldset{background:#fff;border:1px solid #d6d6d6;border-radius:3px;padding:8px 10px 10px;margin:0}
legend{font-size:12px;color:#333;padding:0 4px}
.row{display:flex;align-items:center;gap:6px;margin:4px 0}
.row label{white-space:nowrap;color:#222}
input[type=text],input[type=number],select{border:1px solid #bdbdbd;border-radius:2px;padding:3px 5px;font:12px "Consolas",monospace;background:#fff}
input[type=text]:focus,select:focus{outline:1px solid #7aa7d8}
.grow{flex:1;min-width:0}
button{border:1px solid #b9b9b9;background:#fbfbfb;border-radius:3px;padding:4px 11px;font:12px "Microsoft YaHei UI",sans-serif;cursor:pointer}
button:hover{background:#f0f4fa;border-color:#8fb3dc}
button:disabled{opacity:.5;cursor:default}
button.primary{background:#eaf3ea;border-color:#9cc79c}
button.danger{background:#fdeeee;border-color:#e0b0b0}
.hint{color:#888;font-size:11px;margin:2px 0 0 4px}
.chk{display:flex;align-items:center;gap:6px;color:#222;margin-top:4px}
.chk small{color:#888}
.btns{display:flex;gap:6px;margin-top:8px}
.stline{margin:2px 0;color:#222}
.bar{height:12px;background:#e9e9e9;border:1px solid #d2d2d2;border-radius:6px;overflow:hidden;flex:1}
.bar>i{display:block;height:100%;background:linear-gradient(90deg,#6aa9e9,#3d7fd0);width:0}
.bar.gpu>i{background:linear-gradient(90deg,#7ad07a,#3f9a3f)}
.barwrap{display:flex;align-items:center;gap:6px;margin-top:6px}
.barlbl{width:auto;color:#333;white-space:nowrap;font:12px "Consolas",monospace}
.lists{display:flex;gap:8px}
.lists>div{flex:1;min-width:0}
.listbox{border:1px solid #bdbdbd;background:#fff;height:104px;overflow:auto;font:12px "Consolas",monospace}
.listbox div{padding:1px 6px;cursor:default;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.listbox div.sel{background:#cfe3f8}
.listbox div.miss{color:#b00020}
#loaded{color:#127a12;margin-left:8px}
#tabs{display:flex;gap:0;align-items:center}
#tabs .t{padding:3px 12px;border:1px solid #cfcfcf;border-bottom:none;background:#ececec;cursor:pointer;font-size:12px;border-radius:3px 3px 0 0}
#tabs .t.on{background:#fff;font-weight:600}
#clearlog{margin-left:auto}
#console{flex:1;background:#111;color:#e8e8e8;border:1px solid #333;border-radius:0 3px 3px 3px;
  overflow:auto;font:12px/1.45 "Consolas",monospace;padding:6px 8px;white-space:pre-wrap;word-break:break-all}
#console .ts{color:#787878}
#console .err{color:#ff6b6b}#console .warn{color:#ffc857}#console .ok{color:#7bd88f}#console .info{color:#cfe8ff}
#status{height:26px;padding:0 12px;color:#333;font-size:12px;display:flex;align-items:center}
body.fs #wrap{height:calc(100vh - 30px)}
@media (max-width:1250px){
  #wrap{flex-direction:column;height:auto}
  #left{width:100%;min-width:0}
  #right{min-height:420px}
  #console{min-height:380px}
  .lists{flex-wrap:wrap}
  .btns{flex-wrap:wrap}
  .row{flex-wrap:wrap}
}
</style></head><body>
<div style="padding:8px 12px 0">
  <h1>远程 CLIP / VAE 模型服务端</h1>
  <div class="sub">双机 ComfyUI 分离加载：客户端只跑 UNet，服务端（本机）跑文本编码与 VAE 编解码</div>
  <div><span class="url" id="urlline">ComfyUI 插件里的「服务端地址」填： http://127.0.0.1:8199</span><span class="dev" id="devline"></span></div>
</div>
<div id="wrap">
  <div id="left">
    <fieldset><legend>服务配置</legend>
      <div class="row">
        <label>监听地址</label><input type="text" id="host" style="width:110px">
        <label>端口</label><input type="number" id="port" style="width:74px">
        <label>计算设备</label><span class="hint" id="devhint">自动（NVIDIA/AMD/Intel 交给 ComfyUI 判断）</span>
      </div>
      <div class="row"><label>CLIP 模型目录</label><input type="text" id="clipdir" class="grow"><button onclick="pickDir('clipdir')">浏览...</button></div>
      <div class="row"><label>VAE 模型目录</label><input type="text" id="vaedir" class="grow"><button onclick="pickDir('vaedir')">浏览...</button></div>
      <div class="row"><label>GGUF 模型目录</label><input type="text" id="llmdir" class="grow"><button onclick="pickDir('llmdir')">浏览...</button></div>
      <div class="hint">留空 = 自动用 &lt;ComfyUI&gt;\models\LLM、models\llm（「提示词优化」节点里的模型就从这里来）</div>
      <div class="row"><label>ComfyUI 路径</label><input type="text" id="comfy" class="grow">
        <button onclick="pickComfy()">浏览...</button><button onclick="findIt('comfy')">自动找</button></div>
      <div class="chk"><input type="checkbox" checked disabled><span>限制客户端只能读取上面两个目录里的模型（更安全）</span><small>（本包天生如此：客户端只能加载 config.json 里登记的条目）</small></div>
      <div class="row" style="margin-top:8px"><label>本机 ComfyUI 的 Python</label><input type="text" id="python" class="grow">
        <button onclick="pickPython()">浏览...</button><button onclick="findIt('python')">自动找</button></div>
      <div class="hint">留空自动查找（python_embeded\python.exe 或 .venv\Scripts\python.exe）</div>
      <div class="chk"><input type="checkbox" id="prefer"><span>优先用 ComfyUI 自带的 Python 启动（无窗口；Qwen-Image / Flux / Wan 这类模型必须）</span></div>
      <div class="btns">
        <button class="primary" id="bstart" onclick="startSrv(false)">▶ 启动服务</button>
        <button style="width:104px" class="danger" id="bstop" onclick="stopSrv()">■ 停止服务</button>
        <button onclick="unload()">清空显存 / 卸载模型</button>
        <button onclick="refresh()">刷新模型列表</button>
        <button onclick="openDir()">打开目录</button>
      </div>
    </fieldset>

    <fieldset><legend>运行状态</legend>
      <div class="stline" id="calc">计算方式: — &nbsp; 加载后端: comfy &nbsp; 设备: —</div>
      <div class="stline" id="gpu">GPU: — &nbsp; 显存: — &nbsp; 利用率: —</div>
      <div class="stline" id="cpu">CPU: — &nbsp; 内存: —</div>
      <div class="barwrap"><span class="barlbl" style="width:26px">显存</span><div class="bar"><i id="barvram"></i></div><span class="barlbl" id="lblvram" style="width:300px">0 / 0 GB &nbsp; torch 使用 — MB</span></div>
      <div class="barwrap"><span class="barlbl" style="width:78px">GPU 利用率</span><div class="bar gpu"><i id="barutil"></i></div><span class="barlbl" id="lblutil">—</span></div>
    </fieldset>

    <fieldset><legend>服务端模型</legend>
      <div class="lists">
        <div><div style="color:#555;font-size:12px;margin-bottom:2px">服务端 CLIP 模型</div><div class="listbox" id="listclip"></div></div>
        <div><div style="color:#555;font-size:12px;margin-bottom:2px">服务端 VAE 模型</div><div class="listbox" id="listvae"></div></div>
      </div>
      <div class="row" style="margin-top:6px">
        <button onclick="preload('clip')">预加载选中的 CLIP</button>
        <button onclick="preload('vae')">预加载选中的 VAE</button>
        <button onclick="scanAdd()">扫描目录→加条目</button>
        <button onclick="delEntry()">删除条目</button>
        <span id="loaded">已加载: -</span>
      </div>
    </fieldset>

    <fieldset><legend>提示词优化 LLM（跑在本机 ComfyUI 的 Python 里，模型取上面 GGUF 目录）</legend>
      <div class="lists">
        <div><div style="color:#555;font-size:12px;margin-bottom:2px">GGUF 文本模型</div><div class="listbox" id="listllm"></div></div>
        <div><div style="color:#555;font-size:12px;margin-bottom:2px">视觉投影 mmproj（可选）</div><div class="listbox" id="listmmproj"></div></div>
      </div>
      <div class="row" style="margin-top:6px">
        <button class="primary" onclick="llmLoad()">加载 GGUF 到显存</button>
        <button onclick="llmUnload()">卸载 GGUF</button>
        <button onclick="llmInstall()">安装 llama-cpp-python</button>
        <span style="color:#555">上下文 <input type="number" id="nctx" value="4096" style="width:82px"></span>
      </div>
      <div class="stline" id="llmstate" style="margin-top:6px">LLM 状态: —</div>
      <div class="stline" id="llmrt">实时: 未运行</div>
      <div class="stline" id="llmerr" style="color:#b00020"></div>
    </fieldset>
  </div>

  <div id="right">
    <div style="color:#333;margin:0 0 0 2px">日志</div>
    <div id="tabs"><div class="t on" data-t="local">本机日志</div><div class="t" data-t="srv">服务端日志（实时）</div>
      <button id="clearlog" onclick="clearLog()">清屏</button></div>
    <div id="console"></div>
  </div>
</div>
<div id="status">未运行</div>

<script>
let STATE=null, TAB="local", SEL={clip:new Set(),vae:new Set(),llm:new Set(),llm_vision:new Set()}, LASTLEN={local:0,srv:0};
const $=id=>document.getElementById(id);
function esc(s){return (s+"").replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
async function api(path,body){
  const o={headers:{'Content-Type':'application/json'}};
  if(body!==undefined){o.method='POST';o.body=JSON.stringify(body);}
  const r=await fetch(path,o); return await r.json();
}
function fill(){
  const s=STATE; if(!s)return;
  if(document.activeElement.tagName!=='INPUT'){
    $('host').value=s.cfg.server?.host??'0.0.0.0';
    $('port').value=s.cfg.server?.port??8199;
    $('comfy').value=s.cfg.comfyui_dir||'';
    $('python').value=s.python||'';
    $('clipdir').value=s.cfg.clip_model_dir||'';
    $('vaedir').value=s.cfg.vae_model_dir||'';
    $('llmdir').value=s.cfg.llm_model_dir||'';
  }
  $('urlline').textContent='ComfyUI 插件里的「服务端地址」填： '+s.url+'   （本机 IP: '+s.ip+'）';
  const h=s.health||{};
  const dev=s.running?(h.device||(h.cuda_available?'cuda':'cpu')):'—';
  $('devline').textContent=h.device_name?('设备: '+h.device_name+(h.mock?'（联调 MOCK）':'')):'';
  $('calc').innerHTML='计算方式: '+dev+' &nbsp; 加载后端: comfy &nbsp; 设备: '+(h.device_name||'—');
  let vr=h.vram||{}; let tot=+(vr.total_gb||0), used=Math.max(0,tot-+(vr.free_gb||0));
  if(!tot&&s.gpu&&s.gpu.total_mb){tot=s.gpu.total_mb/1024; used=s.gpu.used_mb/1024;}
  const util=s.gpu&&s.gpu.util!=null?s.gpu.util:null;
  $('gpu').innerHTML='GPU: '+(h.device_name||'—')+' [cuda] &nbsp; 显存: '+used.toFixed(1)+' / '+tot.toFixed(1)+' GB &nbsp; 利用率: '+(util!=null?util+'%':'—');
  $('cpu').innerHTML='CPU: '+(s.cpu||'—')+' &nbsp; 内存: '+(s.ram&&s.ram.total?(s.ram.used.toFixed(1)+' / '+s.ram.total.toFixed(1)+' GB'):'—');
  $('barvram').style.width=(tot?used/tot*100:0)+'%';
  $('lblvram').textContent=used.toFixed(1)+' / '+tot.toFixed(1)+' GB   torch 使用 '+(h.torch_allocated_mb!=null?h.torch_allocated_mb+' MB':'—');
  $('barutil').style.width=(util!=null?util:0)+'%';
  $('lblutil').textContent=(util!=null?Math.round(util)+'%':'—');
  const ld=h.loaded||{}; const parts=[];
  for(const k of ['clip','vae']) if(ld[k]&&ld[k].length) parts.push(k+': '+ld[k].join(', '));
  $('loaded').textContent='已加载: '+(parts.length?parts.join(' | '):'-');
  $('bstart').disabled=s.running; $('bstop').disabled=!s.running;
  $('status').textContent=s.running?('运行中（PID '+s.pid+'）· 端口 '+(s.cfg.server?.port||8199)+(h.uptime_s?' · 已运行 '+Math.round(h.uptime_s)+'s':'')):'未运行';
  renderList('clip',s.models.clip||{}); renderList('vae',s.models.vae||{});
  fillLLM(s);
}
function fillLLM(s){
  const L=s.llm||{}, M=s.llm_models||{};
  renderLLMList('listllm','llm',M.llm||{});
  renderLLMList('listmmproj','llm_vision',M.llm_vision||{});
  const st=L.state||(s.running?'—':'未启动');
  $('llmstate').textContent='LLM 状态: '+st
    +(L.model?('  模型: '+L.model):'')
    +(L.mmproj?('  mmproj: '+L.mmproj):'')
    +(L.n_ctx?('  上下文: '+L.n_ctx):'')
    +(L.handler?('  handler: '+L.handler):'')
    +(L.load_seconds?('  加载耗时: '+L.load_seconds+'s'):'')
    +(L.llama_cpp?('  llama-cpp: '+L.llama_cpp):'');
  const p=L.progress||{}, last=L.last||{};
  const bits=[];
  if(p.tokens!=null) bits.push('生成中: '+p.tokens+' tokens  '+(p.tps||0)+' tok/s  '+p.elapsed+'s');
  else if(last.tokens!=null) bits.push('上次: '+last.tokens+' tokens  '+(last.tps||0)+' tok/s  '+last.seconds+'s'+(last.images?('  '+last.images+' 张图'):'')+(last.at?('  '+last.at):''));
  bits.push('Python: '+(L.python||s.python||'—'));
  bits.push('设备: '+(L.device_name||'—'));
  $('llmrt').textContent='实时: '+bits.join('  |  ');
  $('llmerr').textContent=L.error?('错误: '+L.error):(L.detail?('提示: '+L.detail):'');
}
function renderLLMList(id,kind,items){
  const box=$(id); const names=Object.keys(items);
  if(!names.length){box.innerHTML='<div class="miss" style="color:#888">（没有 GGUF：检查上面的 GGUF 模型目录，或点「刷新模型列表」）</div>';return;}
  box.innerHTML=names.map(n=>{
    const sel=SEL[kind].has(n);
    const sz=items[n]&&items[n].size_gb?('  '+items[n].size_gb+'GB'):'';
    return '<div class="'+(sel?'sel':'')+'" data-k="'+kind+'" data-n="'+esc(n)+'">'+esc(n)+sz+'</div>';
  }).join('');
}
function renderList(kind,items){
  const box=$(kind==='clip'?'listclip':'listvae'); const names=Object.keys(items);
  if(!names.length){box.innerHTML='<div class="miss" style="color:#888">（还没有条目：点「扫描目录→加条目」）</div>';return;}
  box.innerHTML=names.map(n=>{
    const miss=items[n]&&items[n].exists===false;
    const sel=SEL[kind].has(n);
    return '<div class="'+(miss?'miss':'')+(sel?' sel':'')+'" data-k="'+kind+'" data-n="'+esc(n)+'">'+esc(n)+(miss?'   ✗ 文件缺失':'')+'</div>';
  }).join('');
}
document.addEventListener('click',e=>{
  const d=e.target.closest('.listbox div[data-n]'); if(!d)return;
  const k=d.dataset.k,n=d.dataset.n;
  if(!e.ctrlKey&&!e.metaKey) SEL[k].clear();
  if(SEL[k].has(n)) SEL[k].delete(n); else SEL[k].add(n);
  fill();
});
document.addEventListener('dblclick',e=>{
  const d=e.target.closest('.listbox div[data-n]'); if(!d)return;
  SEL[d.dataset.k]=new Set([d.dataset.n]); loadModels(d.dataset.k);
});
async function saveCfg(){
  await api('/api/config',{host:$('host').value,port:$('port').value,
    comfyui_dir:$('comfy').value,python:$('python').value,clip_model_dir:$('clipdir').value,vae_model_dir:$('vaedir').value,
    llm_model_dir:$('llmdir').value});
}
async function startSrv(mock){ await saveCfg(); const r=await api('/api/start',{mock:!!mock}); if(!r.ok)alert(r.message||r.error); refresh(); }
async function stopSrv(){ await api('/api/stop',{}); refresh(); }
async function unload(){ const r=await api('/api/unload',{}); if(!r.ok)alert(r.error||'失败'); refresh(); }
async function refresh(){ STATE=await api('/api/state'); fill(); }
function loadModels(kind){
  const names=[...SEL[kind]]; if(!names.length){alert('先在列表里选一条（Ctrl 可多选）');return;}
  api('/api/preload',{kind,models:names}).then(r=>{ if(!r.ok)alert(r.error||'失败'); refresh(); });
}
function preload(kind){ loadModels(kind); }
async function delEntry(){
  for(const k of ['clip','vae']) for(const n of [...SEL[k]]){
    if(!confirm('从 config.json 里删掉 '+k+' 条目 '+n+' ？'))return;
    await api('/api/delete',{kind:k,name:n}); SEL[k].clear();
  }
  refresh();
}
async function scanAdd(){ const r=await api('/api/scan',{}); if(r.message)alert(r.message); refresh(); }
async function llmLoad(){
  const names=[...SEL.llm]; if(!names.length){alert('先在「GGUF 文本模型」列表里选一条');return;}
  const mm=[...SEL.llm_vision];
  const r=await api('/api/llm_load',{model:names[0],mmproj:mm[0]||'',n_ctx:Number($('nctx').value||4096)});
  if(!r.ok)alert(r.message||r.error); refresh();
}
async function llmUnload(){ const r=await api('/api/llm_unload',{}); if(!r.ok)alert(r.error||'失败'); refresh(); }
async function llmInstall(){
  if(!confirm('给「本机 ComfyUI 的 Python」安装 llama-cpp-python（需要联网，几分钟）？'))return;
  const r=await api('/api/llm_install',{}); if(!r.ok)alert(r.error||'失败'); else alert(r.message||'已开始'); refresh();
}
async function findIt(what){
  const r=await api('/api/find',{what}); if(!r.ok){alert(r.error);return;}
  if(what==='comfy'){$('comfy').value=r.value; if(r.python)$('python').value=r.python;} else {$('python').value=r.value;}
  saveCfg(); refresh();
}
async function openDir(){ const r=await api('/api/open',{}); if(!r.ok)alert(r.error); }
function pickDir(id){ const cur=$(id).value; const p=prompt('输入目录路径（可粘贴）',cur||''); if(p!==null)$(id).value=p; }
function pickComfy(){ const p=prompt('输入 ComfyUI 源码根目录（里面有 comfy 文件夹和 main.py）',$('comfy').value||''); if(p!==null){$('comfy').value=p;} }
function pickPython(){ const p=prompt('输入 ComfyUI 的 python.exe 路径',$('python').value||''); if(p!==null){$('python').value=p;} }
async function clearLog(){ await api('/api/clearlog',{}); $('console').innerHTML=''; LASTLEN={local:0,srv:0}; }
document.querySelectorAll('#tabs .t').forEach(t=>t.onclick=()=>{
  TAB=t.dataset.t; document.querySelectorAll('#tabs .t').forEach(x=>x.classList.toggle('on',x===t)); paint(true);
});
function paint(force){
  if(!STATE)return;
  const logs=STATE.logs[TAB]||[];
  if(force) LASTLEN[TAB]=0;
  if(logs.length<LASTLEN[TAB]) LASTLEN[TAB]=0;
  const box=$('console');
  if(!force&&LASTLEN[TAB]>0&&logs.length===LASTLEN[TAB])return;
  const add=logs.slice(LASTLEN[TAB]);
  if(!add.length)return;
  if(force) box.innerHTML='';
  const atBottom=box.scrollTop+box.clientHeight>=box.scrollHeight-8;
  for(const l of add) box.insertAdjacentHTML('beforeend',
    '<span class="ts">['+l.ts+']</span> <span class="'+l.level+'">'+esc(l.text)+'</span>\n');
  LASTLEN[TAB]=logs.length;
  if(atBottom) box.scrollTop=box.scrollHeight;
}
async function loop(){
  try{ STATE=await api('/api/state'); fill(); paint(false);}catch(e){}
  setTimeout(loop,900);
}
loop();
window.addEventListener('keydown',e=>{if(e.key==='F5'){e.preventDefault();refresh();}});
</script></body></html>
"""


def port_free(port: int) -> bool:
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def open_browser(url: str):
    # 优先用 chrome / edge 的 --app 模式（无地址栏，看起来就是本机应用程序）
    cands = [
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
    ]
    for c in cands:
        if os.path.isfile(c):
            try:
                subprocess.Popen([c, f"--app={url}", "--window-size=1440,900"], creationflags=CREATE_NO_WINDOW)
                return c
            except Exception:
                continue
    try:
        webbrowser.open(url)
        return "default-browser"
    except Exception:
        return ""


def main():
    port = int(os.environ.get("REMOTE_CLIPVAE_PANEL_PORT") or PANEL_PORT)
    while not port_free(port) and port < PANEL_PORT + 20:
        port += 1
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    url = f"http://127.0.0.1:{port}/"
    add_local(f"控制面板已启动：{url}", "ok")
    print("=" * 62)
    print(" RemoteCLIPVAE 控制面板")
    print(f" 地址：{url}")
    print(" 关掉这个黑窗口 = 关掉控制面板（服务端子进程会被一起停掉）")
    print("=" * 62, flush=True)
    if "--no-browser" not in sys.argv:
        threading.Timer(0.3, open_browser, args=(url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        RUNNER.stop()
        srv.server_close()


if __name__ == "__main__":
    main()
