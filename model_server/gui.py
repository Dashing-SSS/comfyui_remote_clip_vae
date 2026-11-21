# -*- coding: utf-8 -*-
"""
gui.py —— RemoteCLIPVAE 的 服务端图形界面（只用标准库 tkinter，零额外依赖）。

做三件事：
  1. 可视化编辑 config.json（监听地址 / 端口 / 计算设备 / ComfyUI 目录 / python / 模型条目）
  2. 一键启动、停止服务端进程，并把它的 stdout 实时抓到这个窗口里（服务端本身没有黑窗口）
  3. 实时显示显存 / GPU 利用率 / 已加载模型，可多选预加载模型、一键清空显存

启动方式：
    0_图形界面启动.bat                     ← 正常用法（会自动找 ComfyUI 的 python）
    python gui.py                          ← 直接跑也行（python 需要带有 tkinter）
    python gui.py --smoke                 ← 自检：建界面、跑一轮刷新、立刻退出（不弹窗给人看）
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
import traceback
import urllib.error
import urllib.request
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

def _writable_dir(d: str) -> bool:
    try:
        probe = os.path.join(d, ".rclv_write_test")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("1")
        os.remove(probe)
        return True
    except Exception:
        return False


def _base_dir() -> tuple:
    """返回 (数据目录, 服务端源码目录)。

    - 数据目录：放 config.json / gui_state.json / 记录文件，exe 版就是 exe 所在目录
      （exe 放在不可写的位置时退到 %APPDATA%\\RemoteCLIPVAE）。
    - 服务端源码目录：app.py / backend.py / codec.py 所在目录。exe 版这些文件已经打进 exe，
      PyInstaller 运行时会解到临时目录（sys._MEIPASS），直接用那个，所以 exe 可以单独拿走。
    """
    if getattr(sys, "frozen", False):
        root = os.path.dirname(os.path.abspath(sys.executable))
        data_dir = root if _writable_dir(root) else os.path.join(
            os.environ.get("APPDATA") or os.path.expanduser("~"), "RemoteCLIPVAE")
        try:
            os.makedirs(data_dir, exist_ok=True)
        except Exception:
            pass
        meipass = getattr(sys, "_MEIPASS", "")
        for cand in (os.path.join(meipass, "model_server") if meipass else "",
                     os.path.join(root, "model_server")):
            if cand and os.path.isfile(os.path.join(cand, "app.py")):
                return data_dir, cand
        return data_dir, root
    here = os.path.dirname(os.path.abspath(__file__))
    return here, here


HERE, SRV_DIR = _base_dir()
CFG_PATH = os.path.join(HERE, "config.json")
STATE_PATH = os.path.join(HERE, "gui_state.json")
PY_RECORD = os.path.join(HERE, "_python.txt")
COMFY_RECORD = os.path.join(HERE, "_comfy.txt")
PKG_ROOT = os.path.dirname(SRV_DIR)

APP_TITLE = "远程 CLIP / VAE 模型服务端"
SUBTITLE = "双机 ComfyUI 分离加载：客户端只跑 UNet，服务端（本机）跑文本编码与 VAE 编解码"

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
MODEL_EXTS = (".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".gguf", ".bin")
# clip / clip_vision 目录里的视觉编码器不能当文本编码器加载
VISION_HINTS = ("eva02", "eva-clip", "sigclip", "clip_vision", "clipvision", "vision")


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


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


def cpu_name() -> str:
    try:
        return (platform.processor() or platform.machine() or "CPU").strip()
    except Exception:
        return "CPU"


def ram_info() -> tuple[float, float]:
    """返回 (已用 GB, 总量 GB)，Windows 走 ctypes，其他平台给 0。"""
    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            total = st.ullTotalPhys / 2**30
            used = total - st.ullAvailPhys / 2**30
            return round(used, 1), round(total, 1)
        except Exception:
            return 0.0, 0.0
    return 0.0, 0.0


def gpu_stats() -> dict | None:
    """nvidia-smi：显存已用/总量、GPC 利用率。没有 nvidia-smi 就返回 None。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4, creationflags=CREATE_NO_WINDOW,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        line = out.stdout.strip().splitlines()[0]
        used, total, util = [x.strip() for x in line.split(",")[:3]]
        return {"used_mb": float(used), "total_mb": float(total), "util": float(util)}
    except Exception:
        return None


def find_comfy_dirs() -> list[str]:
    """在常见位置找 ComfyUI 源码根（含 comfy/ 与 main.py）。"""
    cands: list[str] = []
    seen = set()

    def add(p: str):
        p = os.path.abspath(p)
        if p in seen or not os.path.isdir(p):
            return
        if os.path.isfile(os.path.join(p, "comfy", "sd.py")) and os.path.isfile(os.path.join(p, "main.py")):
            seen.add(p)
            cands.append(p)

    remembered = _read_text(COMFY_RECORD)
    if remembered:
        add(remembered)
    roots = []
    if os.name == "nt":
        for d in "CDEFGH":
            if os.path.isdir(f"{d}:\\"):
                roots.append(f"{d}:\\")
    else:
        roots = [os.path.expanduser("~")]
    for r in roots:
        for pat in ("ComfyUI", "ComfyUI*", os.path.join("*", "ComfyUI*"),
                    os.path.join("*", "*", "ComfyUI*"), os.path.join("*", "ComfyUI")):
            try:
                for p in glob.glob(os.path.join(r, pat)):
                    add(p)
            except Exception:
                continue
    return cands


def pythons_around(comfy_dir: str) -> list[str]:
    """只根据当前选的 ComfyUI 路径去找它自带的 python（不再扫系统 python）。

    覆盖两种真实布局：
      整合包  <ComfyUI>\\python_embeded\\python.exe
      Desktop <ComfyUI>\\.venv\\Scripts\\python.exe（有的机器 .venv 在上一级）
    """
    if not comfy_dir:
        return []
    base = os.path.abspath(comfy_dir)
    layouts = [
        (r"python_embeded", "python.exe"),
        (r"python_embedded", "python.exe"),
        (r"python", "python.exe"),
        (r".venv", os.path.join("Scripts", "python.exe")),
        (r"venv", os.path.join("Scripts", "python.exe")),
        (r"env", os.path.join("Scripts", "python.exe")),
        (r".venv", "python.exe"),
        (r"venv", "python.exe"),
    ]
    ups = [base,
           os.path.abspath(os.path.join(base, os.pardir)),
           os.path.abspath(os.path.join(base, os.pardir, os.pardir))]
    out: list[str] = []
    seen = set()
    for root in ups:
        for sub, exe in layouts:
            p = os.path.abspath(os.path.join(root, sub, exe))
            if p.lower() in seen or not os.path.isfile(p):
                continue
            seen.add(p.lower())
            out.append(p)
    return out


def python_has_torch(py: str) -> bool:
    """不真的去 import（torch 一 import 要好几秒），先看 site-packages 里有没有 torch。"""
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
    """返回 (python, 说明)。优先「自带 torch」的那个，服务端必须能 import torch。"""
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


# --------------------------------------------------------------------------- #
# 主窗口
# --------------------------------------------------------------------------- #
class Gui:
    def __init__(self, root: tk.Tk, smoke: bool = False):
        self.root = root
        self.smoke = smoke
        self.cfg = load_json(CFG_PATH, {"server": {"host": "0.0.0.0", "port": 8199, "token": ""},
                                       "comfyui_dir": "", "device": "auto",
                                       "max_loaded": {"clip": 1, "vae": 1},
                                       "idle_unload_seconds": 600,
                                       "models": {"clip": {}, "vae": {}}})
        self.state = load_json(STATE_PATH, {})
        self.proc: subprocess.Popen | None = None
        self.log_q: queue.Queue = queue.Queue()
        self.health: dict | None = None
        self.health_busy = False
        self.tick = 0
        self.server_started_at = 0.0
        self.models_view: dict = {"clip": {}, "vae": {}}

        self.var_host = tk.StringVar()
        self.var_port = tk.StringVar()
        self._port = 8199
        self.var_port.trace_add("write", self._sync_port)
        self.var_clip_dir = tk.StringVar()
        self.var_vae_dir = tk.StringVar()
        self.var_llm_dir = tk.StringVar()
        self.var_comfy = tk.StringVar()
        self.var_python = tk.StringVar()
        self.var_prefer_embedded = tk.BooleanVar(value=True)
        self.var_loaded = tk.StringVar(value="已加载: -")
        self.var_server_url = tk.StringVar()
        self.var_statusbar = tk.StringVar(value="未运行")
        self.var_llm_state = tk.StringVar(value="LLM: —")
        self.var_llm_rt = tk.StringVar(value="生成中: —")
        self.var_llm_err = tk.StringVar(value="")
        self.var_nctx = tk.StringVar(value="4096")
        self.var_llm_ext = tk.StringVar()
        self.var_llm_backend = tk.StringVar(value="计算后端: —")
        self.var_llm_bench = tk.StringVar(value="跑分: —")
        self.var_llm_srv = tk.StringVar(value="llama-server: 未启动")
        self.llm_view: dict = {"llm": {}, "llm_vision": {}}
        self.llm_status: dict | None = None
        self.llm_busy = False
        self.llm_install_busy = False
        self.llm_server_busy = False
        self._loading = True          # 读配置期间不要触发「跟着 ComfyUI 路径自动填」
        self._comfy_after: str | None = None
        self._llm_after: str | None = None

        self._build()
        self._load_to_ui()
        self._loading = False
        self.var_comfy.trace_add("write", self._on_comfy_changed)
        self.var_llm_dir.trace_add("write", self._on_llm_dir_changed)
        self._refresh_models()
        self.root.after(200, self._tick)

    # ------------------------------------------------------------------ UI
    def _build(self):
        self.root.title(f"{APP_TITLE} —— N卡 / CPU 双机分离加载")
        self.root.geometry(self.state.get("geometry") or "1440x920")
        self.root.minsize(1180, 720)
        try:
            self.root.iconname(APP_TITLE)
        except Exception:
            pass

        head = ttk.Frame(self.root, padding=(12, 10, 12, 4))
        head.pack(fill="x")
        ttk.Label(head, text=APP_TITLE, font=("Microsoft YaHei UI", 17, "bold")).pack(anchor="w")
        ttk.Label(head, text=SUBTITLE, foreground="#666666").pack(anchor="w", pady=(2, 0))
        row = ttk.Frame(head)
        row.pack(anchor="w", pady=(6, 0))
        ttk.Label(row, textvariable=self.var_server_url, font=("Microsoft YaHei UI", 10, "bold"),
                  foreground="#1a6d1a").pack(side="left")
        self.lbl_head_dev = ttk.Label(row, text="", foreground="#666666")
        self.lbl_head_dev.pack(side="left", padx=(16, 0))

        body = ttk.Frame(self.root, padding=(12, 4, 12, 8))
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=False)
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        # 左边内容比窗口高时要能滚（加了「提示词优化 LLM」以后一行放不下了）
        canvas = tk.Canvas(left, borderwidth=0, highlightthickness=0, width=720)
        vbar = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        pane = ttk.Frame(canvas)
        self._left_canvas = canvas
        self._left_pane = pane
        win = canvas.create_window((0, 0), window=pane, anchor="nw")
        pane.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))

        self._build_config(pane)
        self._build_status(pane)
        self._build_llm(pane)
        self._build_model_lists(pane)
        self._build_log(right)
        self._bind_wheel(pane)

        bar = ttk.Frame(self.root, padding=(12, 2, 12, 6))
        bar.pack(fill="x")
        ttk.Label(bar, textvariable=self.var_statusbar, foreground="#333333").pack(side="left")

    def _bind_wheel(self, pane):
        canvas = self._left_canvas

        def on_wheel(e):
            try:
                canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
            except Exception:
                pass
            return "break"

        def walk(w):
            try:
                w.bind("<MouseWheel>", on_wheel)
            except Exception:
                pass
            for c in w.winfo_children():
                walk(c)

        walk(pane)

    def _build_config(self, parent):
        box = ttk.LabelFrame(parent, text="服务配置", padding=10)
        box.pack(fill="x")
        for c in (0, 1, 2):
            box.columnconfigure(c, weight=1 if c == 1 else 0)

        # 监听地址 / 端口（计算设备不在这里选，见右侧说明）
        ttk.Label(box, text="监听地址").grid(row=0, column=0, sticky="w")
        sub = ttk.Frame(box)
        sub.grid(row=1, column=0, columnspan=3, sticky="we", pady=(0, 6))
        ttk.Entry(sub, textvariable=self.var_host, width=15).pack(side="left")
        ttk.Label(sub, text="端口").pack(side="left", padx=(10, 4))
        ttk.Entry(sub, textvariable=self.var_port, width=7).pack(side="left")
        ttk.Label(sub, text="计算设备：自动（NVIDIA/AMD/Intel 交给 ComfyUI 判断）",
                  foreground="#888888").pack(side="left", padx=(12, 0))

        # 模型根目录（可加多个）：给一个 models 目录就够，里面的 vae / clip / text_encoders / clip_vision / LLM
        # 按文件夹名自动归类，不用再分别填 CLIP / VAE / GGUF 三个目录
        ttk.Label(box, text="模型根目录").grid(row=2, column=0, sticky="nw", pady=(0, 2))
        row2 = ttk.Frame(box)
        row2.grid(row=2, column=1, columnspan=2, sticky="we", padx=(4, 0))
        lbwrap = ttk.Frame(row2)
        lbwrap.pack(side="left", fill="x", expand=True)
        sb = ttk.Scrollbar(lbwrap, orient="vertical")
        self.lst_roots = tk.Listbox(lbwrap, height=3, activestyle="none", exportselection=False,
                                    yscrollcommand=sb.set, font=("Consolas", 9))
        sb.config(command=self.lst_roots.yview)
        sb.pack(side="right", fill="y")
        self.lst_roots.pack(side="left", fill="x", expand=True)
        btns2 = ttk.Frame(row2)
        btns2.pack(side="left", padx=(4, 0))
        ttk.Button(btns2, text="添加目录…", width=11, command=self.add_root).pack(fill="x")
        ttk.Button(btns2, text="移除选中", width=11, command=self.remove_root).pack(fill="x", pady=(4, 0))
        ttk.Button(btns2, text="自动找", width=11, command=self.auto_roots).pack(fill="x", pady=(4, 0))
        ttk.Label(box, text="填 models 目录（如 F:\\Comfy-Desktop\\ComfyUI-Shared\\models）："
                            "里面的 vae / clip / text_encoders / clip_vision / LLM 按文件夹名自动归类，可加多个；"
                            "GGUF 和 mmproj 也自动进「提示词优化」的两个列表",
                  foreground="#888888", wraplength=760).grid(row=3, column=1, columnspan=2, sticky="w", padx=(4, 0))

        ttk.Label(box, text="ComfyUI 路径").grid(row=4, column=0, sticky="w", pady=(0, 2))
        row4 = ttk.Frame(box)
        row4.grid(row=4, column=1, columnspan=2, sticky="we", padx=(4, 0))
        ttk.Entry(row4, textvariable=self.var_comfy).pack(side="left", fill="x", expand=True)
        ttk.Button(row4, text="浏览...", width=8, command=self._pick_comfy).pack(side="left", padx=(4, 4))
        ttk.Button(row4, text="自动找", width=7, command=self._auto_find_comfy).pack(side="left")

        ttk.Label(box, text="本机 ComfyUI 的 Python").grid(row=5, column=0, sticky="w", pady=(0, 2))
        row6 = ttk.Frame(box)
        row6.grid(row=5, column=1, columnspan=2, sticky="we", padx=(4, 0))
        ttk.Entry(row6, textvariable=self.var_python).pack(side="left", fill="x", expand=True)
        ttk.Button(row6, text="浏览...", width=8, command=self._pick_python).pack(side="left", padx=(4, 4))
        ttk.Button(row6, text="自动找", width=7, command=self._auto_find_python).pack(side="left")

        ttk.Label(box, text="按上面填的 ComfyUI 路径自动识别（python_embeded\\python.exe 或 .venv\\Scripts\\python.exe）",
                  foreground="#888888").grid(row=6, column=1, columnspan=2, sticky="w", padx=(4, 0))
        ttk.Checkbutton(box, text="优先用 ComfyUI 自带的 Python 启动（无窗口；Qwen-Image / Flux / Wan 这类模型必须）",
                        variable=self.var_prefer_embedded).grid(row=7, column=0, columnspan=3, sticky="w", pady=(4, 2))

        btns = ttk.Frame(parent)
        btns.pack(fill="x", pady=(8, 0))
        self.btn_start = ttk.Button(btns, text="▶  启动服务", width=12, command=self.start_server)
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(btns, text="■  停止服务", width=12, command=self.stop_server, state="disabled")
        self.btn_stop.pack(side="left", padx=(6, 0))
        ttk.Button(btns, text="清空显存 / 卸载模型", command=self.clear_vram).pack(side="left", padx=(6, 0))
        ttk.Button(btns, text="刷新模型列表", command=self._refresh_models).pack(side="left", padx=(6, 0))
        ttk.Button(btns, text="打开目录", command=self.open_dir).pack(side="left", padx=(6, 0))

    def _build_status(self, parent):
        box = ttk.LabelFrame(parent, text="运行状态", padding=10)
        box.pack(fill="x", pady=(8, 0))
        box.columnconfigure(0, weight=1)

        self.lbl_calc = ttk.Label(box, text="计算方式: —    加载后端: comfy    设备: —")
        self.lbl_calc.grid(row=0, column=0, sticky="w")
        self.lbl_gpu = ttk.Label(box, text="GPU: —    显存: —    利用率: —")
        self.lbl_gpu.grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.lbl_cpu = ttk.Label(box, text="CPU: —    内存: —")
        self.lbl_cpu.grid(row=2, column=0, sticky="w", pady=(4, 0))

        vrow = ttk.Frame(box)
        vrow.grid(row=3, column=0, sticky="we", pady=(8, 0))
        ttk.Label(vrow, text="显存", width=6).pack(side="left")
        self.bar_vram = ttk.Progressbar(vrow, maximum=100)
        self.bar_vram.pack(side="left", fill="x", expand=True)
        self.lbl_vram = ttk.Label(vrow, text="0 / 0 GB    torch 使用 — MB", width=34)
        self.lbl_vram.pack(side="left", padx=(6, 0))

        urow = ttk.Frame(box)
        urow.grid(row=4, column=0, sticky="we", pady=(4, 0))
        ttk.Label(urow, text="GPU 利用率", width=10).pack(side="left")
        self.bar_util = ttk.Progressbar(urow, maximum=100)
        self.bar_util.pack(side="left", fill="x", expand=True)
        self.lbl_util = ttk.Label(urow, text="—", width=12)
        self.lbl_util.pack(side="left", padx=(6, 0))

    def _build_llm(self, parent):
        box = ttk.LabelFrame(parent, text="提示词优化 LLM（GGUF，跑在服务端这台机上）", padding=8)
        box.pack(fill="x", pady=(8, 0))
        box.columnconfigure(0, weight=1)
        box.columnconfigure(1, weight=1)

        wrap = ttk.Frame(box)
        wrap.grid(row=0, column=0, columnspan=2, sticky="we")
        wrap.columnconfigure(0, weight=1)
        wrap.columnconfigure(1, weight=1)

        self.lst_llm = self._llm_listbox(wrap, "GGUF 文本模型", 0)
        self.lst_mmproj = self._llm_listbox(wrap, "视觉投影 mmproj（可选）", 1)

        brow = ttk.Frame(box)
        brow.grid(row=1, column=0, columnspan=2, sticky="we", pady=(4, 0))
        ttk.Button(brow, text="加载 GGUF 到显存", command=self.llm_load).pack(side="left")
        ttk.Button(brow, text="卸载 GGUF", command=self.llm_unload).pack(side="left", padx=(6, 0))
        ttk.Button(brow, text="安装 llama-cpp-python", command=self.llm_install).pack(side="left", padx=(6, 0))
        ttk.Button(brow, text="装本地 wheel…", command=self.llm_install_wheel).pack(side="left", padx=(6, 0))
        ttk.Label(brow, text="上下文").pack(side="left", padx=(10, 4))
        ttk.Entry(brow, textvariable=self.var_nctx, width=7).pack(side="left")

        # AMD 显卡（RX 5700 XT / gfx1010 这类没有 CUDA 的）走这里：编出带 GPU 后端的 llama-cpp-python，
        # 跑在同一个服务进程里、同一个 8199 端口，不再另开 llama 端口
        vrow = ttk.Frame(box)
        vrow.grid(row=2, column=0, columnspan=2, sticky="we", pady=(4, 0))
        ttk.Button(vrow, text="装 Vulkan 版（AMD·免SDK）", command=self.llm_install_vulkan).pack(side="left")
        ttk.Button(vrow, text="装 HIP 版（AMD·免SDK）", command=self.llm_install_hip).pack(side="left", padx=(6, 0))
        ttk.Button(vrow, text="检测计算后端", command=self.llm_check_backend).pack(side="left", padx=(6, 0))
        ttk.Label(vrow, textvariable=self.var_llm_backend, foreground="#1a6d1a").pack(side="left", padx=(10, 0))

        # 第三行：查清「到底是哪版在算」+ 卸干净 + 真跑一次量速度
        drow = ttk.Frame(box)
        drow.grid(row=3, column=0, columnspan=2, sticky="we", pady=(4, 0))
        ttk.Button(drow, text="环境诊断（装了几版 llama）", command=self.llm_env_diag).pack(side="left")
        ttk.Button(drow, text="卸载 llama-cpp-python", command=self.llm_uninstall).pack(side="left", padx=(6, 0))
        ttk.Button(drow, text="GPU 跑分（预填/生成）", command=self.llm_bench).pack(side="left", padx=(6, 0))
        ttk.Label(drow, textvariable=self.var_llm_bench, foreground="#1a6d1a").pack(side="left", padx=(10, 0))

        ttk.Label(box, text="GGUF 走服务端 python，共用 8199 端口；AMD 用预编译 wheel（参考图大小由节点上的「参考图最大像素」控制）",
                  wraplength=700,
                  foreground="#666666").grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))

        ttk.Label(box, textvariable=self.var_llm_state, foreground="#1a6d1a").grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.lbl_llm_rt = ttk.Label(box, textvariable=self.var_llm_rt, foreground="#444444")
        self.lbl_llm_rt.grid(row=6, column=0, columnspan=2, sticky="w", pady=(2, 0))
        self.lbl_llm_err = ttk.Label(box, textvariable=self.var_llm_err, foreground="#b00020", wraplength=980)
        self.lbl_llm_err.grid(row=7, column=0, columnspan=2, sticky="w", pady=(2, 0))

    def _llm_listbox(self, parent, title, col):
        box = ttk.LabelFrame(parent, text=title, padding=4)
        box.grid(row=0, column=col, sticky="nsew", padx=(0, 6) if col == 0 else (6, 0))
        sb = ttk.Scrollbar(box, orient="vertical")
        hs = ttk.Scrollbar(box, orient="horizontal")
        lb = tk.Listbox(box, height=3, activestyle="none", exportselection=False,
                        yscrollcommand=sb.set, xscrollcommand=hs.set, font=("Consolas", 9))
        sb.config(command=lb.yview)
        hs.config(command=lb.xview)
        sb.pack(side="right", fill="y")
        hs.pack(side="bottom", fill="x")
        lb.pack(side="left", fill="both", expand=True)
        lb.bind("<Double-Button-1>", lambda e: self.llm_load())
        return lb

    def _build_model_lists(self, parent):
        wrap = ttk.Frame(parent)
        wrap.pack(fill="both", expand=True, pady=(8, 0))
        wrap.columnconfigure(0, weight=1)
        wrap.columnconfigure(1, weight=1)

        self.lst_clip = self._listbox(wrap, "服务端 CLIP 模型", 0)
        self.lst_vae = self._listbox(wrap, "服务端 VAE 模型", 1)

        brow = ttk.Frame(wrap)
        brow.grid(row=1, column=0, columnspan=2, sticky="we", pady=(4, 0))
        ttk.Button(brow, text="预加载选中的 CLIP", command=lambda: self.preload("clip")).pack(side="left")
        ttk.Button(brow, text="预加载选中的 VAE", command=lambda: self.preload("vae")).pack(side="left", padx=(6, 0))
        ttk.Button(brow, text="删除条目", command=self.delete_entry).pack(side="left", padx=(6, 0))
        ttk.Button(brow, text="扫描目录→加条目", command=self.scan_add).pack(side="left", padx=(6, 0))
        ttk.Label(brow, textvariable=self.var_loaded, foreground="#1a6d1a").pack(side="left", padx=(10, 0))

    def _listbox(self, parent, title, col):
        box = ttk.LabelFrame(parent, text=title, padding=4)
        box.grid(row=0, column=col, sticky="nsew", padx=(0, 6) if col == 0 else (6, 0))
        sb = ttk.Scrollbar(box, orient="vertical")
        lb = tk.Listbox(box, height=5, activestyle="none", exportselection=False,
                        selectmode="extended", yscrollcommand=sb.set, font=("Consolas", 9))
        sb.config(command=lb.yview)
        sb.pack(side="right", fill="y")
        lb.pack(side="left", fill="both", expand=True)
        lb.bind("<Double-Button-1>", lambda e, k=("clip" if col == 0 else "vae"): self.preload(k))
        return lb

    def _build_log(self, parent):
        top = ttk.Frame(parent)
        top.pack(fill="both", expand=True)

        nb = ttk.Notebook(top)
        nb.pack(side="left", fill="both", expand=True)
        self.tab_local = tk.Text(nb, bg="#111111", fg="#e8e8e8", insertbackground="#e8e8e8",
                                 relief="flat", wrap="word", font=("Consolas", 9))
        self.tab_srv = tk.Text(nb, bg="#111111", fg="#e8e8e8", insertbackground="#e8e8e8",
                               relief="flat", wrap="word", font=("Consolas", 9))
        for w in (self.tab_local, self.tab_srv):
            w.tag_configure("ts", foreground="#7a7a7a")
            w.tag_configure("err", foreground="#ff6b6b")
            w.tag_configure("warn", foreground="#ffc857")
            w.tag_configure("ok", foreground="#7bd88f")
            w.tag_configure("info", foreground="#9ad2ff")
        nb.add(self.tab_local, text="本机日志")
        nb.add(self.tab_srv, text="服务端日志（实时）")
        self.nb = nb

        btns = ttk.Frame(top)
        btns.pack(side="right", fill="y", padx=(6, 0))
        ttk.Button(btns, text="清屏", width=8, command=self.clear_logs).pack(anchor="n")
        self.tab_local.insert("end", "界面已就绪。点「启动服务」后服务端日志会出现在右边标签页。\n", "ok")
        self.tab_local.see("end")

    # ------------------------------------------------------------- 读写配置
    def _load_to_ui(self):
        srv = self.cfg.get("server") or {}
        self.var_host.set(srv.get("host", "0.0.0.0"))
        self.var_port.set(str(srv.get("port", 8199)))
        self.var_comfy.set(self.cfg.get("comfyui_dir") or "")
        self.var_python.set(self.state.get("python") or _read_text(PY_RECORD))
        self.var_prefer_embedded.set(bool(self.state.get("prefer_embedded", True)))
        # 模型根目录（可多个）：列表里就是 config.json 的 model_roots
        roots = self.cfg.get("model_roots") or []
        if isinstance(roots, str):
            roots = [roots]
        self.lst_roots.delete(0, "end")
        for r in roots:
            r = clean_path(r)
            if r:
                self.lst_roots.insert("end", r.replace("/", "\\") if os.name == "nt" else r)
        if self.lst_roots.size() == 0:
            # 老配置升级上来：把 ComfyUI 的 models 和它旁边的 *Shared*\models 自动填进去
            comfy = clean_path(self.cfg.get("comfyui_dir") or "")
            cands = []
            if comfy:
                cands.append(os.path.join(comfy, "models"))
                parent = os.path.abspath(comfy).rstrip("\\/")
                for _lv in range(2):
                    parent = os.path.dirname(parent)
                    if not parent or parent == os.path.dirname(parent):
                        break
                    try:
                        for name in sorted(os.listdir(parent)):
                            p = os.path.join(parent, name)
                            if os.path.isdir(p) and os.path.isdir(os.path.join(p, "models")):
                                cands.append(os.path.join(p, "models"))
                    except Exception:
                        pass
            for c in cands:
                if os.path.isdir(c):
                    self._add_root_value(c)
        # 老的三个目录字段不再进界面，值清空（残留的 config 键由 _collect_cfg 删掉）
        self.var_clip_dir.set("")
        self.var_vae_dir.set("")
        self.var_llm_dir.set("")
        try:
            self.var_nctx.set(str(int(self.cfg.get("llm_n_ctx") or 4096)))
        except Exception:
            self.var_nctx.set("4096")
        self.var_llm_ext.set("")   # 「外部 LLM 服务地址」v1.7.27 已下线；残留值必须清掉，否则 GGUF 一个都扫不到
        self._update_url_line()

    def _device_value(self) -> str:
        """界面不再让人选设备：一律交给 ComfyUI/torch 判。config.json 里手写了 cpu 才用 cpu。"""
        dev = str(self.cfg.get("device") or "auto").strip().lower()
        return dev if dev in ("auto", "cuda", "cpu") else "auto"

    def _collect_cfg(self) -> dict:
        try:
            port = int(self.var_port.get().strip() or 8199)
        except ValueError:
            port = 8199
        self.cfg.setdefault("server", {})
        self.cfg["server"]["host"] = self.var_host.get().strip() or "0.0.0.0"
        self.cfg["server"]["port"] = port
        self.cfg["device"] = self._device_value()
        self.cfg["comfyui_dir"] = clean_path(self.var_comfy.get()).replace("\\", "/")
        # 模型根目录（可多个）：vae / clip / text_encoders / clip_vision / LLM 由服务端按文件夹名自动归类
        roots = [r.replace("\\", "/") for r in self._roots_from_ui() if r]
        self.cfg["model_roots"] = roots
        # 老的「CLIP 目录 / VAE 目录 / GGUF 目录」已被 model_roots 取代，残留键删掉
        for key in ("clip_model_dir", "vae_model_dir", "llm_model_dir", "clip_vision_model_dir"):
            self.cfg.pop(key, None)
        try:
            self.cfg["llm_n_ctx"] = int(str(self.var_nctx.get()).strip() or 4096)
        except Exception:
            self.cfg["llm_n_ctx"] = 4096
        # 参考图缩放改由 ComfyUI 节点上的「参考图最大像素(MP)」控制，服务端不再二次缩图（0 = 不缩）
        self.cfg["llm_image_max_side"] = 0
        # llm_external 已下线：一律不写回，顺便把老配置里的残留值清掉
        self.cfg.pop("llm_external", None)
        self.cfg.setdefault("models", {}).setdefault("clip", {})
        self.cfg["models"].setdefault("vae", {})
        return self.cfg

    def _save_cfg(self):
        save_json(CFG_PATH, self._collect_cfg())
        self.state["python"] = self.var_python.get().strip()
        self.state["prefer_embedded"] = bool(self.var_prefer_embedded.get())
        try:
            self.state["geometry"] = self.root.geometry()
        except Exception:
            pass
        save_json(STATE_PATH, self.state)

    def _update_url_line(self):
        try:
            port = int(self.var_port.get().strip() or 8199)
        except ValueError:
            port = 8199
        ips = lan_ips()
        ip = ips[0] if ips else "127.0.0.1"
        self.var_server_url.set(f"ComfyUI 插件里的「服务端地址」填： http://{ip}:{port}   （本机 IP: {ip}）")

    # ------------------------------------------------------------- 选路径
    def _on_comfy_changed(self, *_a):
        """ComfyUI 路径一改，就按这个目录重新推 python（防抖 400ms）。"""
        if self._loading:
            return
        if self._comfy_after:
            try:
                self.root.after_cancel(self._comfy_after)
            except Exception:
                pass
        self._comfy_after = self.root.after(400, self._apply_comfy_dir)

    def _apply_comfy_dir(self, scan: bool = True):
        self._comfy_after = None
        comfy = clean_path(self.var_comfy.get())
        if not comfy or not os.path.isdir(comfy):
            return
        # 只按 ComfyUI 目录推 python；CLIP / VAE 模型目录不跟着变，永远只认填写的那两个目录
        self._auto_find_python(quiet=True)
        # 扫的是填写的那两个目录（空的就什么都不扫）
        if scan:
            self.scan_add(quiet=True)

    # ---- 模型根目录（可多个） ---- #
    def _roots_from_ui(self) -> list[str]:
        try:
            return [clean_path(self.lst_roots.get(i)) for i in range(self.lst_roots.size())]
        except Exception:
            return []

    def _add_root_value(self, p: str):
        p = clean_path(p)
        if not p:
            return
        cur = {os.path.normcase(clean_path(x)) for x in self._roots_from_ui()}
        if os.path.normcase(p) in cur:
            return
        self.lst_roots.insert("end", p.replace("/", "\\") if os.name == "nt" else p)

    def add_root(self):
        p = filedialog.askdirectory(initialdir=(self._roots_from_ui() or [HERE])[0],
                                    title="选 models 目录（里面有 vae / clip / text_encoders / clip_vision / LLM）")
        if not p:
            return
        self._add_root_value(p)
        self._after_roots_changed()

    def remove_root(self):
        sel = list(self.lst_roots.curselection())
        if not sel:
            messagebox.showinfo("没选中", "先在列表里选中一行")
            return
        for i in reversed(sel):
            self.lst_roots.delete(i)
        self._after_roots_changed()

    def auto_roots(self):
        """自动加：ComfyUI 自己的 models + 它旁边带 models 的 *Shared* 目录。"""
        cands = []
        comfy = clean_path(self.var_comfy.get())
        if comfy:
            cands.append(os.path.join(comfy, "models"))
            parent = os.path.abspath(comfy).rstrip("\\/")
            for _lv in range(2):
                parent = os.path.dirname(parent)
                if not parent or parent == os.path.dirname(parent):
                    break
                try:
                    for name in sorted(os.listdir(parent)):
                        p = os.path.join(parent, name)
                        if os.path.isdir(p) and os.path.isdir(os.path.join(p, "models")):
                            cands.append(os.path.join(p, "models"))
                except Exception:
                    pass
        for d in find_comfy_dirs():
            cands.append(os.path.join(d, "models"))
        added = 0
        for c in cands:
            if os.path.isdir(c) and os.path.normcase(c) not in {os.path.normcase(x) for x in self._roots_from_ui()}:
                self._add_root_value(c)
                added += 1
        if added:
            self.log(f"自动加入 {added} 个 models 根目录", "ok")
            self._after_roots_changed()
        else:
            self.log("没自动找到新的 models 目录，手动「添加目录…」选一下", "warn")

    def _after_roots_changed(self):
        self._save_cfg()
        if self.is_running():
            self._reload_server()
        self._refresh_models()

    def _pick_dir(self, var: tk.StringVar, kind: str = ""):
        p = filedialog.askdirectory(initialdir=var.get() or HERE)
        if p:
            var.set(p.replace("/", "\\") if os.name == "nt" else p)
            # 选完目录就立刻扫，不用再手点「扫描目录→加条目」
            self.scan_add(quiet=True)

    def _on_llm_dir_changed(self, *_a):
        """GGUF 目录手动改完，600ms 后重新扫一遍（防抖）。"""
        if self._loading:
            return
        if getattr(self, "_llm_after", None):
            try:
                self.root.after_cancel(self._llm_after)
            except Exception:
                pass
        self._llm_after = self.root.after(600, self._apply_llm_dir)

    def _apply_llm_dir(self):
        self._llm_after = None
        self._save_cfg()
        if self.is_running():
            self._reload_server()
        self._refresh_llm_lists()

    def _pick_llm_dir(self):
        p = filedialog.askdirectory(initialdir=clean_path(self.var_llm_dir.get()) or HERE,
                                    title="选放 .gguf 的目录（一般是 <ComfyUI>/models/LLM）")
        if p:
            self.var_llm_dir.set(p.replace("/", "\\") if os.name == "nt" else p)
            self._refresh_llm_lists()

    def _auto_find_llm_dir(self):
        comfy = clean_path(self.var_comfy.get())
        cand_dirs = []
        if comfy:
            for d in ("LLM", "llm", "llm_gguf", "LLM_gguf", "gguf", "text_encoders_llm"):
                cand_dirs.append(os.path.join(comfy, "models", d))
        for d in (find_comfy_dirs() if not comfy else []):
            for sub in ("LLM", "llm", "gguf"):
                cand_dirs.append(os.path.join(d, "models", sub))
        for d in cand_dirs:
            if os.path.isdir(d) and any(f.lower().endswith(".gguf") for f in os.listdir(d)):
                self.var_llm_dir.set(d)
                self._refresh_llm_lists()
                self.log(f"自动找到 GGUF 目录：{d}", "ok")
                return
        self.log("没自动找到带 .gguf 的目录，手动「浏览...」选一下（一般是 <ComfyUI>/models/LLM）", "warn")

    def _pick_comfy(self):
        p = filedialog.askdirectory(initialdir=self.var_comfy.get() or HERE,
                                    title="选 ComfyUI 源码根目录（里面有 comfy 文件夹和 main.py）")
        if p:
            self._loading = True
            self.var_comfy.set(p)
            self._loading = False
            # 只更新 python；CLIP / VAE 目录保持用户填的那两个，不跟着 ComfyUI 变
            self._apply_comfy_dir(scan=False)
            self._refresh_models()

    def _pick_python(self):
        p = filedialog.askopenfilename(initialdir=os.path.dirname(self.var_python.get() or HERE),
                                       filetypes=[("python.exe", "python*.exe"), ("所有文件", "*.*")],
                                       title="选 ComfyUI 自带的 python.exe")
        if p:
            self.var_python.set(p)

    def _auto_find_comfy(self):
        found = find_comfy_dirs()
        if not found:
            messagebox.showwarning("没找到", "没在常见位置找到 ComfyUI，请用「浏览...」手动选。")
            return
        self._loading = True
        self.var_comfy.set(found[0])
        self._loading = False
        self.log(f"自动找到 ComfyUI：{found[0]}", "ok")
        self._apply_comfy_dir(scan=True)
        self._refresh_models()

    def _auto_find_python(self, quiet: bool = False):
        comfy = clean_path(self.var_comfy.get())
        if not comfy:
            if not quiet:
                messagebox.showinfo("先选 ComfyUI", "先选好「ComfyUI 路径」，python 会按这个目录自动识别。")
            return ""
        py, why = best_python(comfy)
        if py and why == "ok":
            self.var_python.set(py)
            if not quiet:
                self.log(f"按 ComfyUI 路径识别到 python（自带 torch）：{py}", "ok")
            return py
        if py:
            self.var_python.set(py)
            self.log(f"按 ComfyUI 路径识别到 python：{py}", "ok")
            if not quiet:
                self.log("注意：这个 python 里没看到 torch，start 时会再探一次；"
                         "若报 No module named 'torch'，请改用 ComfyUI 自己的 python_embeded / .venv。", "warn")
            return py
        if not quiet:
            self.log(f"在 {comfy} 里没找到 python_embeded\\python.exe 或 .venv\\Scripts\\python.exe，"
                     "请手动「浏览...」指定 ComfyUI 用的 python。", "warn")
        return ""

    def _resolve_python(self) -> str:
        p = clean_path(self.var_python.get())
        if p and os.path.isfile(p):
            return p
        if self.var_prefer_embedded.get():
            found = pythons_around(clean_path(self.var_comfy.get()))
            if found:
                self.var_python.set(found[0])
                return found[0]
        if p:
            return p
        return sys.executable

    # ------------------------------------------------------------- 日志
    def log(self, text: str, tag: str = "info", widget: str = "local"):
        w = self.tab_local if widget == "local" else self.tab_srv
        ts = time.strftime("%H:%M:%S")
        w.insert("end", f"[{ts}] ", "ts")
        w.insert("end", text.rstrip() + "\n", tag)
        w.see("end")
        try:
            w.see("end")
        except Exception:
            pass

    def clear_logs(self):
        self.tab_local.delete("1.0", "end")
        self.tab_srv.delete("1.0", "end")

    # ------------------------------------------------------------- 启停服务
    def start_server(self, mock: bool = False):
        if self.proc and self.proc.poll() is None:
            self.log("服务已经在运行了", "warn")
            return
        app_py = os.path.join(SRV_DIR, "app.py")
        if not os.path.isfile(app_py):
            messagebox.showerror("缺少服务端文件",
                                 f"找不到 {app_py}\n\n"
                                 "这个 exe 应该自带 model_server（内置在 exe 里）。如果你是从源码目录跑的 gui.py，"
                                 "请确认 model_server 目录完整。")
            return
        py = self._resolve_python()
        if not py or not os.path.isfile(py):
            messagebox.showerror("没有 python", "找不到可用的 python，请先设置「本机 ComfyUI 的 Python」。")
            return
        comfy = clean_path(self.var_comfy.get())
        if comfy and not os.path.isdir(os.path.join(comfy, "comfy")):
            messagebox.showerror("路径不对", f"{comfy} 里没有 comfy 文件夹，请选 ComfyUI 源码根目录。")
            return

        ok, why = probe_python(py)
        if not ok:
            self.log(f"这个 python 不能用来跑服务端：{py}（{why}）", "err")
            messagebox.showerror(
                "这个 python 跑不了服务端",
                f"{py}\n\n{why}\n\n服务端必须用 ComfyUI 自带的 python"
                "（python_embeded\\python.exe，里面有 torch / numpy / comfy）。\n"
                "点「自动找」或手动填一个。")
            return

        self._save_cfg()
        _write_text(PY_RECORD, py)
        if comfy:
            _write_text(COMFY_RECORD, os.path.abspath(comfy))

        args = [py, app_py, "--config", CFG_PATH]
        dev = self._device_value()
        if dev != "auto":
            args += ["--device", dev]
        if mock:
            args.append("--mock")

        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self.proc = subprocess.Popen(
                args, cwd=SRV_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                env=env, creationflags=CREATE_NO_WINDOW,
            )
        except Exception as e:
            self.log(f"启动失败：{e}", "err")
            messagebox.showerror("启动失败", str(e))
            return

        self.server_started_at = time.time()
        self.log(f"已启动服务端子进程 PID {self.proc.pid}", "ok", "srv")
        self.log(f"命令：{py} app.py --config config.json" + (" --device " + dev if dev != "auto" else "")
                 + (" --mock" if mock else ""), "info")
        threading.Thread(target=self._read_proc, args=(self.proc,), daemon=True).start()
        self.btn_start.state(["disabled"])
        self.btn_stop.state(["!disabled"])
        self.var_statusbar.set(f"启动中…… PID {self.proc.pid}")

    def _read_proc(self, proc: subprocess.Popen):
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                if line:
                    self.log_q.put(line.rstrip("\n"))
        except Exception:
            pass
        finally:
            self.log_q.put(("__EXIT__", None))

    def stop_server(self):
        if not self.proc or self.proc.poll() is not None:
            self.log("服务没在运行", "warn")
            self._after_exit()
            return
        pid = self.proc.pid
        self.log(f"正在停止 PID {pid} ……", "warn", "srv")
        try:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=6)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=4)
        except Exception as e:
            self.log(f"停止时出错：{e}", "err")
        self._after_exit()

    def _after_exit(self):
        self.proc = None
        self.health = None
        self.btn_start.state(["!disabled"])
        self.btn_stop.state(["disabled"])
        self.var_statusbar.set("未运行")
        self.var_loaded.set("已加载: -")
        self.lbl_calc.config(text="计算方式: —    加载后端: comfy    设备: —")
        self.lbl_gpu.config(text="GPU: —    显存: —    利用率: —")
        self.bar_vram["value"] = 0
        self.bar_util["value"] = 0
        self.lbl_vram.config(text="0 / 0 GB    torch 使用 — MB")
        self.lbl_util.config(text="—")
        self.var_llm_state.set("LLM: —")
        self.var_llm_rt.set("生成中: —")
        self.var_llm_err.set("")
        self.llm_status = None

    def is_running(self) -> bool:
        return bool(self.proc and self.proc.poll() is None)

    # ------------------------------------------------------------- HTTP
    def _sync_port(self, *_a):
        try:
            self._port = int(self.var_port.get().strip() or 8199)
        except Exception:
            self._port = 8199

    def _base_url(self) -> str:
        # 只读 self._port：StringVar 只能在主线程访问，工作线程拿它会抛
        # RuntimeError: main thread is not in main loop
        return f"http://127.0.0.1:{self._port}"

    def _http_get(self, path: str, timeout: float = 4) -> dict | None:
        token = str((self.cfg.get("server") or {}).get("token") or "")
        req = urllib.request.Request(self._base_url() + path)
        if token:
            req.add_header("X-Auth-Token", token)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            return None

    def _http_rpc(self, op: str, params: dict | None = None, timeout: float = 30) -> dict | None:
        token = str((self.cfg.get("server") or {}).get("token") or "")
        body = json.dumps({"op": op, "params": params or {}}).encode("utf-8")
        req = urllib.request.Request(self._base_url() + "/rpc", data=body,
                                     headers={"Content-Type": "application/json"})
        if token:
            req.add_header("X-Auth-Token", token)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                return json.loads(e.read().decode("utf-8"))
            except Exception:
                return {"ok": False, "error": f"HTTP {e.code}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _poll_health(self):
        if self.health_busy:
            return
        self.health_busy = True

        def work():
            try:
                # 轻量 health：不扫模型清单（那一步在共享模型盘上要好几秒，会把状态栏拖死）
                self.log_q.put(("__HEALTH__", self._http_get("/health?light=1", timeout=8)))
            finally:
                self.health_busy = False

        threading.Thread(target=work, daemon=True).start()

    def apply_health(self, resp: dict | None):
        if not resp or not resp.get("ok"):
            self.health = None
            if self.is_running():
                self.var_statusbar.set(f"服务进程在，但 /health 还没响应…… PID {self.proc.pid}")
            return
        d = resp.get("data") or {}
        self.health = d
        dev = d.get("device") or ("cuda" if d.get("cuda_available") else "cpu")
        name = d.get("device_name") or "—"
        self.lbl_calc.config(text=f"计算方式: {dev}    加载后端: comfy    设备: {name}")
        self.lbl_head_dev.config(text=f"设备: {name}{'（MOCK）' if d.get('mock') else ''}")

        vram = d.get("vram") or {}
        total = float(vram.get("total_gb") or 0)
        free = float(vram.get("free_gb") or 0)
        used = max(0.0, total - free)
        self.gpu_n = getattr(self, "gpu_n", 0) + 1
        g = (gpu_stats() or {}) if self.gpu_n % 3 == 1 else (self.last_gpu or {})
        self.last_gpu = g
        util = g.get("util")
        self.lbl_gpu.config(
            text=f"GPU: {name} [cuda]    显存: {used:.1f} / {total:.1f} GB"
                 f"    利用率: {('%d%%' % util) if util is not None else '—'}")
        alloc = d.get("torch_allocated_mb")
        self.lbl_vram.config(text=f"{used:.1f} / {total:.1f} GB    torch 使用 {('%.1f MB' % alloc) if alloc is not None else '—'}")
        self.bar_vram["value"] = (used / total * 100) if total else 0
        if util is not None:
            self.bar_util["value"] = util
            self.lbl_util.config(text=f"{util:.0f}%")
        else:
            self.bar_util["value"] = 0
            self.lbl_util.config(text="—")

        ru, rt = ram_info()
        self.lbl_cpu.config(text=f"CPU: {cpu_name()}    内存: {ru:.1f} / {rt:.1f} GB" if rt else f"CPU: {cpu_name()}")

        loaded = d.get("loaded") or {}
        parts = []
        for k in ("clip", "vae"):
            if loaded.get(k):
                parts.append(f"{k}: {', '.join(loaded[k])}")
        self.var_loaded.set("已加载: " + (" | ".join(parts) if parts else "-"))
        self.apply_llm(d.get("llm"))
        up = d.get("uptime_s")
        self.var_statusbar.set(f"运行中（PID {self.proc.pid if self.proc else '?'}）· 端口 {d.get('port', self.var_port.get())}"
                               f" · 已运行 {up:.0f}s" if isinstance(up, (int, float)) else "运行中")

    # ------------------------------------------------------------- 模型列表
    def _model_bases(self, kind: str) -> list[str]:
        """相对路径可能挂在哪几个根下（按优先级）：先模型根目录，再 ComfyUI 自己的 models。"""
        bases = [clean_path(r) for r in self._roots_from_ui() if clean_path(r)]
        comfy = clean_path(self.var_comfy.get())
        if comfy:
            bases.append(os.path.join(comfy, "models"))
            bases.append(comfy)
        bases.append(HERE)
        return bases

    def _resolve_model_file(self, kind: str, p: str) -> str:
        p = str(p).replace("/", os.sep)
        if os.path.isabs(p):
            return p
        for b in self._model_bases(kind):
            cand = os.path.join(b, p)
            if os.path.exists(cand):
                return cand
        bases = self._model_bases(kind)
        return os.path.join(bases[0], p) if bases else p

    def _local_models_view(self) -> dict:
        """直接按 config.json + 界面上两个模型目录算，不依赖服务端是否在跑。"""
        models = self.cfg.get("models") or {}
        out = {}
        for k in ("clip", "vae"):
            out[k] = {}
            for name, entry in (models.get(k) or {}).items():
                if isinstance(entry, dict):
                    files = entry.get("files") or ([entry["file"]] if entry.get("file") else [])
                    note = entry.get("note", "")
                elif isinstance(entry, str):
                    files = [entry]
                    note = ""
                else:
                    files, note = [], "配置格式错误"
                files = [str(x) for x in files]
                resolved = [self._resolve_model_file(k, f) for f in files]
                exists = bool(resolved) and all(os.path.exists(r) for r in resolved)
                out[k][name] = {"files": files, "resolved": resolved, "exists": exists, "note": note}
        return out

    def _refresh_models(self):
        self._collect_cfg()
        view = self._local_models_view()
        # 服务在跑时用服务端的判定（它和 客户端加载用的是同一套）、但以本地条目为准，
        # 这样「刚扫描出来的条目」不会因为服务端还是旧快照而显示 0 条。
        if self.is_running():
            r = self._http_get("/models")
            if r and r.get("ok") and isinstance(r.get("data"), dict):
                srv = r["data"]
                for k in ("clip", "vae"):
                    for name, info in (srv.get(k) or {}).items():
                        mine = view[k].get(name)
                        if mine is not None:
                            if isinstance(info, dict) and "exists" in info:
                                mine["exists"] = info["exists"]
                                mine["resolved"] = info.get("resolved") or mine["resolved"]
                        elif isinstance(info, dict):
                            view[k][name] = info
        self.models_view = view
        for k, lb in (("clip", self.lst_clip), ("vae", self.lst_vae)):
            lb.delete(0, "end")
            for name, info in (self.models_view.get(k) or {}).items():
                exists = info.get("exists", None)
                mark = "" if exists is not False else "   ✗ 文件缺失"
                lb.insert("end", f"{name}{mark}")
                if exists is False:
                    lb.itemconfig(lb.size() - 1, foreground="#b00020")
            if lb.size() == 0:
                lb.insert("end", "（还没有条目：点「扫描目录→加条目」或运行 1_）")
                lb.itemconfig(0, foreground="#888888")
        self._refresh_llm_lists()
        self.log(f"模型列表已刷新：CLIP {len((self.models_view.get('clip') or {}))} 条，"
                 f"VAE {len((self.models_view.get('vae') or {}))} 条，"
                 f"GGUF {len((self.llm_view.get('llm') or {}))} + mmproj {len((self.llm_view.get('llm_vision') or {}))}",
                 "info")

    # ------------------------------------------------------------- 提示词优化 LLM
    def _llm_dirs(self) -> list[str]:
        out = []
        roots = [clean_path(r) for r in self._roots_from_ui() if clean_path(r)]
        comfy = clean_path(self.var_comfy.get())
        if comfy:
            # ComfyUI 自己的 models + 它旁边的 *Shared*\models（CLIP/VAE 就是从共享目录扫到的，
            # GGUF 一般也放在同一个共享盘的 models\LLM 里）
            roots.append(os.path.join(comfy, "models"))
            parent = comfy.rstrip("\\/")
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
        seen = {os.path.normcase(x) for x in out}
        for root in roots:
            for sub in ("LLM", "llm", "llm_gguf", "LLM_gguf", "gguf", "text_encoders_llm"):
                p = os.path.join(root, sub)
                if os.path.isdir(p) and os.path.normcase(p) not in seen:
                    seen.add(os.path.normcase(p))
                    out.append(p)
        # 兜底：根目录里任何层级下有 .gguf 的文件夹（用户目录结构可能不叫 LLM）
        for root in roots:
            if not os.path.isdir(root):
                continue
            for dirpath, dirs, files in os.walk(root):
                if dirpath[len(root):].count(os.sep) > 2:
                    dirs[:] = []
                    continue
                if not any(f.lower().endswith(".gguf") for f in files):
                    continue
                np = os.path.normcase(dirpath)
                if np not in seen:
                    seen.add(np)
                    out.append(dirpath)
        return out

    def _local_llm_view(self) -> dict:
        view = {"llm": {}, "llm_vision": {}}
        for d in self._llm_dirs():
            if not os.path.isdir(d):
                continue
            try:
                names = sorted(os.listdir(d))
            except Exception:
                continue
            for fn in names:
                if not fn.lower().endswith(".gguf"):
                    continue
                low = fn.lower()
                kind = "llm_vision" if any(h in low for h in ("mmproj", "vision", "vit")) else "llm"
                p = os.path.join(d, fn)
                try:
                    size_gb = round(os.path.getsize(p) / (1024 ** 3), 2)
                except Exception:
                    size_gb = 0
                view[kind].setdefault(fn, {"files": [fn], "path": p, "dir": d,
                                            "exists": True, "size_gb": size_gb})
        return view

    def _refresh_llm_lists(self):
        view = self._local_llm_view()
        if self.is_running():
            r = self._http_get("/models")
            if r and r.get("ok") and isinstance(r.get("data"), dict):
                for k in ("llm", "llm_vision"):
                    for name, info in ((r["data"].get(k) or {}).items()):
                        view[k][name] = info if isinstance(info, dict) else {"exists": True}
        self.llm_view = view
        for k, lb in (("llm", self.lst_llm), ("llm_vision", self.lst_mmproj)):
            lb.delete(0, "end")
            for name, info in (view.get(k) or {}).items():
                gb = info.get("size_gb")
                txt = name if not gb else f"{name}   ({gb} GB)"
                lb.insert("end", txt)
                if info.get("exists") is False:
                    lb.itemconfig(lb.size() - 1, foreground="#b00020")
            if lb.size() == 0:
                hint = ("（目录里没有 .gguf：先在上面填 GGUF 模型目录并放模型进去）" if k == "llm"
                        else "（可选：没有 mmproj 就只能做纯文本优化）")
                lb.insert("end", hint)
                lb.itemconfig(0, foreground="#888888")

    def _llm_selected(self, kind: str) -> str:
        lb = self.lst_llm if kind == "llm" else self.lst_mmproj
        src = list((self.llm_view.get(kind) or {}).keys())
        for idx in lb.curselection():
            if 0 <= idx < len(src):
                return src[idx]
        return ""

    def llm_load(self):
        if not self.is_running():
            messagebox.showinfo("服务没开", "先点「启动服务」，再加载 GGUF。")
            return
        model = self._llm_selected("llm")
        if not model:
            messagebox.showinfo("没选中", "先在「GGUF 文本模型」里选一个（.gguf）。")
            return
        mmproj = self._llm_selected("llm_vision")
        try:
            nctx = int(str(self.var_nctx.get()).strip() or 4096)
        except Exception:
            nctx = 4096
        self._save_cfg()
        self.var_llm_state.set("LLM: 加载中…（首次读 6GB 权重，等十几秒）")
        self.log(f"请求加载 GGUF：{model}" + (f" + mmproj {mmproj}" if mmproj else "（纯文本）")
                 + f"，上下文 {nctx}", "info")

        def work():
            r = self._http_rpc("llm_load", {"model": model, "mmproj": mmproj, "n_ctx": nctx},
                               timeout=1800)
            self.log_q.put(("__LLM_OK__" if r and r.get("ok") else "__LLM_ERR__", r or {}))
            self.log_q.put(("__HEALTH_Q__", None))

        threading.Thread(target=work, daemon=True).start()

    def llm_unload(self):
        if not self.is_running():
            messagebox.showinfo("服务没开", "服务没开。")
            return

        def work():
            r = self._http_rpc("llm_unload", {}, timeout=120)
            self.log_q.put(("__LLM_OK__" if r and r.get("ok") else "__LLM_ERR__", r or {}))
            self.log_q.put(("__HEALTH_Q__", None))

        threading.Thread(target=work, daemon=True).start()

    def llm_install(self):
        if self.llm_install_busy:
            messagebox.showinfo("正在装", "上一次安装还在跑，看右边「服务端日志」。")
            return
        py = self._resolve_python()
        if not py or not os.path.isfile(py):
            messagebox.showerror("没有 python", "先设置「本机 ComfyUI 的 Python」，llama-cpp-python 必须装在它里面。")
            return
        if not messagebox.askyesno("确认安装",
                                   f"要给这个 python 装 llama-cpp-python：\n{py}\n\n"
                                   "会先按它自己的 Python 版本 + torch 的 CUDA 版本去下**预编译 wheel**，"
                                   "不走源码编译（Windows 上编译基本必败）。会联网，可能几分钟。继续？"):
            return
        self._run_install(py, "")

    def llm_install_wheel(self):
        if self.llm_install_busy:
            messagebox.showinfo("正在装", "上一次安装还在跑，看右边「服务端日志」。")
            return
        py = self._resolve_python()
        if not py or not os.path.isfile(py):
            messagebox.showerror("没有 python", "先设置「本机 ComfyUI 的 Python」。")
            return
        whl = filedialog.askopenfilename(title="选 llama_cpp_python 的 .whl 文件",
                                         filetypes=[("wheel", "*.whl"), ("所有文件", "*.*")])
        if not whl:
            return
        self._run_install(py, whl)

    def _run_install(self, py: str, whl: str):
        self.llm_install_busy = True
        self.var_llm_state.set("LLM: 安装依赖中…（日志在右边「服务端日志」）")
        self.log(f"开始安装 llama-cpp-python → {py}" + (f"（用本地 wheel {os.path.basename(whl)}）" if whl else ""),
                 "warn", "srv")

        def run():
            res = {}
            try:
                if SRV_DIR not in sys.path:
                    sys.path.insert(0, SRV_DIR)
                import llm as llm_mod
                res = llm_mod.install_llama(py, log=lambda s: self.log_q.put(("__LLM_LINE__", s)),
                                            wheel_url=whl)
            except Exception as e:
                self.log_q.put(("__LLM_LINE__", f"安装流程异常：{e}"))
                res = {"ok": False, "error": str(e)}
            self.log_q.put(("__LLM_INSTALL_RESULT__", res))

        threading.Thread(target=run, daemon=True).start()

    # ---- AMD 显卡：Vulkan 路线 ---- #
    def llm_install_vulkan(self):
        if self.llm_install_busy:
            messagebox.showinfo("正在装", "上一次安装还在跑，看右边「服务端日志」。")
            return
        py = self._resolve_python()
        if not py or not os.path.isfile(py):
            messagebox.showerror("没有 python", "先设置「本机 ComfyUI 的 Python」。")
            return
        if not messagebox.askyesno(
                "装 Vulkan 版 llama-cpp-python",
                f"给这个 python 装/重装成 Vulkan 后端：\n{py}\n\n"
                "先自动下官方**预编译** Vulkan wheel（约 42MB，自带 ggml-vulkan.dll）——\n"
                "不用装 Vulkan SDK、不用 MSVC、不用 CMake，AMD 卡（含 RX 5700 XT / gfx1010）直接吃显卡。\n"
                "GGUF 还是跑在服务端同一个进程、同一个 8199 端口里。"
                "只有下载失败且本机确实装了 Vulkan SDK 才会退去现场编译。继续？"):
            return
        self.llm_install_busy = True
        self.var_llm_state.set("LLM: 正在装预编译 Vulkan wheel…（日志在右边「服务端日志」）")
        self.log(f"开始安装 Vulkan 版 llama-cpp-python → {py}", "warn", "srv")

        def run():
            res = {}
            try:
                if SRV_DIR not in sys.path:
                    sys.path.insert(0, SRV_DIR)
                import llm as llm_mod
                res = llm_mod.install_llama_vulkan(py, log=lambda s: self.log_q.put(("__LLM_LINE__", s)))
            except Exception as e:
                self.log_q.put(("__LLM_LINE__", f"Vulkan 安装异常：{e}"))
                res = {"ok": False, "error": str(e)}
            self.log_q.put(("__LLM_INSTALL_RESULT__", res))

        threading.Thread(target=run, daemon=True).start()

    # ---- llama-server（官方 Vulkan 版二进制，不用编 llama-cpp-python） ---- #
    def _srv_exe(self) -> str:
        d = os.path.join(HERE, "llama_server")
        try:
            if SRV_DIR not in sys.path:
                sys.path.insert(0, SRV_DIR)
            import llm as llm_mod
            return llm_mod.find_llama_server(d)
        except Exception:
            return ""

    def _llm_model_path(self, kind: str) -> str:
        name = self._llm_selected(kind)
        if not name:
            return ""
        info = (self.llm_view.get(kind) or {}).get(name) or {}
        p = info.get("path") or ""
        if p and os.path.exists(p):
            return p
        for d in self._llm_dirs():
            cand = os.path.join(d, name)
            if os.path.exists(cand):
                return cand
        return ""

    def llm_server_download(self):
        if self.llm_server_busy:
            messagebox.showinfo("正在下", "上一次下载还在跑，看右边「服务端日志」。")
            return
        if not messagebox.askyesno(
                "下载 Vulkan 版 llama-server",
                "从 GitHub（ggml-org/llama.cpp）下最新 release 里的 Windows Vulkan 包，"
                "解压到：\n" + os.path.join(HERE, "llama_server") + "\n\n"
                "约 40~80MB，需要这台机能上网。下完点「启动 llama-server」就能用 AMD 显卡算。继续？"):
            return
        self.llm_server_busy = True
        self.var_llm_srv.set("llama-server: 下载中…")
        self.log("开始下载 Vulkan 版 llama-server", "warn", "srv")

        def run():
            res = {}
            try:
                if SRV_DIR not in sys.path:
                    sys.path.insert(0, SRV_DIR)
                import llm as llm_mod
                res = llm_mod.download_llama_server_vulkan(
                    os.path.join(HERE, "llama_server"),
                    log=lambda s: self.log_q.put(("__LLM_LINE__", s)))
            except Exception as e:
                self.log_q.put(("__LLM_LINE__", f"下载异常：{e}"))
                res = {"ok": False, "error": str(e)}
            self.log_q.put(("__LLM_SRV_RESULT__", {"action": "download", **res}))

        threading.Thread(target=run, daemon=True).start()

    def llm_server_from_zip(self):
        if self.llm_server_busy:
            messagebox.showinfo("忙", "上一个 llama-server 操作还没结束。")
            return
        z = filedialog.askopenfilename(title="选 llama-*-bin-win-vulkan-x64.zip",
                                       filetypes=[("zip", "*.zip"), ("所有文件", "*.*")])
        if not z:
            return
        self.llm_server_busy = True
        self.var_llm_srv.set("llama-server: 解压中…")

        def run():
            res = {}
            try:
                if SRV_DIR not in sys.path:
                    sys.path.insert(0, SRV_DIR)
                import llm as llm_mod
                res = llm_mod.import_llama_server_zip(
                    z, os.path.join(HERE, "llama_server"),
                    log=lambda s: self.log_q.put(("__LLM_LINE__", s)))
            except Exception as e:
                res = {"ok": False, "error": str(e)}
            self.log_q.put(("__LLM_SRV_RESULT__", {"action": "download", **res}))

        threading.Thread(target=run, daemon=True).start()

    def llm_server_start(self):
        if self.llm_server_busy:
            messagebox.showinfo("忙", "上一个 llama-server 操作还没结束。")
            return
        exe = self._srv_exe()
        if not exe:
            messagebox.showinfo("还没下载",
                                "先点「下载 Vulkan 版 llama-server」（或自己把 llama.cpp 的 "
                                "Vulkan 包解压到 " + os.path.join(HERE, "llama_server") + "）。")
            return
        model = self._llm_model_path("llm")
        if not model:
            messagebox.showinfo("没选中模型", "先在「GGUF 文本模型」里选一个要跑的 .gguf。")
            return
        mmproj = self._llm_model_path("llm_vision")
        try:
            nctx = int(str(self.var_nctx.get()).strip() or 4096)
        except Exception:
            nctx = 4096
        self.llm_server_busy = True
        self.var_llm_srv.set("llama-server: 启动中…（首次加载模型要等一会）")
        self.log(f"启动 llama-server：{os.path.basename(model)}"
                 + (f" + {os.path.basename(mmproj)}" if mmproj else ""), "warn", "srv")

        def run():
            res = {}
            try:
                if SRV_DIR not in sys.path:
                    sys.path.insert(0, SRV_DIR)
                import llm as llm_mod
                res = llm_mod.start_llama_server(exe, model, mmproj,
                                                 port=llm_mod.LLAMA_SERVER_PORT,
                                                 ngl=99, n_ctx=nctx,
                                                 log=lambda s: self.log_q.put(("__LLM_LINE__", s)))
            except Exception as e:
                self.log_q.put(("__LLM_LINE__", f"启动异常：{e}"))
                res = {"ok": False, "error": str(e)}
            self.log_q.put(("__LLM_SRV_RESULT__", {"action": "start", **res}))

        threading.Thread(target=run, daemon=True).start()

    def llm_server_stop(self):
        try:
            if SRV_DIR not in sys.path:
                sys.path.insert(0, SRV_DIR)
            import llm as llm_mod
            llm_mod.stop_llama_server()
        except Exception as e:
            self.log(f"停止 llama-server 失败：{e}", "err", "srv")
        self.var_llm_srv.set("llama-server: 已停止")

    def llm_install_hip(self):
        if self.llm_install_busy:
            messagebox.showinfo("正在装", "上一次安装还在跑，看右边「服务端日志」。")
            return
        py = self._resolve_python()
        if not py or not os.path.isfile(py):
            messagebox.showerror("没有 python", "先设置「本机 ComfyUI 的 Python」。")
            return
        if not messagebox.askyesno(
                "装 HIP 版 llama-cpp-python",
                f"给这个 python 装/重装成 HIP 后端：\n{py}\n\n"
                "同样先自动下官方**预编译** hip-radeon wheel（约 42MB，免 SDK、免编译）；\n"
                "下不到且本机确实装了 AMD ROCm（HIP SDK，有 hipcc.exe）才退去现场编译。\n"
                "AMD 卡若不是非 HIP 不可，更推荐「装 Vulkan 版」，那条只要显卡驱动。\n"
                "GGUF 跑在服务端同一个 8199 端口里。继续？"):
            return
        self.llm_install_busy = True
        self.var_llm_state.set("LLM: 正在装预编译 HIP-Radeon wheel…（日志在右边「服务端日志」）")
        self.log(f"开始安装 HIP 版 llama-cpp-python → {py}", "warn", "srv")

        def run():
            res = {}
            try:
                if SRV_DIR not in sys.path:
                    sys.path.insert(0, SRV_DIR)
                import llm as llm_mod
                res = llm_mod.install_llama_hip(py, log=lambda s: self.log_q.put(("__LLM_LINE__", s)))
            except Exception as e:
                self.log_q.put(("__LLM_LINE__", f"HIP 安装异常：{e}"))
                res = {"ok": False, "error": str(e)}
            self.log_q.put(("__LLM_INSTALL_RESULT__", res))

        threading.Thread(target=run, daemon=True).start()

    def llm_check_backend(self):
        py = self._resolve_python()
        if not py or not os.path.isfile(py):
            messagebox.showerror("没有 python", "先设置「本机 ComfyUI 的 Python」。")
            return
        self.var_llm_backend.set("计算后端: 检测中…")

        def run():
            try:
                if SRV_DIR not in sys.path:
                    sys.path.insert(0, SRV_DIR)
                import llm as llm_mod
                r = llm_mod.llama_backend(py)
            except Exception as e:
                r = {"backend": "?", "error": str(e)}
            be = str(r.get("backend") or "?")
            detail = ""
            libs = r.get("libs") or []
            for k in ("ggml-vulkan", "ggml-cuda", "ggml-hip", "ggml-cpu", "ggml-metal"):
                if any(k in n for n in libs):
                    detail = k
                    break
            text = f"计算后端: {be.upper()}" + (f"（{detail}）" if detail else "")
            if r.get("error"):
                text += f" · {str(r['error'])[:80]}"
            self.root.after(0, lambda: self.var_llm_backend.set(text))
            self.log_q.put(("__LLM_LINE__", f"[后端检测] {text}"))

        threading.Thread(target=run, daemon=True).start()

    def llm_env_diag(self):
        """把「这个 python 里到底装了几份 llama、哪份在算」一次说清。"""
        py = self._resolve_python()
        if not py or not os.path.isfile(py):
            messagebox.showerror("没有 python", "先设置「本机 ComfyUI 的 Python」。")
            return
        self.var_llm_bench.set("诊断中…")

        def run():
            try:
                if SRV_DIR not in sys.path:
                    sys.path.insert(0, SRV_DIR)
                import llm as llm_mod
                d = llm_mod.llama_env(py)
            except Exception as e:
                d = {"error": str(e)}
            if d.get("error"):
                self.log_q.put(("__LLM_LINE__", f"[诊断] 失败：{d['error'][:200]}"))
                self.root.after(0, lambda: self.var_llm_bench.set("诊断失败，看日志"))
                return
            lines = [f"[诊断] {d.get('what')}   版本 {d.get('version') or '?'}",
                     f"  安装来源：{d.get('import_file') or '（import 不到）'}"]
            dists = d.get("dists") or []
            lines.append(f"  已登记的分发包：{len(dists)} 个"
                         + ("（" + "；".join(f"{x.get('name')} {x.get('version')}" for x in dists) + "）" if dists else ""))
            leftover = d.get("stray") or []
            if leftover:
                lines.append(f"  ⚠ site-packages 里还有 {len(leftover)} 个 llama_cpp 残留（CPU 版残留最大嫌疑，点「卸载」会一起清）：")
                for p in leftover[:6]:
                    lines.append(f"      {p}")
            elif len(d.get("dists") or []) > 1:
                lines.append("  ⚠ 登记了多份 llama-cpp-python（下面列表里看），点「卸载」能一次清干净")
            libs = d.get("libs") or []
            lines.append("  原生库：" + (", ".join(libs[:12]) if libs else "（没有 lib 目录）"))
            lines.append(f"  Vulkan 运行时 vulkan-1.dll：{d.get('vulkan_runtime') or '没找到（显卡跑不起来会静默回落 CPU）'}")
            tl = d.get("torch_lib_gpu") or []
            if tl:
                lines.append("  torch 自带 GPU 库：" + ", ".join(tl[:10]))
            if d.get("devices"):
                lines.append("  看到的设备：" + ", ".join(d["devices"]))
            if d.get("env"):
                lines.append("  相关环境变量：" + "; ".join(f"{k}={v}" for k, v in d["env"].items()))
            if not d.get("gpu_capable"):
                lines.append("  结论：这是 CPU 版 —— 点「卸载 llama-cpp-python」，再点「装 Vulkan 版（AMD·免SDK）」重装")
            else:
                lines.append("  结论：带 GPU 后端。若仍然慢，点「GPU 跑分」看预填/生成两段速度")
            for x in lines:
                self.log_q.put(("__LLM_LINE__", x))
            short = f"{d.get('what')}" + (f"（{d.get('version')}）" if d.get("version") else "")
            self.root.after(0, lambda: self.var_llm_bench.set(short[:46]))
            self.log_q.put(("__HEALTH_Q__", None))

        threading.Thread(target=run, daemon=True).start()

    def llm_uninstall(self):
        """卸载 llama-cpp-python（含旧 CPU 版残留），卸完再装 Vulkan 版。"""
        if self.llm_install_busy:
            messagebox.showinfo("正在忙", "上一次安装还在跑，等它结束。")
            return
        py = self._resolve_python()
        if not py or not os.path.isfile(py):
            messagebox.showerror("没有 python", "先设置「本机 ComfyUI 的 Python」。")
            return
        if not messagebox.askyesno(
                "卸载 llama-cpp-python",
                f"把这个 python 里的 llama-cpp-python 卸干净（含源码编译留下的残留目录）：\n{py}\n\n"
                "卸完需要再点「装 Vulkan 版（AMD·免SDK）」重装，否则 GGUF 加载不了。继续？"):
            return
        if self.is_running():
            if not messagebox.askyesno("服务还在跑", "先停止服务再卸（不然 pyd 被占用删不掉）。现在停掉？"):
                return
            self.stop_server()
        self.llm_install_busy = True
        self.var_llm_state.set("LLM: 正在卸载 llama-cpp-python…")
        self.log(f"开始卸载 llama-cpp-python → {py}", "warn", "srv")

        def run():
            res = {}
            try:
                if SRV_DIR not in sys.path:
                    sys.path.insert(0, SRV_DIR)
                import llm as llm_mod
                res = llm_mod.uninstall_llama(py, log=lambda s: self.log_q.put(("__LLM_LINE__", s)))
            except Exception as e:
                self.log_q.put(("__LLM_LINE__", f"卸载异常：{e}"))
                res = {"ok": False, "error": str(e)}
            self.log_q.put(("__LLM_INSTALL_RESULT__", res))
            if res.get("ok"):
                self.root.after(0, lambda: self.var_llm_bench.set("已卸载 · 接着点「装 Vulkan 版」"))

        threading.Thread(target=run, daemon=True).start()

    def llm_bench(self):
        """让服务端真跑一次，量出预填(PP)/生成两段速度——这两个数字能直接判定谁在算。"""
        if not self.is_running():
            messagebox.showinfo("服务没开", "先启动服务、加载 GGUF，再跑分。")
            return
        self.var_llm_bench.set("跑分中…（约 10~60 秒，看模型大小）")

        def run():
            r = self._http_rpc("llm_bench", {"prompt_tokens": 512, "gen_tokens": 64}, timeout=900)
            if not r or not r.get("ok"):
                err = (r or {}).get("error") or "无响应"
                self.log_q.put(("__LLM_LINE__", f"[跑分] 失败：{err[:200]}"))
                self.root.after(0, lambda: self.var_llm_bench.set("跑分失败，看日志"))
                return
            d = r.get("data") or {}
            lines = [f"[跑分] 模型 {d.get('model') or '?'}  后端 {d.get('backend') or '?'}  设备 {d.get('device_name') or '?'}",
                     f"  预填 PP: {d.get('pp_tps')} tokens/s    生成: {d.get('gen_tps')} tokens/s"
                     f"    （{d.get('gen_tokens')} tokens / {d.get('seconds')}s）",
                     f"  上卡：{d.get('offloaded') or '—'}",
                     f"  结论：{d.get('verdict') or ''}"]
            for ln in d.get("log") or []:
                lines.append("  " + str(ln)[:160])
            for x in lines:
                self.log_q.put(("__LLM_LINE__", x))
            short = f"PP {d.get('pp_tps')} / 生成 {d.get('gen_tps')} tok/s"
            self.root.after(0, lambda: self.var_llm_bench.set(short[:46]))

        threading.Thread(target=run, daemon=True).start()

    def apply_llm(self, st: dict | None):
        if not isinstance(st, dict) or not st:
            self.var_llm_state.set("LLM: 未启动（点「加载 GGUF 到显存」）")
            self.var_llm_rt.set("生成中: —")
            return
        self.llm_status = st
        state = st.get("state") or "未启动"
        model = st.get("model") or "—"
        mm = st.get("mmproj") or ""
        nctx = st.get("n_ctx") or "—"
        handler = st.get("handler") or "—"
        load_s = st.get("load_seconds")
        lcpp = st.get("llama_cpp") or "—"
        dev = st.get("device_name") or "—"
        line = f"LLM: {state}    模型: {model}"
        if st.get("external"):
            line = f"LLM: 外部服务 {st.get('base_url') or ''}    模型: {model}"
        if mm:
            line += f"    mmproj: {mm}"
        line += f"    上下文: {nctx}    handler: {handler}"
        if isinstance(load_s, (int, float)) and model != "—":
            line += f"    加载 {load_s:.1f}s"
        if model != "—":
            line += f"    llama-cpp: {lcpp}"
        line += f"    设备: {dev}"
        bend = st.get("backend") or ""
        if bend and not st.get("external"):
            line += f"    后端: {bend}"
            off = str(st.get("offloaded") or "").strip()
            if off:
                line += f"    上卡: {off[:36]}"
            gb, cb = st.get("gpu_buf_mib"), st.get("cpu_buf_mib")
            if isinstance(gb, (int, float)) and isinstance(cb, (int, float)) and (gb or cb):
                line += f"    缓冲: 显存 {gb/1024:.1f}G / 内存 {cb/1024:.1f}G"
            note = st.get("note") or ""
            if note:
                line += f"    ⚠ {note}"
        self.var_llm_state.set(line)
        last = st.get("last") or {}
        prog = st.get("progress") or {}
        if prog:
            self.var_llm_rt.set(f"生成中: {prog.get('tokens', 0)} tokens · "
                                f"{prog.get('tps') or '—'} tok/s · {prog.get('elapsed', 0)}s"
                                + (f" · {prog.get('images')} 张图" if prog.get("images") else ""))
        elif last:
            self.var_llm_rt.set(f"上次生成: {last.get('tokens', '—')} tokens · "
                                f"{(last.get('tps') or last.get('tokens_per_s') or '—')} tok/s · "
                                f"{last.get('seconds', '—')}s"
                                + (f" · {last.get('images')} 张图" if last.get("images") else "")
                                + (f" · {st.get('image_note')}" if st.get("image_note") else ""))
        else:
            self.var_llm_rt.set("生成中: —（等节点调用）")
        err = st.get("error") or ""
        if err:
            self.var_llm_err.set(f"错误: {err}")
        elif "失败" in state:
            stderr = st.get("stderr") or []
            self.var_llm_err.set("提示: " + (str(stderr[-1])[:160] if stderr else "见右边服务端日志"))
        else:
            self.var_llm_err.set("")

    def _selected_names(self, kind: str) -> list[str]:
        lb = self.lst_clip if kind == "clip" else self.lst_vae
        names = []
        src = list((self.models_view.get(kind) or {}).keys())
        for idx in lb.curselection():
            if 0 <= idx < len(src):
                names.append(src[idx])
        return names

    def preload(self, kind: str):
        if not self.is_running():
            messagebox.showinfo("服务没开", "先点「启动服务」，再预加载模型。")
            return
        names = self._selected_names(kind)
        if not names:
            messagebox.showinfo("没选中", f"先在「服务端 {kind.upper()} 模型」里选一条（可 Ctrl 多选）。")
            return
        self.log(f"请求预加载 {kind}: {', '.join(names)}（首次会真的读权重，耐心等）", "info")

        def work():
            for n in names:
                r = self._http_rpc("load", {"kind": kind, "model": n}, timeout=600)
                if r and r.get("ok"):
                    self.log_q.put(("__PRELOAD_OK__", f"{kind}/{n}"))
                else:
                    msg = (r or {}).get("error", "无响应")
                    self.log_q.put(("__PRELOAD_ERR__", f"{kind}/{n}: {msg}"))
            self.log_q.put(("__HEALTH_Q__", None))

        threading.Thread(target=work, daemon=True).start()

    def delete_entry(self):
        kind = None
        name = None
        for k, lb in (("clip", self.lst_clip), ("vae", self.lst_vae)):
            sel = lb.curselection()
            if sel:
                src = list((self.models_view.get(k) or {}).keys())
                if 0 <= sel[0] < len(src):
                    kind, name = k, src[sel[0]]
                    break
        if not kind:
            messagebox.showinfo("没选中", "先选中一条模型条目")
            return
        if not messagebox.askyesno("确认", f"从 config.json 里删掉 {kind} 条目 {name} ？"):
            return
        (self.cfg.get("models", {}).get(kind) or {}).pop(name, None)
        save_json(CFG_PATH, self._collect_cfg())
        self.log(f"已删除条目 {kind}/{name}", "warn")
        if self.is_running():
            self._reload_server()
        self._refresh_models()

    def scan_add(self, quiet: bool = False):
        """扫「模型根目录」里的 vae / clip / text_encoders / clip_vision，按文件夹名自动归类成条目。
        GGUF 不用登记：服务端会直接从这些根目录下的 LLM 文件夹扫（面板的 GGUF / mmproj 列表里直接出现）。"""
        self._collect_cfg()
        roots = [r.replace("\\", "/") for r in self._roots_from_ui() if r]
        if not roots:
            if not quiet:
                self.log("还没加「模型根目录」：点上面的「添加目录…」选一个 models 目录"
                         "（如 F:\\Comfy-Desktop\\ComfyUI-Shared\\models）", "warn")
            return
        added = 0
        scanned = []
        try:
            if SRV_DIR not in sys.path:
                sys.path.insert(0, SRV_DIR)
            import config as cfgmod
            items = cfgmod.scan_roots(roots)
        except Exception as e:
            self.log(f"扫描失败：{e}", "err")
            return
        counts = {}
        for it in items:
            kind = it["kind"]
            counts[kind] = counts.get(kind, 0) + 1
            if kind not in ("clip", "vae", "clip_vision"):
                continue
            entries = self.cfg.setdefault("models", {}).setdefault(kind, {})
            # 同一个文件已经在 config 里登记过就跳过（按文件名比，兼容手写的相对路径）
            existing = set()
            for e in entries.values():
                files = e.get("files") if isinstance(e, dict) else ([e] if isinstance(e, str) else [])
                for f in files or []:
                    existing.add(os.path.basename(str(f)).lower())
            name = os.path.basename(it["path"]).lower()
            if name in existing:
                continue
            key = cfgmod.entry_name(it["rel"])
            while key in entries:
                key += "_2"
            entries[key] = {"files": [it["path"]], "note": "根目录扫描"}
            added += 1
        for k in ("clip", "vae", "clip_vision", "llm", "llm_vision"):
            if counts.get(k):
                scanned.append(f"{k} {counts[k]}")
        save_json(CFG_PATH, self._collect_cfg())
        if not quiet:
            self.log("扫描根目录：" + ("、".join(scanned) if scanned else "没扫到模型文件"), "info")
        if added:
            self.log(f"新增 {added} 个条目（clip_type 由服务端加载时自动识别）", "ok")
        elif not quiet:
            self.log("没有新条目（扫到的都已经登记过了）", "info")
        if self.is_running():
            self._reload_server()
        self._refresh_models()

    def _reload_server(self):
        """服务端是启动时读的 config.json，扫描/改目录后让它重新读一遍，不用重启。"""
        r = self._http_get("/reload")
        if r and r.get("ok"):
            d = r.get("data") or {}
            counts = d.get("models") or {}
            self.log(f"服务端已重新读取配置：CLIP {counts.get('clip', 0)} 条，VAE {counts.get('vae', 0)} 条", "ok", "srv")
        else:
            self.log(f"服务端没能重读配置（{(r or {}).get('error', '无响应')}），请点「停止服务」再「启动服务」一次",
                     "warn")


    def clear_vram(self):
        if not self.is_running():
            messagebox.showinfo("服务没开", "服务没开，没有需要释放的显存。")
            return
        r = self._http_rpc("unload", {})
        if r and r.get("ok"):
            self.log("已请求服务端卸载全部模型并清空显存", "ok")
        else:
            self.log(f"清空显存失败：{(r or {}).get('error')}", "err")

    def open_dir(self):
        target = clean_path(self.var_comfy.get()) or HERE
        try:
            if os.name == "nt":
                os.startfile(target)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", target])
        except Exception as e:
            self.log(f"打开目录失败：{e}", "err")

    # ------------------------------------------------------------- 事件循环
    def _tick(self):
        try:
            self._drain_queue()
            self.tick += 1
            if self.tick % 4 == 0 and self.is_running():
                self._poll_health()
            if self.tick % 10 == 0:
                self._update_url_line()
            if self.is_running() and self.proc and self.proc.poll() is not None:
                self.log(f"服务端进程已退出（exit {self.proc.returncode}）", "warn", "srv")
                self._after_exit()
        except Exception:
            self.log("界面事件循环异常：\n" + traceback.format_exc(), "err")
        finally:
            self.root.after(250, self._tick)

    def _drain_queue(self):
        while True:
            try:
                item = self.log_q.get_nowait()
            except queue.Empty:
                return
            if isinstance(item, tuple):
                tag, payload = item
                if tag == "__EXIT__":
                    self.log("服务端日志流结束", "info", "srv")
                elif tag == "__HEALTH__":
                    self.apply_health(payload)
                elif tag == "__HEALTH_Q__":
                    self._poll_health()
                elif tag == "__PRELOAD_OK__":
                    self.log(f"预加载完成：{payload}", "ok")
                    self._poll_health()
                elif tag == "__PRELOAD_ERR__":
                    self.log(f"预加载失败：{payload}", "err")
                elif tag == "__LLM_OK__":
                    ok = payload.get("ok")
                    d = payload.get("data") if isinstance(payload.get("data"), dict) else {}
                    if ok:
                        self.log(f"LLM 操作完成：state={d.get('state', '已加载')}", "ok", "srv")
                    else:
                        self.log(f"LLM 操作失败：{payload.get('error')}", "err", "srv")
                elif tag == "__LLM_ERR__":
                    msg = payload.get("error") or "无响应"
                    self.log(f"LLM 操作失败：{msg}", "err", "srv")
                    self.var_llm_err.set(f"错误: {msg}")
                elif tag == "__LLM_LINE__":
                    self.log(str(payload), "info", "srv")
                elif tag == "__LLM_INSTALL_DONE__":
                    self.llm_install_busy = False
                    self.log("llama-cpp-python 安装流程结束；提示词优化节点跑通即说明可用", "ok", "srv")
                elif tag == "__LLM_SRV_RESULT__":
                    self.llm_server_busy = False
                    res = payload or {}
                    act = res.get("action") or ""
                    if act == "download":
                        if res.get("ok"):
                            self.var_llm_srv.set(f"llama-server: 已下载（{res.get('tag')}）")
                            self.log(f"Vulkan llama-server 下好了：{res.get('exe')}", "ok", "srv")
                            self.log("下一步：选好 GGUF → 点「启动 llama-server（吃显卡）」"
                                     "（会自动把「外部 LLM 服务地址」填成 http://127.0.0.1:8300）", "info", "srv")
                        else:
                            self.var_llm_srv.set("llama-server: 下载失败")
                            self.log("下载失败：" + str(res.get("error") or "")[:200], "err", "srv")
                            self.log("可手动下载：https://github.com/ggml-org/llama.cpp/releases "
                                     "里找 llama-*-bin-win-vulkan-x64.zip，解压到 "
                                     + os.path.join(HERE, "llama_server"), "warn", "srv")
                    else:
                        if res.get("ok"):
                            self.var_llm_srv.set(f"llama-server: 运行中 :{res.get('port')}"
                                                 f"（PID {res.get('pid')}）")
                            self.var_llm_ext.set(str(res.get("base_url") or
                                                     f"http://127.0.0.1:{res.get('port')}"))
                            self._save_cfg()
                            self.log("llama-server 已经就绪，面板「外部 LLM 服务地址」已自动填好；"
                                     "点「启动服务」重开服务端后就开始用它算", "ok", "srv")
                        else:
                            self.var_llm_srv.set("llama-server: 启动失败")
                            self.log("llama-server 启动失败：" + str(res.get("error") or "")[:200],
                                     "err", "srv")
                elif tag == "__LLM_INSTALL_RESULT__":
                    self.llm_install_busy = False
                    res = payload or {}
                    if "removed" in res or "still_importable" in res:          # 卸载结果
                        if res.get("ok"):
                            self.var_llm_state.set("LLM: 已卸载 llama-cpp-python（点「装 Vulkan 版」重装）")
                            self.log("llama-cpp-python 已卸载干净（含残留目录）；"
                                     "AMD 卡接着点「装 Vulkan 版（AMD·免SDK）」重装", "ok", "srv")
                        else:
                            self.var_llm_state.set("LLM: 没卸干净（看右边日志）")
                            self.log("卸载没完成：" + str(res.get("error") or "")[:200], "err", "srv")
                    elif res.get("ok"):
                        self.var_llm_state.set(
                            f"LLM: 依赖就绪（llama-cpp-python {res.get('version')}），点「加载 GGUF 到显存」")
                        self.log(f"llama-cpp-python {res.get('version')} 装好了（{res.get('how')}）；"
                                 "现在可以点「加载 GGUF 到显存」", "ok", "srv")
                    else:
                        self.var_llm_state.set("LLM: 依赖没装上（看右边日志最后几行）")
                        last = (res.get("log") or [""])[-1:]
                        self.log("llama-cpp-python 没装上。" + (str(last[0]) if last else ""), "err", "srv")
                        self.log("兜底：AMD 卡点「装 Vulkan 版（AMD·免SDK）」再试一次（会续传上次没下完的部分）；"
                                 "或者按日志里的链接手动下 wheel，再用「装本地 wheel…」", "warn", "srv")
                continue
            line = str(item)
            low = line.lower()
            if "!!" in line or "error" in low or "traceback" in low or "失败" in line:
                self.log(line, "err", "srv")
            elif "warn" in low or "注意" in line or "警告" in line:
                self.log(line, "warn", "srv")
            elif "已启动" in line or "[ok]" in low or "完成" in line:
                self.log(line, "ok", "srv")
            else:
                self.log(line, "info", "srv")

    # ------------------------------------------------------------- 关闭
    def on_close(self):
        try:
            self._save_cfg()
        except Exception:
            pass
        try:
            if SRV_DIR not in sys.path:
                sys.path.insert(0, SRV_DIR)
            import llm as llm_mod
            llm_mod.stop_llama_server()      # 关面板时别把 llama-server 留在显存里
        except Exception:
            pass
        if self.is_running():
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.root.destroy()


def _default_cfg() -> dict:
    return {"server": {"host": "0.0.0.0", "port": 8199, "token": "", "max_request_mb": 256},
            "comfyui_dir": "", "device": "auto",
            "max_loaded": {"clip": 1, "vae": 1},
            "idle_unload_seconds": 600,
            "models": {"clip": {}, "vae": {}}}


def ensure_config() -> None:
    """配置文件不存在就当场生成一份空白模板，保证界面永远能打开。"""
    if not os.path.isfile(CFG_PATH):
        try:
            save_json(CFG_PATH, _default_cfg())
        except Exception:
            pass


def _write_smoke_report(gui: "Gui") -> None:
    """--smoke 自检：把包内自带的服务端是否可用写成 _smoke.txt，方便确认 exe 能独立运行。"""
    app_py = os.path.join(SRV_DIR, "app.py")
    py = clean_path(gui.var_python.get())
    lines = [
        f"frozen        : {bool(getattr(sys, 'frozen', False))}",
        f"data_dir      : {HERE}",
        f"srv_dir       : {SRV_DIR}",
        f"app.py        : {'OK' if os.path.isfile(app_py) else '缺失'} ({app_py})",
        f"codec/backend : "
        f"{'OK' if all(os.path.isfile(os.path.join(SRV_DIR, f)) for f in ('codec.py', 'backend.py', 'config.py')) else '缺失'}",
        f"config.json   : {'OK' if os.path.isfile(CFG_PATH) else '缺失'} ({CFG_PATH})",
        f"python        : {py or '(未设置)'}",
        f"python_torch  : {python_has_torch(py) if py else 'n/a'}",
    ]
    try:
        _write_text(os.path.join(HERE, "_smoke.txt"), "\n".join(lines) + "\n")
    except Exception:
        pass


def main():
    smoke = "--smoke" in sys.argv
    ensure_config()
    root = tk.Tk()
    if os.name == "nt":
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    gui = Gui(root, smoke=smoke)
    root.protocol("WM_DELETE_WINDOW", gui.on_close)
    if smoke:
        root.withdraw()
        root.after(1200, lambda: (gui._save_cfg(), _write_smoke_report(gui), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
