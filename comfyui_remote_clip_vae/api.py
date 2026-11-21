# -*- coding: utf-8 -*-
"""
api.py —— 给前端「🔄 刷新模型列表」按钮用的本机代理路由。

浏览器直接跨域请求服务端会被 CORS 挡，所以由 ComfyUI 自己的 aiohttp 代发一次短请求：

    GET /remote_clipvae/probe?url=http://10.0.10.2:8199&token=&kind=clip
    → {"ok": true, "device": "NVIDIA GeForce RTX 4070", "vram_gb": 12.0,
       "models": ["amix_clip_768", ...], "error": ""}
"""

from __future__ import annotations

import asyncio

from . import remote_api


def _fallback_token(token: str) -> str:
    """节点上已经没有 token 栏了：前端没传就退回 default_server.json / 侧边栏设置里的 token。"""
    if token:
        return token
    try:
        from .settings import default_server

        return str(default_server().get("token") or "")
    except Exception:
        return ""


def _is_cpu_only(data: dict) -> bool:
    """服务端是不是纯 CPU 在算：device 是 cpu，或者压根没报到显卡型号。"""
    dev = str(data.get("device") or "").strip().lower()
    name = str(data.get("device_name") or "").strip()
    if data.get("mock"):
        return False
    if dev.startswith("cpu"):
        return True
    if dev in ("", "auto") and (not name or name.upper() == "CPU"):
        return True
    return False


def _device_text(data: dict) -> str:
    """「服务端显卡」的展示文案：只有型号，不带 CUDA / ROCm 这类计算方式。"""
    name = str(data.get("device_name") or "").strip()
    if data.get("mock"):
        return name or "MOCK（联调）"
    if _is_cpu_only(data) or not name or name.upper() == "CPU":
        return "（无显卡，纯 CPU 计算）"
    return name


def _compute_text(data: dict) -> str:
    """「计算设备」：远端到底在用显卡还是 CPU 算（节点侧自动识别，用户不用选）。"""
    if data.get("mock"):
        return "MOCK"
    return "CPU" if _is_cpu_only(data) else "显卡"


def _looks_like_comfyui(url: str, token: str) -> bool:
    """端口上是不是个 ComfyUI：它的 /system_stats 会回 {"system": {...}}。"""
    import json
    import urllib.request

    base = (url or "").strip()
    if not base:
        return False
    if not base.startswith("http://") and not base.startswith("https://"):
        base = "http://" + base
    if ":" not in base.split("//", 1)[-1]:
        base += ":8199"
    try:
        req = urllib.request.Request(base.rstrip("/") + "/system_stats")
        if token:
            req.add_header("X-Auth-Token", token)
        with urllib.request.urlopen(req, timeout=3.5) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
        return isinstance(data, dict) and ("system" in data or "devices" in data)
    except Exception:
        return False


def _probe(url: str, token: str, kind: str) -> dict:
    try:
        data = remote_api.health(url, token, timeout=5.0)
    except Exception as e:
        msg = str(e)[:300]
        hint = ""
        if "Unexpected endpoint" in msg or "HTTP 404" in msg or "HTTP 405" in msg:
            if _looks_like_comfyui(url, token):
                hint = ("\n→ 这个端口上跑的是 **ComfyUI**（它的 /system_stats 有响应），不是本服务端。"
                        "把「服务端地址」改成服务端 exe 面板标题上那个 http://<服务端IP>:8199，"
                        "并确认面板「运行状态」是运行中。")
            else:
                hint = ("\n→ 端口对了但没有这个接口：说明这个地址上跑的不是 RemoteCLIPVAE 服务端"
                        "（ComfyUI 8188 / LM Studio / Ollama 之类也会回这种 404）。"
                        "请把「服务端地址」改成服务端面板标题上写的 http://<服务端IP>:8199，"
                        "并确认面板状态是「运行中」。")
        elif "无法连接" in msg or "Connection refused" in msg or "refused" in msg:
            hint = ("\n→ 这个端口没人监听：服务端 exe 没启动，或 IP/端口填错，或被防火墙挡了"
                    "（Windows 要放行入站 TCP 8199）。")
        return {"ok": False, "device": "", "compute": "", "models": [], "error": msg + hint}
    models = sorted(((data.get("models") or {}).get(kind) or {}).keys())
    empty_hint = f"服务端 models.{kind} 为空"
    if kind == "llm":
        empty_hint = "服务端没扫到 GGUF：在面板「GGUF 模型目录」里选 <ComfyUI>\\models\\LLM，或点「刷新模型列表」"
    elif kind == "llm_vision":
        empty_hint = "服务端没扫到 mmproj（视觉投影）：纯文本模型可以只用文本，接图做反推才需要"
    return {"ok": True, "device": _device_text(data), "compute": _compute_text(data),
            "vram_gb": data.get("vram"),
            "models": models, "error": "" if models else empty_hint}


def _unload(url: str, token: str, kind: str, model: str) -> dict:
    """「🗑 卸载模型」按钮：让服务端把指定模型踢出显存。"""
    try:
        if kind == "llm":
            data = remote_api.llm_unload(url, token, timeout=180)
        else:
            data = remote_api.unload(url, token, kind, model, timeout=180)
    except Exception as e:
        msg = str(e)[:300]
        hint = ""
        if "无法连接" in msg or "refused" in msg.lower() or "Connection" in msg:
            hint = "\n→ 连不上服务端：exe 没启动 / IP 端口错 / 防火墙没放行。"
        return {"ok": False, "freed": [], "error": msg + hint}
    freed = []
    vram_before = vram_after = None
    loaded = None
    if isinstance(data, dict):
        freed = data.get("freed") or []
        vram_before = data.get("vram_before")
        vram_after = data.get("vram_after")
        loaded = data.get("loaded")
        if not freed and data.get("state"):
            freed = [f"llm: {data.get('state')}"]
        elif not freed and kind == "llm":
            freed = ["llm: worker 进程已退出"]
        if vram_after is None:
            vram_after = data.get("vram")
    return {"ok": True, "freed": freed, "loaded": loaded,
            "vram_before": vram_before, "vram_after": vram_after, "error": ""}


def _local_models() -> dict:
    """切到「本地」时用：把本机 <ComfyUI>/models/LLM 里的 GGUF 给前端，填 文生图/图生图/视觉模型 三个下拉。"""
    try:
        from .nodes import _local_gguf_scan, _local_gguf_dirs
    except Exception as e:
        return {"ok": False, "t2i": [], "i2i": [], "llm_vision": [], "error": f"插件加载异常：{e}"}
    try:
        names, visions = _local_gguf_scan()
        dirs = _local_gguf_dirs()
    except Exception as e:
        return {"ok": False, "t2i": [], "i2i": [], "llm_vision": [], "error": str(e)[:300]}

    def _hit(n, keys):
        low = n.lower()
        return any(k in low for k in keys)

    t2i_only = [n for n in names if _hit(n, ("t2i", "text2image", "txt2img"))]
    i2i_only = [n for n in names if _hit(n, ("i2i", "img2img", "image2image", "edit", "inpaint"))]
    rest = [n for n in names if n not in t2i_only and n not in i2i_only]
    t2i = t2i_only + rest if t2i_only else list(names)
    i2i = i2i_only + rest if i2i_only else list(names)
    err = ""
    if not names:
        err = ("本机 models/LLM 里没扫到 GGUF：" + ("、".join(dirs) if dirs else
               "连 models/LLM 目录都没有（就在 <ComfyUI>/models/ 下建一个 LLM 文件夹把 gguf 放进去）"))
    return {"ok": True, "models": names, "t2i": t2i, "i2i": i2i,
            "llm_vision": visions, "dirs": dirs, "error": err}


def register() -> None:
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return
    try:
        srv = getattr(PromptServer, "instance", None)
        routes = getattr(srv, "routes", None) if srv is not None else None
    except Exception:
        routes = None
    if routes is None:
        return

    @routes.get("/remote_clipvae/probe")
    async def _probe_route(request):
        url = request.query.get("url", "") or ""
        token = _fallback_token(request.query.get("token", "") or "")
        kind = request.query.get("kind", "clip") or "clip"
        if kind not in ("clip", "vae", "llm", "llm_vision"):
            kind = "clip"
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, lambda: _probe(url, token, kind))
        except Exception as e:  # 绝不把异常抛给前端
            result = {"ok": False, "device": "", "compute": "", "models": [], "error": str(e)[:300]}
        return web.json_response(result)

    @routes.get("/remote_clipvae/local_models")
    async def _local_models_route(request):
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, _local_models)
        except Exception as e:
            result = {"ok": False, "t2i": [], "i2i": [], "llm_vision": [], "error": str(e)[:300]}
        return web.json_response(result)

    @routes.get("/remote_clipvae/unload")
    async def _unload_route(request):
        url = request.query.get("url", "") or ""
        token = _fallback_token(request.query.get("token", "") or "")
        kind = request.query.get("kind", "clip") or "clip"
        model = request.query.get("model", "") or ""
        if kind not in ("clip", "vae", "clip_vision", "llm", ""):
            kind = "clip"
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, lambda: _unload(url, token, kind, model))
        except Exception as e:
            result = {"ok": False, "freed": [], "error": str(e)[:300]}
        return web.json_response(result)
