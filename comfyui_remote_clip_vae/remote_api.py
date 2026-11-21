# -*- coding: utf-8 -*-
"""
remote_api.py —— ComfyUI 侧访问服务端（跑 CLIP/VAE 那台机器）的轻量客户端。
只用标准库 urllib，不依赖 requests，也不加载任何模型到 A 的显存。
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request

from .codec import pack, unpack


def _base(url: str) -> str:
    url = (url or "").strip()
    if not url:
        raise RuntimeError("server_url 为空，请在节点里填写服务端地址，例如 http://192.168.1.20:8199")
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "http://" + url
    if ":" not in url.split("//", 1)[-1]:
        url = url + ":8199"
    return url.rstrip("/")


def _headers(token: str) -> dict:
    h = {"Content-Type": "application/json", "Accept": "application/json"}
    if token:
        h["X-Auth-Token"] = token
    return h


def _short(err: str, limit: int = 400) -> str:
    err = (err or "").strip()
    return err if len(err) <= limit else err[:limit] + " ..."


def rpc(url: str, token: str, op: str, params: dict | None = None, timeout: float | None = None):
    """调用服务端 /rpc，返回已反序列化的 Python 对象。失败抛 RuntimeError（文案直接给用户看）。

    v1.7.34 起 `timeout` 默认 None = **没有超时限制**：CLIP/VAE 首次加载、大图编码、
    提示词模型慢慢生成，都会一直等到服务端算完为止（以前 120s / 600s 会中途砍掉）。
    """
    base = _base(url)
    body = json.dumps({"op": op, "params": pack(params or {})}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(base + "/rpc", data=body, method="POST", headers=_headers(token))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raw = e.read() or b"{}"
        try:
            err = json.loads(raw.decode("utf-8", "replace")).get("error", "")
        except Exception:
            err = raw.decode("utf-8", "replace")
        raise RuntimeError(f"【服务端】{op} 失败（HTTP {e.code}）：{_short(err)}") from None
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
        reason = getattr(e, "reason", e)
        raise RuntimeError(
            f"无法连接服务端 {base}：{reason}。\n"
            f"排查：① 服务端是否在运行；② 两台机器是否同一局域网；"
            f"③ 服务端防火墙是否放行该端口（Windows 需放行入站 TCP）；④ IP/端口是否填对"
        ) from None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        raise RuntimeError(f"【服务端】{op} 返回的不是合法 JSON：{_short(raw.decode('utf-8', 'replace'), 200)}") from None
    if not payload.get("ok"):
        raise RuntimeError(f"【服务端】{op} 失败：{_short(payload.get('error', '未知错误'))}")
    return unpack(payload.get("data"))


def health(url: str, token: str = "", timeout: float | None = None) -> dict:
    base = _base(url)
    req = urllib.request.Request(base + "/health", headers=_headers(token))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read() or b""
        txt = raw.decode("utf-8", "replace").strip()
        try:
            obj = json.loads(txt)
            txt = str(obj.get("error") or obj.get("detail") or txt)
        except Exception:
            pass
        raise RuntimeError(
            f"这个地址上跑的不是本服务端（HTTP {e.code}：{(txt or '无响应内容')[:200]}）。\n"
            f"本服务端的 /health 会返回 {{\"ok\": true, ...}}；现在这个端口是别的程序在回话。"
        ) from None
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
        raise RuntimeError(f"无法连接服务端 {base}：{getattr(e, 'reason', e)}") from None
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error", "未知错误"))
    return payload.get("data") or {}


def unload(url: str, token: str = "", kind: str = "", model: str = "", timeout: float | None = None) -> dict:
    """让服务端把 CLIP / VAE / clip_vision 踢出显存；kind 留空 = 全部，model 留空 = 该类全部。"""
    return rpc(url, token, "unload", {"kind": kind or "", "model": model or ""}, timeout=timeout)


def llm_unload(url: str, token: str = "", timeout: float | None = None) -> dict:
    """让服务端把提示词优化用的 GGUF（llama-cpp 子进程）卸载/退出。"""
    return rpc(url, token, "llm_unload", {}, timeout=timeout)


def list_models(url: str, token: str = "", kind: str = "clip", timeout: float = 4.0) -> tuple[list[str], str]:
    """返回 (模型名列表, 错误信息)。错误时列表为空，不影响 ComfyUI 启动。"""
    try:
        data = health(url, token, timeout)
    except Exception as e:
        return [], _short(str(e), 160)
    models = (data.get("models") or {}).get(kind) or {}
    if not models:
        return [], f"服务端 config.json 的 models.{kind} 为空"
    return sorted(models.keys()), ""
