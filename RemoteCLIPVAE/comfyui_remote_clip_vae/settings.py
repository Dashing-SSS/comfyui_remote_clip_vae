# -*- coding: utf-8 -*-
"""settings.py —— A 端默认服务器地址（节点里也可以临时改）。"""

from __future__ import annotations

import json
import os

_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_FILE = os.path.join(_DIR, "default_server.json")

_FALLBACK_URL = "http://192.168.1.20:8199"


def default_server() -> dict:
    url = _FALLBACK_URL
    token = ""
    try:
        with open(_DEFAULT_FILE, "r", encoding="utf-8") as f:
            d = json.load(f) or {}
        url = str(d.get("url") or url).strip() or _FALLBACK_URL
        token = str(d.get("token") or "")
    except Exception:
        pass
    env_url = os.environ.get("REMOTE_CLIPVAE_SERVER")
    if env_url:
        url = env_url.strip()
    env_token = os.environ.get("REMOTE_CLIPVAE_TOKEN")
    if env_token is not None:
        token = env_token
    return {"url": url.rstrip("/"), "token": token}
