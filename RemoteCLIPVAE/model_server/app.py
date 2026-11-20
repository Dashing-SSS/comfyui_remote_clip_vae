# -*- coding: utf-8 -*-
"""
server.py —— B 电脑上的 HTTP 服务端（只用标准库，零额外依赖）。

启动：
    python server.py --config config.json
    python server.py --config config.json --mock     # 联调自测模式，不加载真模型

接口：
    GET  /health               健康检查 + 模型/显存状态
    GET  /models               可用模型清单（客户端节点的下拉框就是读这个）
    POST /rpc  {"op": ..., "params": {...}}   实际工作接口
         op = clip_encode | vae_decode | vae_encode | load | unload | status | models
    （可选鉴权：config.json 里设置 server.token 后，请求需带 X-Auth-Token 头）
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 整合包/便携版 ComfyUI 的 python_embeded 带 python3xx._pth，会丢掉 sys.path[0]（脚本所在目录），
# 导致 import codec 直接失败 —— 这里显式补上本目录。
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import codec
from backend import Backend, MockBackend, ServerError

CFG = {}
STATE = {"backend": None, "started": time.time(), "debug": False}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "RemoteModelServer/1.0"

    # ---------------- 基础工具 ---------------- #
    def _json(self, code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _auth_ok(self) -> bool:
        token = str((CFG.get("server") or {}).get("token") or "")
        if not token:
            return True
        got = self.headers.get("X-Auth-Token", "")
        if got != token:
            self._json(401, {"ok": False, "error": "token 校验失败，请检查 客户端节点里的 token 与 服务端 config.json 一致"})
            return False
        return True

    def _read_body(self) -> bytes:
        max_mb = float((CFG.get("server") or {}).get("max_request_mb", 256) or 256)
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            raise ServerError("请求没有 Content-Length")
        if length > max_mb * 1024 * 1024:
            raise ServerError(f"请求体 {length / 1024 / 1024:.1f}MB 超过上限 {max_mb}MB")
        chunks, got = [], 0
        while got < length:
            part = self.rfile.read(min(1 << 20, length - got))
            if not part:
                break
            chunks.append(part)
            got += len(part)
        return b"".join(chunks)

    def log_message(self, fmt, *args):  # 只记关键信息，避免刷屏
        if STATE["debug"]:
            sys.stderr.write("[HTTP] %s - %s\n" % (self.address_string(), fmt % args))

    # ---------------- GET ---------------- #
    def do_GET(self):
        global CFG
        path = self.path.split("?")[0].rstrip("/") or "/"
        if not self._auth_ok():
            return
        if path in ("", "/", "/health", "/status"):
            try:
                status = STATE["backend"].status()
                status["uptime_s"] = round(time.time() - STATE["started"], 1)
                status["server"] = "RemoteModelServer"
                self._json(200, {"ok": True, "data": status})
            except Exception as e:
                self._json(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})
            return
        if path == "/models":
            try:
                self._json(200, {"ok": True, "data": STATE["backend"].list_models()})
            except Exception as e:
                self._json(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})
            return
        if path == "/llm_status":
            # 面板/客户端轮询「LLM 运行实时信息」用（已加载模型、tokens/s、进度、错误）
            try:
                be = STATE["backend"]
                st = be.llm.status() if hasattr(be, "llm") else {}
                st["ok"] = True
                self._json(200, st)
            except Exception as e:
                self._json(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})
            return
        if path == "/reload":
            # 界面扫描/改模型目录后，让服务端重读 config.json，不用重启进程
            try:
                cfg_path = STATE.get("config_path")
                new_cfg = CFG
                if cfg_path and os.path.exists(cfg_path):
                    try:
                        import config as _cfg
                        new_cfg = _cfg.load(cfg_path)
                    except Exception:
                        with open(cfg_path, "r", encoding="utf-8") as f:
                            new_cfg = json.load(f)
                CFG = new_cfg
                fn = getattr(STATE["backend"], "reload", None)
                data = fn(new_cfg) if fn else {"note": "该后端不支持重读配置"}
                self._json(200, {"ok": True, "data": data})
            except Exception as e:
                self._json(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})
            return
        self._json(404, {"ok": False, "error": f"未知路径 {self.path}，可用：/health /models /rpc"})

    # ---------------- POST ---------------- #
    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        if not self._auth_ok():
            return
        if path != "/rpc":
            self._json(404, {"ok": False, "error": f"未知路径 {self.path}，只支持 POST /rpc"})
            return
        t0 = time.time()
        try:
            body = self._read_body()
            req = json.loads(body.decode("utf-8"))
            op = str(req.get("op", ""))
            params = codec.unpack(req.get("params") or {})
            result = STATE["backend"].handle(op, params)
            dt = time.time() - t0
            if STATE["debug"]:
                print(f"[B] op={op} ok {dt * 1000:.0f}ms", flush=True)
            self._json(200, {"ok": True, "data": codec.pack(result), "ms": round(dt * 1000, 1)})
        except ServerError as e:
            self._fail(400, str(e), e)
        except json.JSONDecodeError as e:
            self._fail(400, f"请求不是合法 JSON：{e}", e)
        except Exception as e:
            self._fail(500, f"{type(e).__name__}: {e}", e)

    def _fail(self, code: int, msg: str, exc):
        tb = traceback.format_exc()
        print(f"[B] !! 失败：{msg}\n{tb}", flush=True)
        payload = {"ok": False, "error": msg}
        if STATE["debug"]:
            payload["trace"] = tb
        self._json(code, payload)


def _lan_ips() -> list[str]:
    ips = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    try:  # 兜底：用 UDP 探测出口网卡
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.3)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip and ip not in ips:
            ips.insert(0, ip)
    except Exception:
        pass
    return sorted([ip for ip in ips if not ip.startswith("127.")], key=_ip_rank)


def _ip_rank(ip: str) -> int:
    """普通局域网优先（192.168 / 10 / 172），tailscale 的 100.64~100.127 往后排。"""
    try:
        a, b = ip.split(".")[:2]
        a, b = int(a), int(b)
    except Exception:
        return 9
    if a == 100 and 64 <= b <= 127:
        return 5
    if ip.startswith("192.168."):
        return 0
    if a == 10:
        return 1
    if a == 172:
        return 2
    return 3


def main():
    ap = argparse.ArgumentParser(description="RemoteCLIPVAE 模型服务端（CLIP 文本编码 + VAE 编解码）")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"))
    ap.add_argument("--host", default=None, help="覆盖 config.json 里的监听地址")
    ap.add_argument("--port", type=int, default=None, help="覆盖 config.json 里的端口")
    ap.add_argument("--backend", default="comfy", choices=["comfy"],
                    help="模型后端，本包只支持 comfy（复用 ComfyUI 的 comfy.sd，保证与 客户端 UNet 同源）")
    ap.add_argument("--comfy-path", default=None, dest="comfy_path",
                    help="覆盖 config.json 里的 comfyui_dir（ComfyUI 源码根目录）")
    ap.add_argument("--mock", action="store_true", help="联调模式：不加载真模型，返回随机张量")
    ap.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"],
                    help="计算设备：auto（默认，有显卡用显卡）/ cuda / cpu（强制纯 CPU）")
    ap.add_argument("--debug", action="store_true", help="打印每个请求与完整堆栈")
    args = ap.parse_args()

    global CFG
    cfg_path = os.path.abspath(args.config)
    if os.path.exists(cfg_path):
        try:
            import config as _cfg          # 复用同一套清洗/回退逻辑
            CFG = _cfg.load(cfg_path)
        except Exception:
            with open(cfg_path, "r", encoding="utf-8") as f:
                CFG = json.load(f)
        print(f"[B] 已读取配置 {cfg_path}")
    else:
        print(f"[B] 警告：找不到 {cfg_path}，使用默认配置")
        CFG = {"server": {"host": "0.0.0.0", "port": 8199}, "comfyui_dir": "", "models": {"clip": {}, "vae": {}}}

    if args.comfy_path:
        CFG["comfyui_dir"] = args.comfy_path
        print(f"[B] 命令行指定 ComfyUI 目录：{args.comfy_path}")

    if args.device:
        CFG["device"] = args.device
        print(f"[B] 命令行指定计算设备：{args.device}")

    STATE["debug"] = bool(args.debug)
    srv_cfg = CFG.get("server") or {}
    host = args.host or srv_cfg.get("host", "0.0.0.0")
    port = args.port or int(srv_cfg.get("port", 8199))

    config_dir = os.path.dirname(cfg_path)
    STATE["config_path"] = cfg_path
    STATE["backend"] = (MockBackend if args.mock else Backend)(CFG, config_dir)

    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True

    ips = _lan_ips()
    print("=" * 62)
    print(f" 模型服务已启动{' 【MOCK 联调模式】' if args.mock else ''}")
    print(f" 监听：http://{host}:{port}")
    for ip in ips:
        print(f" 客户端节点里填：http://{ip}:{port}")
    print(f" 自检：浏览器打开 http://{(ips[0] if ips else '127.0.0.1')}:{port}/health")
    print("=" * 62)
    if not args.mock:
        try:
            models = STATE["backend"].list_models()
            missing = [f"{k}/{n}" for k, v in models.items() for n, m in v.items() if not m.get("exists")]
            if missing:
                print(f"[B] 注意：以下模型文件的路径不存在，请检查 config.json：{missing}")
            if not models.get("clip") or not models.get("vae"):
                print("[B] 注意：models.clip 或 models.vae 为空，客户端将没有模型可选")
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[B] 收到 Ctrl+C，退出")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
