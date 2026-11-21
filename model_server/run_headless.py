# -*- coding: utf-8 -*-
"""
run_headless.py —— 控制台式启动入口（保持与常见一键包一致的调用方式）：

    python run_headless.py --backend comfy --comfy-path "F:\\ComfyUI" [--port 8199] [--mock]

本包的模型后端只有一个：comfy —— 因为 CLIP/VAE 必须用 ComfyUI 自己的
comfy.sd 加载，才能和 A 端的 UNet 完全同源（这是能出正常图的关键）。
所以这里不接受 --backend hf 之类的独立实现，避免"能跑但出鬼图"。
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser(description="RemoteCLIPVAE 服务端（headless 控制台模式）")
    ap.add_argument("--backend", default="comfy", choices=["comfy"],
                    help="只支持 comfy：复用 ComfyUI 的 comfy.sd 加载 CLIP/VAE")
    ap.add_argument("--comfy-path", dest="comfy_path", default=None,
                    help="ComfyUI 源码根目录（含 comfy 文件夹）")
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--mock", action="store_true", help="联调模式：不加载真模型")
    ap.add_argument("--debug", action="store_true")
    args, unknown = ap.parse_known_args()

    argv = [sys.argv[0], "--config", args.config, "--backend", args.backend]
    if args.comfy_path:
        argv += ["--comfy-path", args.comfy_path]
    if args.host:
        argv += ["--host", args.host]
    if args.port:
        argv += ["--port", str(args.port)]
    if args.mock:
        argv += ["--mock"]
    if args.debug:
        argv += ["--debug"]

    sys.argv = argv
    import app  # 同目录的 app.py
    return app.main()


if __name__ == "__main__":
    sys.exit(main())
