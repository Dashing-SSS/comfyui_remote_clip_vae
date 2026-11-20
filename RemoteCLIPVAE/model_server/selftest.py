# -*- coding: utf-8 -*-
"""
selftest.py —— 不发 ComfyUI，直接按节点背后的逻辑调一遍服务端。

    python selftest.py --url http://127.0.0.1:8199
    python selftest.py --url http://192.168.1.20:8199 --clip flux_auto_clip --vae flux_ae
    python selftest.py --url ... --skip-clip     # 只测 VAE

退出码 0 表示全部通过。必须用 ComfyUI 的 python（要 torch / PIL）。
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PKG_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8199")
    ap.add_argument("--token", default="")
    ap.add_argument("--clip", default=None, help="CLIP 模型名，默认取服务端第一个")
    ap.add_argument("--vae", default=None, help="VAE 模型名，默认取服务端第一个")
    ap.add_argument("--text", default="a photo of a cat, masterpiece, best quality")
    ap.add_argument("--skip-vae", action="store_true", help="只测 CLIP")
    ap.add_argument("--skip-clip", action="store_true", help="只测 VAE")
    ap.add_argument("--tiled", action="store_true", help="VAE 用分块，验证 tile 路径")
    args = ap.parse_args()

    import torch
    import comfyui_remote_clip_vae.nodes as nodes
    import comfyui_remote_clip_vae.remote_api as rc

    def step(t):
        print(f"\n=== {t} ===")

    step("1. /health")
    st = rc.health(args.url, args.token, timeout=10)
    print(f"    设备 {st.get('device_name')}  CUDA={st.get('cuda_available')}  "
          f"torch={st.get('torch')}  mock={bool(st.get('mock'))}")
    clips = sorted(((st.get("models") or {}).get("clip") or {}).keys())
    vaes = sorted(((st.get("models") or {}).get("vae") or {}).keys())
    print(f"    CLIP: {clips}")
    print(f"    VAE : {vaes}")
    if not clips and not args.skip_clip:
        raise SystemExit("服务端没有 CLIP 模型，请先改 config.json")

    clip_name = args.clip or (clips[0] if clips else None)
    vae_name = args.vae or (vaes[0] if vaes else None)

    if not args.skip_clip:
        step("2. CLIP 加载 + 文本编码（首次会触发服务端加载模型）")
        h = nodes.RemoteCLIPLoader().load(args.url, clip_name, args.token, 0, True)[0]
        conds = nodes.RemoteCLIPTextEncode().encode(h, args.text, "cpu", 1800)[0]
        ctx = conds[0][0]
        print(f"    CONDITIONING {len(conds)} 条  context={tuple(ctx.shape)} dtype={ctx.dtype}")
        print(f"    附加键={ {k: (tuple(v.shape) if hasattr(v, 'shape') else v) for k, v in conds[0][1].items()} }")
        assert ctx.dim() == 3 and ctx.shape[0] == 1, f"context 形状异常 {tuple(ctx.shape)}"
        assert ctx.shape[-1] in (768, 1024, 1280, 2048, 4096, 12288), f"context 维度可疑 {ctx.shape[-1]}"

    if args.skip_vae or not vae_name:
        print("\n[OK] CLIP 链路通过（跳过 VAE）")
        return 0

    step("3. VAE 加载 + 解码")
    vh = nodes.RemoteVAELoader().load(args.url, vae_name, args.token, True)[0]
    info = rc.rpc(args.url, args.token, "vae_info", {"model": vae_name}, 600)
    print(f"    VAE 信息 {info}")
    ch = int(info.get("latent_channels", 4))
    ratio = int(info.get("downscale_ratio", 8))
    img = nodes.RemoteVAEDecode().decode(vh, {"samples": torch.zeros(1, ch, 32, 32)},
                                         bool(args.tiled), "cpu", 512, 64, 1800)[0]
    print(f"    IMAGE={tuple(img.shape)} 范围=({img.min():.2f},{img.max():.2f})")
    assert img.dim() == 4 and img.shape[-1] in (3, 4), f"图像形状异常 {tuple(img.shape)}"
    assert img.shape[1] == 32 * ratio, f"图像尺寸应 = latent×{ratio}，实际 {tuple(img.shape)}"

    step("4. VAE 编码")
    lat = nodes.RemoteVAEEncode().encode(vh, img, bool(args.tiled), 512, 64, 1800)[0]
    print(f"    LATENT={tuple(lat['samples'].shape)}")
    assert lat["samples"].dim() == 4 and lat["samples"].shape[1] == ch

    step("5. 释放模型")
    rc.rpc(args.url, args.token, "unload", {}, 120)
    print("    已请求服务端释放")

    print("\n[OK] 全链路通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
