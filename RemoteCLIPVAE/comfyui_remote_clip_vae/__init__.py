# -*- coding: utf-8 -*-
"""
ComfyUI-RemoteCLIPVAE —— 图片生成机（本机 ComfyUI）侧的插件。

本机只跑 UNet / DiT 出图，文本编码（CLIP）和 VAE 编解码交给同一网络里另一台机器，
那台机器上跑的是本套件里的「RemoteCLIPVAE服务端」。两边用 HTTP 传张量，无需共享显存。

节点（分类：comfyui_remote_clip_vae）
    远程 CLIP 加载器 (Remote CLIP Loader)   → clip
    远程 VAE 加载器 (Remote VAE)            → vae
    远程 CLIP 文本编码 (Remote CLIP Encode) → 正向条件 / 负向条件
    远程 CLIP 图文编码 (Remote CLIP Encode + Images) → 正向条件 / 负向条件（图片1..10 自动增槽）
    提示词优化 (Prompt Optimizer, Remote)   → prompt / json / 是否反推（走远端 OpenAI 兼容 LLM）
    远程 VAE 解码 (Remote VAE Decode)       → 图像
    远程 VAE 编码 (Remote VAE Encode)       → latent
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from . import api

api.register()

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

print("[remote-clipvae] 节点已加载：远程 CLIP 加载器 / 远程 VAE 加载器 / 远程 CLIP 文本编码 / "
      "远程 CLIP 图文编码 / 提示词优化 / 远程 VAE 解码·编码")
