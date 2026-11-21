# -*- coding: utf-8 -*-
"""
codec.py —— 两端共享的数据协议（A 端插件与 B 端服务端这个文件必须**逐字节一致**）。

设计目标
--------
ComfyUI 的节点数据里既有"张量"也有"嵌套容器"（例如 CONDITIONING 就是
[[cond_tensor, {"pooled_output": tensor, ...}], ...]，LATENT 是 {"samples": tensor}）。
所以这里提供一个与 Python 对象一一对应的 JSON 结构：

    torch.Tensor  -> {"__t__": "<zlib(torch.save)> 的 base64"}   保 dtype / 形状 / 数值
    dict(str键)    -> {"__d__": {key: pack(value)}}
    dict(非str键)  -> {"__dk__": [[pack(k), pack(v)], ...]}
    list           -> {"__l__": [...]}
    tuple          -> {"__p__": [...]}                           保留元组语义
    bytes          -> {"__b__": "<base64>"}
    numpy.ndarray  -> 自动转 torch.Tensor
    标量 / None    -> 原样透传

浮点精度：张量用 torch.save 原样落盘，float32 无损（不会像 PNG 那样压到 8bit）。

图像通道（v1.7.7 起的变化）
--------------------------
* 多帧 PNG 的解码 / 编码走线程池并行（zlib 压缩会释放 GIL，多帧图像明显更快）。
* 解码不再产生 float64 中间数组，直接在 uint8 上堆叠后一次性转 float32（省两次全图拷贝）。
* 支持 16bit PNG（灰度/彩色的 16bit 会自动降到 8bit）、调色板 P、LA、CMYK、1bit，统一成
  RGB / RGBA / L，多帧之间自动对齐通道数。
* 显式 close 图片句柄，长时间跑不会攒着临时对象。
"""

from __future__ import annotations

import base64
import io
import os
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

MAX_TENSOR_BYTES = 1 << 30  # 1GB，防异常请求撑爆内存
_ZLIB_LEVEL = 6
_PNG_LEVEL = 3

# 解码/编码 PNG 用的线程池：图像是纯 CPU 活，多帧时并行能省一半左右时间
_WORKERS = max(2, min(8, (os.cpu_count() or 4)))
_POOL = ThreadPoolExecutor(max_workers=_WORKERS, thread_name_prefix="codec-img")


# --------------------------------------------------------------------------- #
# 张量 <-> base64
# --------------------------------------------------------------------------- #
def encode_tensor(t: torch.Tensor) -> str:
    if not torch.is_tensor(t):
        raise TypeError(f"encode_tensor 需要 torch.Tensor，收到 {type(t)!r}")
    t = t.detach()
    if t.device.type != "cpu":
        t = t.cpu()
    if not t.is_contiguous():
        t = t.contiguous()
    buf = io.BytesIO()
    torch.save(t, buf)
    return base64.b64encode(zlib.compress(buf.getvalue(), _ZLIB_LEVEL)).decode("ascii")


def decode_tensor(s: str) -> torch.Tensor:
    raw = zlib.decompress(base64.b64decode(s))
    if len(raw) > MAX_TENSOR_BYTES:
        raise ValueError(f"张量数据过大：{len(raw) / 1024 / 1024:.1f} MB")
    stream = io.BytesIO(raw)
    try:
        return torch.load(stream, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0 没有 weights_only 参数
        return torch.load(stream, map_location="cpu")
    except Exception:  # 极少数版本对 zip 格式有兼容问题
        stream.seek(0)
        return torch.load(stream, map_location="cpu", weights_only=False)


# --------------------------------------------------------------------------- #
# 任意对象 <-> JSON
# --------------------------------------------------------------------------- #
def pack(obj):
    if torch.is_tensor(obj):
        return {"__t__": encode_tensor(obj)}
    if isinstance(obj, np.ndarray):
        return {"__t__": encode_tensor(torch.from_numpy(obj))}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        if all(isinstance(k, str) for k in obj):
            return {"__d__": {k: pack(v) for k, v in obj.items()}}
        return {"__dk__": [[pack(k), pack(v)] for k, v in obj.items()]}
    if isinstance(obj, tuple):
        return {"__p__": [pack(v) for v in obj]}
    if isinstance(obj, list):
        return {"__l__": [pack(v) for v in obj]}
    if isinstance(obj, bytes):
        return {"__b__": base64.b64encode(obj).decode("ascii")}
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise TypeError(f"无法序列化的类型：{type(obj)!r}")


def unpack(obj):
    if isinstance(obj, dict):
        if "__t__" in obj:
            return decode_tensor(obj["__t__"])
        if "__d__" in obj:
            return {k: unpack(v) for k, v in obj["__d__"].items()}
        if "__dk__" in obj:
            return {unpack(k): unpack(v) for k, v in obj["__dk__"]}
        if "__l__" in obj:
            return [unpack(v) for v in obj["__l__"]]
        if "__p__" in obj:
            return tuple(unpack(v) for v in obj["__p__"])
        if "__b__" in obj:
            return base64.b64decode(obj["__b__"])
        # 没有标签的字典一律当作普通字典处理
        return {k: unpack(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [unpack(v) for v in obj]
    return obj


# --------------------------------------------------------------------------- #
# 图像帧（PNG）
# --------------------------------------------------------------------------- #
def _encode_one_frame(arr: np.ndarray, compress_level: int = _PNG_LEVEL) -> str:
    if arr.ndim == 2:
        im = Image.fromarray(np.ascontiguousarray(arr))
    elif arr.shape[-1] == 4:
        im = Image.fromarray(np.ascontiguousarray(arr))
    elif arr.shape[-1] == 3:
        im = Image.fromarray(np.ascontiguousarray(arr))
    else:  # 单通道 / 通道数异常
        im = Image.fromarray(np.ascontiguousarray(arr[..., 0]))
    buf = io.BytesIO()
    im.save(buf, format="PNG", compress_level=compress_level)
    return base64.b64encode(buf.getvalue()).decode("ascii")


_ALIGN = 32     # Qwen-VL: patch 16 + n_merge 2 → 宽高对齐到 32 才不浪费 token


def fit_max_pixels(image: torch.Tensor, max_pixels: int) -> torch.Tensor:
    """IMAGE [B,H,W,C] 按「最大像素数」等比缩小（只缩不放），宽高对齐到 32。
    AMD + Vulkan 的视觉编码会整段回落 CPU，耗时随像素数平方涨，所以在节点这边先缩。"""
    if not max_pixels or max_pixels <= 0 or not torch.is_tensor(image):
        return image
    t = image.detach()
    if t.dim() == 3:
        t = t.unsqueeze(0)
    if t.dim() != 4:
        return image
    h, w = int(t.shape[1]), int(t.shape[2])
    if h * w <= max_pixels:
        return image
    k = (float(max_pixels) / float(h * w)) ** 0.5
    nw = max(_ALIGN, int(round(w * k / _ALIGN)) * _ALIGN)
    nh = max(_ALIGN, int(round(h * k / _ALIGN)) * _ALIGN)
    if (nw, nh) == (w, h):
        return image
    x = t.permute(0, 3, 1, 2).float()            # B,C,H,W
    x = torch.nn.functional.interpolate(x, size=(nh, nw), mode="bicubic", align_corners=False,
                                        antialias=True)
    x = x.clamp(0.0, 1.0).permute(0, 2, 3, 1).contiguous()
    print(f"[参考图] {w}×{h}（{h*w/1e6:.2f}MP）→ {nw}×{nh}（{nw*nh/1e6:.2f}MP）", flush=True)
    return x.to(dtype=image.dtype, device=image.device)


def tensor_to_png_frames(image: torch.Tensor, compress_level: int = _PNG_LEVEL,
                         max_pixels: int = 0) -> list[str]:
    """
    IMAGE 张量 [B,H,W,C] float(0~1) -> 每帧一张 PNG 的 base64 列表。
    用 PNG 而不是原始 float：1024x1024 图像 12MB -> PNG 后约 1~2MB，千兆局域网 20~40ms。
    max_pixels > 0 时先按像素上限等比缩小（只缩不放）。
    """
    if not torch.is_tensor(image):
        raise TypeError("tensor_to_png_frames 需要 torch.Tensor")
    t = fit_max_pixels(image, max_pixels).detach()
    if t.device.type != "cpu":
        t = t.cpu()
    if t.dtype != torch.float32:
        t = t.float()
    if t.dim() == 3:
        t = t.unsqueeze(0)
    if t.dim() != 4:
        raise ValueError(f"IMAGE 张量维度应为 4（B,H,W,C），收到 {tuple(t.shape)}")
    t = t.clamp(0.0, 1.0).mul_(255.0).round_()
    arr = t.to(torch.uint8).numpy()
    if arr.shape[0] == 1:
        return [_encode_one_frame(arr[0], compress_level)]
    return list(_POOL.map(lambda a: _encode_one_frame(a, compress_level),
                          [arr[i] for i in range(arr.shape[0])]))


def _decode_one_frame(payload) -> np.ndarray:
    """单帧解码：PNG / JPEG / WebP / BMP 都行，统一返回 uint8 数组（H,W,C）或（H,W）。"""
    raw = base64.b64decode(payload) if isinstance(payload, str) else bytes(payload)
    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        mode = im.mode
        if mode in ("P", "PA"):
            im = im.convert("RGBA" if (mode == "PA" or "transparency" in im.info) else "RGB")
        elif mode in ("CMYK", "YCbCr", "LAB", "HSV"):
            im = im.convert("RGB")
        elif mode in ("1", "L", "LA", "I", "I;16", "I;16B", "I;16L", "F"):
            if mode in ("1",):
                im = im.convert("L")
        arr = np.asarray(im)
    if arr.dtype != np.uint8:
        if arr.dtype == np.uint16:
            arr = (arr >> 8).astype(np.uint8)
        elif arr.dtype.kind == "i":  # I;16 在 PIL 里常是 int32
            arr = np.clip(arr, 0, 65535)
            arr = (arr.astype(np.uint32) >> 8).astype(np.uint8)
        elif arr.dtype.kind == "f":
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return arr


def _fit_channels(arr: np.ndarray, ch: int) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == ch:
        return arr
    if arr.shape[-1] > ch:
        return arr[..., :ch]
    # 通道不够（比如灰度当灰度用）：复制最后一维
    return np.repeat(arr[..., -1:], ch - arr.shape[-1] + 1, axis=-1)[..., :ch]


def png_frames_to_tensor(frames: list[str], channels: int | None = None) -> torch.Tensor:
    """PNG/JPEG base64 列表 -> IMAGE 张量 [B,H,W,C] float(0~1)，多帧自动对齐通道数。"""
    if not frames:
        raise ValueError("图像帧列表为空")
    if len(frames) == 1:
        arrs = [_decode_one_frame(frames[0])]
    else:
        arrs = list(_POOL.map(_decode_one_frame, frames))

    ref = arrs[0]
    ch = ref.shape[-1] if ref.ndim == 3 else 1
    hw = ref.shape[:2]
    arrs = [_fit_channels(a, ch) for a in arrs]
    for i, a in enumerate(arrs):
        if a.shape[:2] != hw:  # 尺寸不一致：按最近邻缩放到第一帧
            with Image.fromarray(a[..., 0] if ch == 1 else a) as im:
                a = np.asarray(im.resize((hw[1], hw[0]), Image.NEAREST))
                a = a[..., None] if a.ndim == 2 else a

    stack = np.stack(arrs, axis=0)
    t = torch.from_numpy(np.ascontiguousarray(stack))
    t = t.to(torch.float32).div_(255.0)

    if channels is not None and t.shape[-1] != channels:
        if t.shape[-1] > channels:
            t = t[..., :channels]
        else:
            t = t.repeat(1, 1, 1, channels)[..., :channels]
    return t.contiguous()


# --------------------------------------------------------------------------- #
# 自测
# --------------------------------------------------------------------------- #
def _selftest():
    cond = [[torch.randn(1, 77, 768, dtype=torch.float32),
             {"pooled_output": torch.randn(1, 1280), "area": (0, 0, 1024, 1024)}]]
    back = unpack(pack(cond))
    assert torch.equal(back[0][0], cond[0][0]), "张量往返不一致"
    assert back[0][1]["area"] == (0, 0, 1024, 1024), "元组未保留"
    assert isinstance(back[0], list), "外层应是 list"

    img = torch.rand(2, 64, 64, 3)
    fr = tensor_to_png_frames(img)
    back_img = png_frames_to_tensor(fr)
    assert back_img.shape == img.shape, f"{back_img.shape} != {img.shape}"
    assert (back_img - img).abs().max() < 1 / 255 + 1e-6, "PNG 往返误差异常"

    # 灰度、RGBA、单帧
    g = torch.rand(1, 32, 32, 1)
    assert png_frames_to_tensor(tensor_to_png_frames(g)).shape == g.shape
    rgba = torch.rand(1, 32, 32, 4)
    assert png_frames_to_tensor(tensor_to_png_frames(rgba)).shape == rgba.shape

    latent = {"samples": torch.randn(1, 4, 32, 32)}
    assert torch.equal(unpack(pack(latent))["samples"], latent["samples"])

    print(f"[OK] codec 自测通过（{_WORKERS} 线程解码），图像往返误差 < 1/255")


if __name__ == "__main__":
    _selftest()
