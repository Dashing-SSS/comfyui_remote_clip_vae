# -*- coding: utf-8 -*-
"""
nodes.py —— ComfyUI 侧的「远程加载」节点。

设计要点：
  * 本机只保存一个极小的"代理对象"（地址 + 模型名 + 超时），绝不加载模型，
    显存全部留给图片生成模型（UNet / DiT）。
  * 加载器输出的是**真正的 CLIP / VAE 类型**（鸭子类型代理），所以可以直连
    ComfyUI 自带的 CLIPTextEncode / VAEDecode / VAEEncode / CLIPSetLastLayer，
    也可以连本套件的「远程 CLIP 文本编码」「远程 VAE 解码 / 编码」。
  * 文本编码结果（CONDITIONING）和 VAE 结果都是纯张量，跨机无损；图像走 PNG。

节点（分类 comfyui_remote_clip_vae）：
  远程 CLIP 加载器 (Remote CLIP Loader)   → clip (类型 CLIP)
  远程 VAE 加载器 (Remote VAE)            → vae  (类型 VAE)
  远程 CLIP 文本编码 (Remote CLIP Encode) → 正向条件 / 负向条件
  远程 VAE 解码 (Remote VAE Decode)       → 图像
  远程 VAE 编码 (Remote VAE Encode)       → latent
"""

from __future__ import annotations

import json
import os
import random
import re
import time

import torch

from . import remote_api
from .codec import png_frames_to_tensor, tensor_to_png_frames
from .settings import default_server

CATEGORY = "comfyui_remote_clip_vae"
_LIST_TTL = 60.0

GPU_HINT = "（点「🔄 刷新模型列表」获取）"
FAIL_HINT = "（连接失败，检查服务端）"
NO_MODEL = "（没有可用模型，去服务端加目录后重扫）"

CLIP_TYPES = [
    "自动（按文件名判断）",
    "stable_diffusion",
    "sdxl",
    "sd3",
    "flux",
    "qwen_image",
    "hidream",
    "wan",
    "chroma",
    "ltxv",
    "hunyuan_video",
    "stable_cascade",
    "t5",
    "mochi",
]

_LIST_CACHE: dict = {}
_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache.json")


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _load_cache() -> dict:
    try:
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_cache(key: str, names: list[str]) -> None:
    """把上次成功拿到的清单落盘：ComfyUI 重启时服务端还没开，下拉也是真名字。"""
    try:
        data = _load_cache()
        data[key] = list(names)
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _model_options(kind: str, url: str, token: str) -> tuple[list[str], str]:
    key = (kind, (url or "").strip(), (token or "").strip())
    now = time.time()
    hit = _LIST_CACHE.get(key)
    if hit and now - hit[0] < _LIST_TTL:
        return hit[1], hit[2]
    names, err = remote_api.list_models(url, token, kind)
    if names:
        _save_cache(f"{kind}|{key[1]}", names)
    else:
        disk = _load_cache().get(f"{kind}|{key[1]}") or []
        if isinstance(disk, list) and disk:
            names, err = disk, ""
    _LIST_CACHE[key] = (now, names, err)
    return names, err


def _cfg_token() -> str:
    """token 不再出现在节点上：统一取默认服务端配置（default_server.json / 侧边栏设置）里的 token。"""
    try:
        return str(default_server().get("token") or "")
    except Exception:
        return ""


# ---- 「卸载模型」四档：自动 / 保持加载 / 运行后自动卸载 / 运行前后自动卸载 ----
UNLOAD_AUTO = "自动"
UNLOAD_KEEP = "保持加载"
UNLOAD_AFTER = "运行后自动卸载"
UNLOAD_BOTH = "运行前后自动卸载"
UNLOAD_MODES = [UNLOAD_AUTO, UNLOAD_KEEP, UNLOAD_AFTER, UNLOAD_BOTH]


def _unload_plan(mode: str) -> tuple[bool, bool]:
    """返回 (运行前是否卸载, 运行后是否卸载)。"""
    m = (mode or UNLOAD_AUTO).strip()
    if m == UNLOAD_BOTH:
        return True, True
    if m == UNLOAD_AFTER:
        return False, True
    return False, False


def _unload_remote(url: str, token: str, kind: str, model: str = "", why: str = "") -> list:
    """让服务端把某个模型踢出显存（kind 留空 = 该类全部）；失败只打印，不打断出图。"""
    try:
        data = remote_api.rpc(url, token, "unload",
                              {"kind": kind or "", "model": model or ""}, timeout=180) or {}
        freed = data.get("freed") or []
        print(f"[远程卸载] {why or kind}: {kind or '*'}={model or '*'} → 释放 {freed or '（本来就没加载）'}")
        return freed
    except Exception as e:
        print(f"[远程卸载] {why or kind} 跳过：{e}")
        return []


# ---- 提示词优化的「本地模式」：直接在本机 ComfyUI 的 python 里跑 llama-cpp ----
_LOCAL_LLM: dict = {}
LOCAL_GGUF_SUBS = ("LLM", "llm", "llm_gguf", "LLM_gguf", "gguf", "text_encoders_llm")


def _local_gguf_dirs() -> list:
    try:
        import folder_paths

        root = folder_paths.models_dir
    except Exception:
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    dirs = []
    for sub in LOCAL_GGUF_SUBS:
        p = os.path.join(root, sub)
        if os.path.isdir(p) and p not in dirs:
            dirs.append(p)
    return dirs


def _local_gguf_scan() -> tuple[list, list]:
    """扫本机 models/LLM 一类的目录：返回 (文本 GGUF, mmproj 视觉投影)。"""
    names, visions = [], []
    for d in _local_gguf_dirs():
        for dp, _dn, fs in os.walk(d):
            for f in fs:
                if not f.lower().endswith(".gguf"):
                    continue
                rel = os.path.relpath(os.path.join(dp, f), d).replace("\\", "/")
                low = f.lower()
                if any(k in low for k in ("mmproj", "vision", "vit")):
                    visions.append(rel)
                else:
                    names.append(rel)
    return sorted(set(names)), sorted(set(visions))


def _local_gguf_path(name: str) -> str:
    """把下拉里的相对名解析成本机绝对路径；绝对路径也直接认。"""
    n = str(name or "").strip()
    if not n or n.startswith("（") or n.startswith("("):
        return ""
    if os.path.isabs(n) and os.path.exists(n):
        return n
    for d in _local_gguf_dirs():
        p = os.path.join(d, n.replace("/", os.sep))
        if os.path.exists(p):
            return p
    base = os.path.basename(n)
    for d in _local_gguf_dirs():
        for dp, _dn, fs in os.walk(d):
            if base in fs:
                return os.path.join(dp, base)
    return ""


def _local_llm_free():
    for k, obj in list(_LOCAL_LLM.items()):
        try:
            del obj
        except Exception:
            pass
        _LOCAL_LLM.pop(k, None)
    try:
        import gc

        gc.collect()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _local_llm_get(path: str, mmproj: str, n_ctx: int):
    """按 (模型, mmproj, n_ctx) 缓存一个本地 Llama 实例；换模型就先把旧的放掉。"""
    key = (path, mmproj, int(n_ctx))
    hit = _LOCAL_LLM.get(key)
    if hit is not None:
        return hit
    try:
        from llama_cpp import Llama
    except Exception as e:
        raise RuntimeError(
            "本机没有 llama-cpp-python，用不了本地模式。\n"
            "二选一：① 在本机 ComfyUI 的 python 里装 `llama-cpp-python`（Windows 上建议用 "
            "JamePeng/llama-cpp-python 的预编译 wheel，和 CUDA 版本对齐）；\n"
            f"② 把开关「使用远程模型」打开，改用服务端那台机器上的 GGUF。\n原始错误：{e}"
        ) from None
    _local_llm_free()
    kw = dict(model_path=path, n_ctx=int(n_ctx), n_gpu_layers=-1, verbose=False)
    llm = Llama(**kw)
    handler_name = ""
    if mmproj:
        handler = None
        for cls_name in ("Qwen25VLChatHandler", "Qwen2VLChatHandler", "Llava15ChatHandler", "LlavaChatHandler"):
            try:
                mod = __import__("llama_cpp.llama_chat_format", fromlist=[cls_name])
                handler = getattr(mod, cls_name)(clip_model_path=mmproj, verbose=False)
                handler_name = cls_name
                break
            except Exception:
                handler = None
                continue
        if handler is None:
            raise RuntimeError("本机这份 llama-cpp-python 不支持多模态 chat handler，带图请把开关打开走服务端。")
        llm.chat_handler = handler
    _LOCAL_LLM[key] = llm
    print(f"[提示词优化-本地] 已加载本地 GGUF：{os.path.basename(path)} | mmproj={os.path.basename(mmproj) or '无'}"
          f"{' | handler=' + handler_name if handler_name else ''} | n_ctx={n_ctx}")
    return llm


def _combo(kind: str, url: str, token: str) -> tuple[list, dict]:
    names, err = _model_options(kind, url, token)
    if not names:
        msg = FAIL_HINT if err else NO_MODEL
        return [msg], {"default": msg}
    return names, {"default": names[0]}


def _pick_model(model: str, manual: str, kind: str) -> str:
    """手动模型名（仅编码节点还有）优先；都没填就报清楚的中文错误。"""
    name = (manual or "").strip() or (model or "").strip()
    if not name or name.startswith("（") or name.startswith("("):
        raise ValueError(
            f"没有可用的 {kind.upper()} 模型：先在服务端界面选好模型目录并扫描，"
            f"再点节点上的「🔄 刷新模型列表」，在「模型」下拉里选一个。"
        )
    return name


def _clip_type_value(v: str) -> str:
    """自动 = 让服务端按 config.json 里的 clip_type 走。
    兼容旧标签写法，如 "flux（Flux 的 t5xxl / clip_l）" → "flux"。"""
    v = (v or "").strip()
    if not v or v.startswith("自动"):
        return ""
    for sep in ("（", "("):
        if sep in v:
            v = v.split(sep)[0]
    return v.strip()


def _norm_conditioning(conds) -> list:
    """统一成 ComfyUI 的 CONDITIONING：[[tensor, dict], ...]"""
    out = []
    for item in conds or []:
        if torch.is_tensor(item):
            out.append([item, {}])
        elif isinstance(item, dict):
            d = dict(item)
            cond = d.pop("cond", None)
            if cond is None:
                raise RuntimeError("服务端返回的 conditioning 结构无法识别")
            out.append([cond, d])
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            out.append([item[0], dict(item[1] or {})])
        else:
            raise RuntimeError(f"服务端返回的 conditioning 结构无法识别：{type(item)!r}")
    if not out:
        raise RuntimeError("服务端返回的 conditioning 为空")
    return out


def _target_device(pref: str):
    """pref: cpu / gpu / auto。auto = 与原生节点一致（模型设备）。"""
    if pref == "cpu":
        return None
    try:
        import comfy.model_management as mm
        return mm.get_torch_device()
    except Exception:
        return None


def _to_device(t, pref: str):
    if isinstance(t, list):
        return [_to_device(v, pref) for v in t]
    if isinstance(t, tuple):
        return tuple(_to_device(v, pref) for v in t)
    if isinstance(t, dict):
        return {k: _to_device(v, pref) for k, v in t.items()}
    if not torch.is_tensor(t):
        return t
    dev = _target_device(pref)
    return t if dev is None else t.to(dev)


def _check_url(url: str):
    if not url or url.strip().startswith("<") or url.strip().startswith("（"):
        raise ValueError("请先填写「服务端地址」，形如 http://192.168.1.20:8199")


def _validate_model(name: str) -> bool | str:
    """「模型」是运行时从服务端拉的动态清单，ComfyUI 启动时固化的那份列表必然过时，
    所以不走白名单校验，只把明显是占位符的值拦下来。"""
    v = (name or "").strip()
    if not v or v.startswith("（") or v.startswith("("):
        return ("请点「🔄 刷新模型列表」选一个模型，或者在「手动模型名」里填服务端 config.json 的条目名")
    return True


def _handle_of(obj):
    """老工作流里连过来的是 v1.7.3 的字典句柄，也兼容。"""
    return obj if isinstance(obj, dict) else None


def _connect_error(url: str, e) -> RuntimeError:
    return RuntimeError(
        f"连不上服务端 {url}：{e}\n"
        f"排查：① 服务端程序是否在运行；② 两台机器是否同一局域网；"
        f"③ 服务端防火墙是否放行该端口；④ 地址里的 IP/端口是否填对"
    )


# --------------------------------------------------------------------------- #
# 图片接口（动态增槽）
# --------------------------------------------------------------------------- #
IMAGE_SLOTS = 10

MODE_TEXT = "关闭（只编码文字）"
MODE_VL = "图片 → 服务端文本编码器（Qwen-VL 图文混编，Qwen-Image 编辑用）"
MODE_LATENT = "参考图 → 服务端 VAE 编码（Qwen/Flux Edit 式）"
MODE_VISION = "图片 → 服务端 CLIP 视觉（Redux 式）"
IMAGE_MODES = [MODE_TEXT, MODE_VL, MODE_LATENT, MODE_VISION]


def _collect_images(kw: dict) -> list:
    """把 图片1..图片10 里连上的图按顺序收起来。"""
    out = []
    for i in range(1, IMAGE_SLOTS + 1):
        v = kw.get(f"图片{i}")
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            out.extend([x for x in v if x is not None])
        else:
            out.append(v)
    return out


def _apply_extra(conds: list, extra: dict) -> list:
    """给 conditioning 挂附加信息（reference_latents / clip_vision_output）。"""
    try:
        import node_helpers  # ComfyUI 自带
        for k, v in extra.items():
            conds = node_helpers.conditioning_set_values(conds, {k: v}, append=False)
        return conds
    except Exception:
        out = []
        for c, d in conds:
            d2 = dict(d)
            d2.update(extra)
            out.append([c, d2])
        return out


class _VisionOut:
    """clip_vision 输出的鸭子类型替身（有 comfy 时用官方 Output 类）。"""

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, item):
        setattr(self, key, item)


def _make_vision_output(payload: dict) -> object:
    try:
        from comfy.clip_vision import Output as _Out
        out = _Out()
    except Exception:
        out = _VisionOut()
    for k in ("image_embeds", "penultimate_hidden_states", "last_hidden_state"):
        v = payload.get(k)
        if torch.is_tensor(v):
            out[k] = v
    out["image_sizes"] = payload.get("image_sizes") or []
    return out


def _cat_vision_outputs(parts: list) -> object:
    out = _make_vision_output({})
    for k in ("image_embeds", "penultimate_hidden_states", "last_hidden_state"):
        vals = [p[k] for p in parts if torch.is_tensor(getattr(p, k, None)) and getattr(p, k).shape[0] > 0]
        if not vals:
            continue
        try:
            out[k] = torch.cat(vals, dim=0)
        except Exception:
            out[k] = vals[0]
    sizes = []
    for p in parts:
        sizes.extend(list(getattr(p, "image_sizes", []) or []))
    out["image_sizes"] = sizes
    return out


# --------------------------------------------------------------------------- #
# 远程 CLIP 代理（鸭子类型：可直接接原生 CLIPTextEncode / CLIPSetLastLayer）
# --------------------------------------------------------------------------- #
class RemoteCLIP:
    def __init__(self, url: str, token: str, model: str, clip_type: str = "",
                 timeout: float = 120.0, last_layer: int = 0):
        self.remote = True
        self.server_url = url
        self.token = token or ""
        self.model = model
        self.clip_type = clip_type or ""
        self.timeout = float(timeout or 120)
        self.last_layer = int(last_layer or 0)
        # 原生代码偶尔会摸这几个属性
        self.device = torch.device("cpu")
        self.layer_idx = None
        self.patcher = None
        self.cond_stage_model = None
        self.use_clip_schedule = False
        self.options = {}
        self._img_cache = {}

    def __repr__(self) -> str:
        return f"<RemoteCLIP model={self.model!r} server={self.server_url!r} layer={self.last_layer}>"

    # ---- 原生节点用到的接口 --------------------------------------------- #
    def tokenize(self, text, return_word_ids=False, images=None, keep_vision=None,
                 prevent_empty_text=False, llama_template=None, **kwargs):
        """原生节点（如「文本编码 Qwen Image 编辑 Plus」）是这样调的：
        clip.tokenize(prompt, images=images_vl, keep_vision=..., prevent_empty_text=True)

        真正的分词在服务端做，所以这里必须把 images / keep_vision 一起带过去；
        否则图片会被悄悄丢掉（图片没进 vision 槽 → 参考 latent 插错位置 → 出图糊）。
        """
        payload = {"text": text if isinstance(text, str) else str(text),
                   "keep_vision": None if keep_vision is None else bool(keep_vision),
                   "prevent_empty_text": bool(prevent_empty_text),
                   "llama_template": llama_template if isinstance(llama_template, str) else ""}
        if images:
            payload["images"] = self._pack_images(images)
        # 文本放两份：encode_from_tokens_scheduled 直接读 __remote_clip_text__，
        # 早先只塞在内层 payload 里、_text_of 又只查顶层，导致服务端收到空文本（出图糊/重影）
        return {"__remote_clip_text__": payload["text"], "__remote_clip__": payload}

    def _pack_images(self, images) -> list:
        groups = []
        for img in images:
            if img is None:
                continue
            if not torch.is_tensor(img):
                img = torch.as_tensor(img)
            key = (id(img), tuple(img.shape))
            hit = self._img_cache.get(key)
            if hit is None:
                t = img
                if t.dim() == 3:
                    t = t.unsqueeze(0)
                if t.shape[-1] > 3:
                    t = t[..., :3]
                hit = tensor_to_png_frames(t.detach().float().cpu())
                if len(self._img_cache) > 8:
                    self._img_cache.clear()
                self._img_cache[key] = hit
            groups.append(hit)
        return groups

    @staticmethod
    def _text_of(tokens) -> str:
        if isinstance(tokens, dict):
            if "__remote_clip_text__" in tokens:
                return tokens["__remote_clip_text__"]
            inner = tokens.get("__remote_clip__")
            if isinstance(inner, dict) and isinstance(inner.get("text"), str):
                return inner["text"]
            if "text" in tokens:
                return tokens["text"]
            for v in tokens.values():  # 兼容原生 tokenize 产出的结构
                if isinstance(v, str):
                    return v
                if isinstance(v, (list, tuple)) and v and isinstance(v[0], str):
                    return " ".join(str(x) for x in v)
            return ""
        return tokens if isinstance(tokens, str) else str(tokens)

    def encode_from_tokens_scheduled(self, tokens, opts=None):
        text = self._text_of(tokens)
        payload = {"model": self.model, "text": text, "clip_type": self.clip_type,
                   "last_layer": self.last_layer}
        if isinstance(tokens, dict) and "__remote_clip__" in tokens:
            p = tokens["__remote_clip__"]
            if p.get("images"):
                # 原生节点已经把图缩放到 32 的倍数，服务端不要再缩一次
                payload["images"] = p["images"]
                payload["no_resize"] = True
                payload["prevent_empty_text"] = bool(p.get("prevent_empty_text"))
                if p.get("llama_template"):
                    payload["llama_template"] = p["llama_template"]
                if p.get("keep_vision") is not None:
                    payload["keep_vision"] = bool(p["keep_vision"])
        try:
            data = remote_api.rpc(
                self.server_url, self.token, "clip_encode", payload,
                timeout=self.timeout,
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise _connect_error(self.server_url, e) from None
        return _norm_conditioning((data or {}).get("conditioning"))

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        conds = self.encode_from_tokens_scheduled(tokens)
        cond, extra = conds[0]
        pooled = extra.get("pooled_output")
        if return_dict:
            d = {"cond": cond}
            d.update(extra)
            return d
        if return_pooled:
            return cond, pooled
        return cond

    def encode(self, text, **kwargs):
        return self.encode_from_tokens_scheduled(self.tokenize(text))

    def clip_layer(self, layer_idx: int):
        # 与原生一致：原地改 + 返回自己（CLIPSetLastLayer 就是这么用的）
        self.last_layer = int(layer_idx)
        self.layer_idx = int(layer_idx)
        return self

    def clone(self):
        return RemoteCLIP(self.server_url, self.token, self.model, self.clip_type,
                          self.timeout, self.last_layer)

    def encode_with_images(self, text: str, images: list, vision_pixels: int = 1048576,
                           vae_model: str = ""):
        """图文混编：把参考图交给服务端文本编码器（Qwen-VL 系支持 images= 的 vision token）。
        给了 vae_model 时服务端会同时用 VAE 编 reference_latents（官方 Qwen-Image-Edit 的做法）。"""
        groups = []
        for img in images:
            t = img if torch.is_tensor(img) else torch.as_tensor(img)
            if t.dim() == 3:
                t = t.unsqueeze(0)
            # 不在这里砍 alpha：服务端 _qwen_vl_conditioning 会「视觉塔按白底合成、VAE 保留 alpha」
            groups.append(tensor_to_png_frames(t.detach().float().cpu()))
        px = int(vision_pixels or 0)
        res = int(round(px ** 0.5)) if px > 0 else 0
        try:
            data = remote_api.rpc(
                self.server_url, self.token, "clip_encode",
                {"model": self.model, "text": text, "clip_type": self.clip_type,
                 "last_layer": self.last_layer, "images": groups,
                 "vision_pixels": px, "vae": vae_model or "", "resolution": res,
                 # 有 VAE 时服务端会自己挂 reference_latents → 必须 keep_vision=False，
                 # 否则 vision token 与 latent 双重插入（出图糊/重影）
                 "keep_vision": not bool(vae_model)},
                timeout=self.timeout,
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise _connect_error(self.server_url, e) from None
        return _norm_conditioning((data or {}).get("conditioning"))

    def set_clip_options(self, options: dict):
        self.options = dict(options or {})

    def get_clip_options(self):
        return dict(self.options)


# --------------------------------------------------------------------------- #
# 远程 VAE 代理（鸭子类型：可直接接原生 VAEDecode / VAEEncode / VAEDecodeTiled）
# --------------------------------------------------------------------------- #
class RemoteVAE:
    def __init__(self, url: str, token: str, model: str, timeout: float = 600.0):
        self.remote = True
        self.server_url = url
        self.token = token or ""
        self.model = model
        self.timeout = float(timeout or 600)
        # 原生代码会读的 VAE 属性，先给默认值；第一次 decode 时向服务端问真实值
        self.downscale_ratio = 8
        self.upscale_ratio = 8
        self.spatial_compression_ratio = 8
        self.latent_channels = 4
        self.output_channels = 3
        self.latent_dim = 2
        self.device = torch.device("cpu")
        self.vae_dtype = torch.float32
        self.first_stage_model = None
        self.memory_used_decode = lambda *a, **k: 0
        self.memory_used_encode = lambda *a, **k: 0
        self.server_info: dict = {}

    def __repr__(self) -> str:
        return f"<RemoteVAE model={self.model!r} server={self.server_url!r}>"

    # ---- 内部 ---------------------------------------------------------- #
    def _info(self) -> dict:
        try:
            data = remote_api.rpc(self.server_url, self.token, "vae_info",
                                  {"model": self.model}, timeout=min(self.timeout, 60.0))
        except Exception:
            return {}
        if isinstance(data, dict):
            self.server_info = dict(data)
            for key in ("latent_channels", "downscale_ratio", "upscale_ratio"):
                val = data.get(key)
                if isinstance(val, int) and val > 0:
                    setattr(self, key, val)
                    if key == "downscale_ratio":
                        self.spatial_compression_ratio = val
        return self.server_info

    # ---- 原生接口 ------------------------------------------------------- #
    def decode(self, samples_in):
        _ = self._info()
        t = samples_in if torch.is_tensor(samples_in) else torch.as_tensor(samples_in)
        data = remote_api.rpc(
            self.server_url, self.token, "vae_decode",
            {"model": self.model, "latent": t.detach().float().cpu()},
            timeout=self.timeout,
        )
        return png_frames_to_tensor(data["frames"], channels=int(data.get("channels") or 3))

    def encode(self, pixels):
        _ = self._info()
        t = pixels if torch.is_tensor(pixels) else torch.as_tensor(pixels)
        frames = tensor_to_png_frames(t.detach().float().cpu())
        data = remote_api.rpc(
            self.server_url, self.token, "vae_encode",
            {"model": self.model, "frames": frames},
            timeout=self.timeout,
        )
        latent = data["samples"]
        return latent if torch.is_tensor(latent) else torch.as_tensor(latent)

    def encode_ref(self, image, max_pixels: int = 1048576):
        """参考图 → 服务端 VAE latent（Qwen/Flux Edit 的 reference_latents 用）。"""
        _ = self._info()
        t = image if torch.is_tensor(image) else torch.as_tensor(image)
        if t.dim() == 3:
            t = t.unsqueeze(0)
        if t.shape[-1] > 3:
            t = t[..., :3]
        frames = tensor_to_png_frames(t.detach().float().cpu())
        try:
            data = remote_api.rpc(
                self.server_url, self.token, "ref_latent_encode",
                {"model": self.model, "frames": frames,
                 "max_pixels": int(max_pixels or 0), "multiple": 8},
                timeout=self.timeout,
            )
        except RuntimeError:
            raise
        except Exception as e:
            raise _connect_error(self.server_url, e) from None
        latent = data.get("samples") if isinstance(data, dict) else data
        if not torch.is_tensor(latent):
            raise RuntimeError("服务端没有返回参考图 latent（ref_latent_encode）")
        return latent

    def decode_tiled(self, samples, tile_x=None, tile_y=None, overlap=None, **kwargs):
        _ = self._info()
        t = samples if torch.is_tensor(samples) else torch.as_tensor(samples)
        data = remote_api.rpc(
            self.server_url, self.token, "vae_decode",
            {"model": self.model, "latent": t.detach().float().cpu(),
             "tiled": True, "tile_size": int(tile_x or 512), "overlap": int(overlap or 64)},
            timeout=self.timeout,
        )
        return png_frames_to_tensor(data["frames"], channels=int(data.get("channels") or 3))

    def encode_tiled(self, pixels, tile_x=None, tile_y=None, overlap=None, **kwargs):
        _ = self._info()
        t = pixels if torch.is_tensor(pixels) else torch.as_tensor(pixels)
        frames = tensor_to_png_frames(t.detach().float().cpu())
        data = remote_api.rpc(
            self.server_url, self.token, "vae_encode",
            {"model": self.model, "frames": frames,
             "tiled": True, "tile_size": int(tile_x or 512), "overlap": int(overlap or 64)},
            timeout=self.timeout,
        )
        latent = data["samples"]
        return latent if torch.is_tensor(latent) else torch.as_tensor(latent)

    def throw_exception_if_invalid(self):
        self._info()
        return None

    # 原生 VAEDecodeTiled 等节点会问这几个（视频 VAE 才有时间维压缩，返回 None = 不压缩）
    def temporal_compression_decode(self):
        return None

    def temporal_compression_encode(self):
        return None

    # 注意 ComfyUI 源码里就是拼成 spacial_ 的，照抄才不会 AttributeError
    def spacial_compression_decode(self):
        self._info()
        return int(self.downscale_ratio or 8)

    def spacial_compression_encode(self):
        self._info()
        return int(self.downscale_ratio or 8)

    def process_input(self, image):
        return image

    def process_output(self, image):
        return image

    # 有些节点会问"这个 vae 有多大 / 什么 dtype"
    def get_vaes(self):
        return [self]


# --------------------------------------------------------------------------- #
# 1. 远程 CLIP 加载器
# --------------------------------------------------------------------------- #
class RemoteCLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):
        cfg = default_server()
        models, model_opt = _combo("clip", cfg["url"], cfg["token"])
        return {
            "required": {
                "服务端显卡": ("STRING", {"default": GPU_HINT,
                                        "tooltip": "服务端显卡，点「🔄 刷新模型列表」自动填，只用于看，不用改"}),
                "服务端地址": ("STRING", {"default": cfg["url"],
                                        "tooltip": "另一台机器（跑 CLIP/VAE 那台）的地址，含端口"}),
                "模型": (models, {"default": model_opt["default"],
                                "tooltip": "服务端已配置的 CLIP 条目名"}),
                "超时时间": ("INT", {"default": 5, "min": 1, "max": 3600, "step": 1,
                                    "tooltip": "连接服务端的超时（秒）；真正编码的等待在编码节点里"}),
                "模型类型": (CLIP_TYPES, {"default": "自动（按文件名判断）",
                                        "tooltip": "自动 = 用服务端的 clip_type；Flux/SD3 等对不上时可手动指定"}),
                "卸载模型": (UNLOAD_MODES, {"default": UNLOAD_AUTO,
                                        "tooltip": "自动=交给服务端按需管理；保持加载=不自动卸载；"
                                                   "运行后/运行前后自动卸载=服务端显存紧时用"}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = ("远程 CLIP（类型就是 CLIP，可直接接自带的 CLIPTextEncode / CLIPSetLastLayer）。"
                   "文本编码在服务端执行，本机不占显存。")

    @classmethod
    def VALIDATE_INPUTS(cls, 模型=None, 模型类型=None):
        if 模型 is None:
            return True
        return _validate_model(模型)

    def load(self, **kw):
        url = kw.get("服务端地址", "")
        _check_url(url)
        # 「手动模型名」已删除；老工作流里如果还有这个值，仍然认（不影响新节点）
        model = _pick_model(kw.get("模型", ""), kw.get("手动模型名", ""), "clip")
        token = _cfg_token()
        before, after = _unload_plan(kw.get("卸载模型"))
        if before:
            _unload_remote(url, token, "clip", model, "CLIP 加载器(运行前)")
        try:
            remote_api.health(url, token, timeout=float(kw.get("超时时间") or 5))
        except Exception as e:
            raise _connect_error(url, e) from None
        clip = RemoteCLIP(url, token, model, _clip_type_value(kw.get("模型类型", "")), timeout=120.0)
        print(f"[远程CLIP] 句柄就绪：{url} | 模型 {model}"
              f"{' | clip_type=' + clip.clip_type if clip.clip_type else ''}")
        if after:
            _unload_remote(url, token, "clip", model, "CLIP 加载器(运行后)")
        return (clip,)


# --------------------------------------------------------------------------- #
# 2. 远程 VAE 加载器
# --------------------------------------------------------------------------- #
class RemoteVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        cfg = default_server()
        models, model_opt = _combo("vae", cfg["url"], cfg["token"])
        return {
            "required": {
                "服务端显卡": ("STRING", {"default": GPU_HINT}),
                "服务端地址": ("STRING", {"default": cfg["url"]}),
                "模型": (models, {"default": model_opt["default"]}),
                "超时时间": ("INT", {"default": 5, "min": 1, "max": 3600, "step": 1}),
                "卸载模型": (UNLOAD_MODES, {"default": UNLOAD_AUTO}),
            },
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = ("远程 VAE（类型就是 VAE，可直接接自带的 VAEDecode / VAEEncode / VAEDecodeTiled）。"
                   "编解码在服务端执行。")

    @classmethod
    def VALIDATE_INPUTS(cls, 模型=None):
        if 模型 is None:
            return True
        return _validate_model(模型)

    def load(self, **kw):
        url = kw.get("服务端地址", "")
        _check_url(url)
        # 「手动模型名」已删除；老工作流里如果还有这个值，仍然认（不影响新节点）
        model = _pick_model(kw.get("模型", ""), kw.get("手动模型名", ""), "vae")
        token = _cfg_token()
        before, after = _unload_plan(kw.get("卸载模型"))
        if before:
            _unload_remote(url, token, "vae", model, "VAE 加载器(运行前)")
        try:
            remote_api.health(url, token, timeout=float(kw.get("超时时间") or 5))
        except Exception as e:
            raise _connect_error(url, e) from None
        vae = RemoteVAE(url, token, model, timeout=600.0)
        print(f"[远程VAE] 句柄就绪：{url} | 模型 {model}")
        if after:
            _unload_remote(url, token, "vae", model, "VAE 加载器(运行后)")
        return (vae,)


# --------------------------------------------------------------------------- #
# 3. 远程 CLIP 文本编码（正向 + 负向一次搞定）
# --------------------------------------------------------------------------- #
class RemoteCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        cfg = default_server()
        models, model_opt = _combo("clip", cfg["url"], cfg["token"])
        out = {
            "required": {
                "正向提示词": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "负向提示词": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "服务端显卡": ("STRING", {"default": GPU_HINT}),
                "服务端地址": ("STRING", {"default": cfg["url"]}),
                "模型": (models, {"default": model_opt["default"]}),
                "超时时间": ("INT", {"default": 120, "min": 1, "max": 7200, "step": 1,
                                  "tooltip": "单次编码超时（秒），首次加载大 CLIP 可能要几分钟"}),
                "手动模型名": ("STRING", {"default": ""}),
                "模型类型": (CLIP_TYPES, {"default": "自动（按文件名判断）"}),
                "卸载模型": (UNLOAD_MODES, {"default": UNLOAD_AUTO,
                                        "tooltip": "自动=交给服务端；保持加载=跑完不卸；"
                                                   "运行后/运行前后自动卸载=编码完就把 CLIP 踢出服务端显存"}),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "可选：接「远程 CLIP 加载器」的 clip，用它带的地址/模型；不接就用本节点自己的字段"}),
                "device": (["auto", "cpu", "gpu"], {"default": "auto",
                                                   "tooltip": "条件张量放哪：auto/gpu 与原生一致；cpu 更省显存"}),
            },
        }
        return out

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("正向条件", "负向条件")
    FUNCTION = "encode"
    CATEGORY = CATEGORY
    DESCRIPTION = ("一个节点出正负两路条件。也可以不用它："
                   "「远程 CLIP 加载器」的输出直接接自带的 CLIPTextEncode 同样能跑。")

    @classmethod
    def VALIDATE_INPUTS(cls, 模型=None, 模型类型=None):
        if 模型 is None:
            return True
        return _validate_model(模型)

    def encode(self, 正向提示词, 负向提示词, 服务端显卡, 服务端地址, 模型, 超时时间, 手动模型名,
               模型类型="自动（按文件名判断）", **kw):
        clip = kw.get("clip")
        token = _cfg_token()
        device = kw.get("device", "auto") or "auto"

        def to_dev(conds):
            return [[_to_device(c, device), {k: _to_device(v, device) for k, v in d.items()}]
                    for c, d in conds]

        if clip is not None:
            # 输入端「clip」连了节点：本节点自己的模型字段（地址/模型/手动模型名/模型类型）全部让位，
            # 一律用连过来的这个 clip 编码。
            proxy = clip if isinstance(clip, RemoteCLIP) else None
            handle = _handle_of(clip)
            if proxy is None and handle is None:
                # 本机原生 CLIP（如自带的 CLIPLoader / DualCLIPLoader）：本机直接编码
                print("[远程CLIP编码] 用输入端口连过来的本地 clip 编码（忽略本节点的模型设置）")
                return (to_dev(clip.encode_from_tokens_scheduled(clip.tokenize(正向提示词 or ""))),
                        to_dev(clip.encode_from_tokens_scheduled(clip.tokenize(负向提示词 or ""))))
            if proxy is not None:
                url, tok, name = proxy.server_url, proxy.token, proxy.model
                ctype, timeout = proxy.clip_type, float(超时时间 or proxy.timeout)
            else:
                url, tok = handle.get("url") or 服务端地址, handle.get("token") or token
                name = handle.get("model") or _pick_model(模型, 手动模型名, "clip")
                ctype, timeout = handle.get("clip_type") or _clip_type_value(模型类型), float(超时时间 or 120)
        else:
            url, tok = 服务端地址, token
            name = _pick_model(模型, 手动模型名, "clip")
            ctype, timeout = _clip_type_value(模型类型), float(超时时间 or 120)

        _check_url(url)
        clip_proxy = RemoteCLIP(url, tok, name, ctype, timeout=timeout)

        before_u, after_u = _unload_plan(kw.get("卸载模型"))
        if before_u:
            _unload_remote(url, tok, "clip", name, "文本编码(运行前)")

        def one(text: str):
            return clip_proxy.encode_from_tokens_scheduled(clip_proxy.tokenize(text or ""))

        try:
            pos = to_dev(one(正向提示词))
            neg = to_dev(one(负向提示词))
        finally:
            if after_u:
                _unload_remote(url, tok, "clip", name, "文本编码(运行后)")
        return (pos, neg)


# --------------------------------------------------------------------------- #
# 3b. 远程 CLIP 图文编码（文本 + 参考图，图片接口自动增槽）
# --------------------------------------------------------------------------- #
class RemoteCLIPImageEncode:
    """和「远程 CLIP 文本编码」一样的正负两路输出，但多了参考图输入。

    - 图片接口 `图片1…图片10`：前端接一张就自动多出一个空槽（见 web/refresh.js）
    - 接了 `vae`：服务端按 Qwen-Image-Edit 的做法，图缩到「参考图像素」后一边进文本编码器
      vision 槽、一边编成 reference_latents 挂到条件上（keep_vision=False）
    - 没接 `vae`：只把图作为 vision 输入（keep_vision=True）
    """

    @classmethod
    def INPUT_TYPES(cls):
        cfg = default_server()
        models, model_opt = _combo("clip", cfg["url"], cfg["token"])
        required = {
            "正向提示词": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
            "负向提示词": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
            "服务端显卡": ("STRING", {"default": GPU_HINT}),
            "服务端地址": ("STRING", {"default": cfg["url"]}),
            "模型": (models, {"default": model_opt["default"]}),
            "超时时间": ("INT", {"default": 120, "min": 1, "max": 7200, "step": 1}),
            "手动模型名": ("STRING", {"default": ""}),
            "模型类型": (CLIP_TYPES, {"default": "自动（按文件名判断）"}),
            "参考图像素": ("INT", {"default": 1048576, "min": 0, "max": 4194304, "step": 1024,
                              "tooltip": "参考图缩到的总像素（1048576 = 1024×1024）；0 = 保持原尺寸（按 32 取整）"}),
            "卸载模型": (UNLOAD_MODES, {"default": UNLOAD_AUTO,
                                    "tooltip": "自动=交给服务端；保持加载=跑完不卸；"
                                               "运行后/运行前后自动卸载=编码完就把 CLIP 踢出服务端显存"}),
        }
        optional = {
            "clip": ("CLIP", {"tooltip": "可选：接「远程 CLIP 加载器」的 clip"}),
            "vae": ("VAE", {"tooltip": "可选：接「远程 VAE 加载器」的 vae（接了就同时编 reference_latents）"}),
            "device": (["auto", "cpu", "gpu"], {"default": "auto"}),
        }
        for i in range(1, IMAGE_SLOTS + 1):
            optional[f"图片{i}"] = ("IMAGE", {"tooltip": "参考图；接一张会自动多出一个空接口"})
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("正向条件", "负向条件")
    FUNCTION = "encode"
    CATEGORY = CATEGORY
    DESCRIPTION = ("远程 Qwen-VL 图文混编：文字 + 参考图（可选 VAE latent）一次出正负两路条件。"
                   "图片接一张自动多一个接口，最多 10 张。")

    @classmethod
    def VALIDATE_INPUTS(cls, 模型=None, 模型类型=None):
        if 模型 is None:
            return True
        return _validate_model(模型)

    def encode(self, 正向提示词, 负向提示词, 服务端显卡, 服务端地址, 模型, 超时时间, 手动模型名,
               模型类型="自动（按文件名判断）", 参考图像素=1048576, **kw):
        clip = kw.get("clip")
        vae = kw.get("vae")
        token = _cfg_token()
        device = kw.get("device", "auto") or "auto"
        imgs = _collect_images(kw)

        def to_dev(conds):
            return [[_to_device(c, device), {k: _to_device(v, device) for k, v in d.items()}]
                    for c, d in conds]

        proxy = clip if isinstance(clip, RemoteCLIP) else None
        vproxy = vae if isinstance(vae, RemoteVAE) else None
        handle = _handle_of(clip)
        vhandle = _handle_of(vae) if not vproxy else None

        if clip is not None and proxy is None and handle is None:
            # 输入端「clip」连了本机原生 CLIP：本节点自己的模型字段全部让位，用它来编
            keep = vae is None
            px = int(参考图像素 or 0)

            def one_local(text: str):
                toks = None
                if imgs:
                    try:
                        toks = clip.tokenize(text or "", images=list(imgs),
                                             keep_vision=keep, prevent_empty_text=True)
                    except TypeError:
                        toks = None
                if toks is None:
                    toks = clip.tokenize(text or "")
                return clip.encode_from_tokens_scheduled(toks)

            print(f"[远程CLIP图文编码] 用输入端口连过来的本地 clip 编码（忽略本节点的模型设置）| "
                  f"图 {len(imgs)} 张 | keep_vision={keep} | 参考图像素={px or '原尺寸'}")
            return (to_dev(one_local(正向提示词)), to_dev(one_local(负向提示词)))

        if proxy is not None:
            url, tok, name = proxy.server_url, proxy.token, proxy.model
            ctype, timeout = proxy.clip_type, float(超时时间 or proxy.timeout)
        elif vproxy is not None and not (服务端地址 or "").strip():
            url, tok = vproxy.server_url, vproxy.token
            name = _pick_model(模型, 手动模型名, "clip")
            ctype, timeout = _clip_type_value(模型类型), float(超时时间 or 120)
        elif handle is not None:
            url, tok = handle.get("url") or 服务端地址, handle.get("token") or token
            name = handle.get("model") or _pick_model(模型, 手动模型名, "clip")
            ctype, timeout = handle.get("clip_type") or _clip_type_value(模型类型), float(超时时间 or 120)
        else:
            url, tok = 服务端地址, token
            name = _pick_model(模型, 手动模型名, "clip")
            ctype, timeout = _clip_type_value(模型类型), float(超时时间 or 120)

        _check_url(url)
        clip_proxy = RemoteCLIP(url, tok, name, ctype, timeout=timeout)

        vae_model = ""
        if vproxy is not None:
            if vproxy.server_url.rstrip("/") != url.rstrip("/"):
                raise ValueError("「vae」与「clip」指向不同服务端，请用同一台服务端的 CLIP / VAE 加载器")
            vae_model = vproxy.model
        elif vhandle is not None:
            vae_model = vhandle.get("model") or ""

        if not imgs:
            def one(text: str):
                return clip_proxy.encode_from_tokens_scheduled(clip_proxy.tokenize(text or ""))
        else:
            px = int(参考图像素 or 0)

            def one(text: str):
                return clip_proxy.encode_with_images(text or "", imgs, vision_pixels=px, vae_model=vae_model)

        before_u, after_u = _unload_plan(kw.get("卸载模型"))
        if before_u:
            _unload_remote(url, tok, "clip", name, "图文编码(运行前)")
        try:
            return (to_dev(one(正向提示词)), to_dev(one(负向提示词)))
        finally:
            if after_u:
                _unload_remote(url, tok, "clip", name, "图文编码(运行后)")


# --------------------------------------------------------------------------- #
# 3c. 远程提示词优化（Qwen-Image 2.1 风格）：把 GGUF 模型放到远端 LLM 服务上
# --------------------------------------------------------------------------- #
SYS_T2I = (
    "你是 Qwen-Image 2.1 的提示词工程师。根据用户需求写一条可直接用于 Qwen-Image 2.1 文生图的提示词：\n"
    "1) 用自然语言的连贯描述，覆盖主体、动作、外观细节、服装材质、环境、光线、镜头与画风；\n"
    "2) 不要堆砌逗号关键词，不要出现「杰作/8K/最佳画质」这类空话；\n"
    "3) 画面里不要出现文字、水印；\n"
    "4) 若用户要求透明背景，主体描述之外明确说明背景完全透明（保留 alpha）。\n"
    '只输出严格 JSON：{"prompt": "…", "negative_prompt": "…"}\n'
)
SYS_I2I = (
    "你是 Qwen-Image 2.1 图像编辑模型的提示词工程师。用户给了参考图和修改要求，"
    "你要把要求改写成一条清晰的编辑指令：\n"
    "1) 明确「保留什么」（人物长相、姿势、构图、背景）与「改什么」（服装、材质、颜色、道具、光线）；\n"
    "2) 只描述最终画面，不要解释过程，不要出现「图一/图二」之外的指代混淆；\n"
    "3) 不要写美术风格清单式的空话。\n"
    '只输出严格 JSON：{"prompt": "…", "negative_prompt": "…"}\n'
)
SYS_REV = (
    "你是专业图像描述员。看参考图，反推出一条可直接用于 Qwen-Image 2.1 的提示词，"
    "尽量完整覆盖：主体与数量、外貌与表情、姿势与视角、服装与材质、道具、环境与背景、光线与色调、镜头与画风。\n"
    "不要猜测无法看到的内容，不要写成标签堆叠。\n"
    '只输出严格 JSON：{"prompt": "…", "negative_prompt": "…"}\n'
)

MODE_AUTO = "自动（有图无词=反推 / 有图有词=图生图 / 无图=文生图）"
IF_REMOTE = "RemoteCLIPVAE 服务端 (用远端 ComfyUI 里的 GGUF)"
IF_OPENAI = "OpenAI 兼容 (llama.cpp / LM Studio / vLLM)"
IF_OLLAMA = "Ollama 原生 (/api/chat)"
IFACES = [IF_REMOTE, IF_OPENAI, IF_OLLAMA]
NO_MMPROJ = "（不使用 / 纯文本）"
SLOT_EVERY = "自动"
UNLOAD_AUTO = "自动"
UNLOAD_KEEP = "保持加载"
UNLOAD_AFTER = "运行后自动卸载"
UNLOAD_BOTH = "运行前后自动卸载"
UNLOAD_MODES = [UNLOAD_AUTO, UNLOAD_KEEP, UNLOAD_AFTER, UNLOAD_BOTH]
SEED_MODES = ["随机", "固定"]
DEFAULT_CTX = 4096
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMP = 0.7
DEFAULT_TIMEOUT = 600.0


def _opt_json_of(text: str):
    """从模型返回里抠出 JSON（容忍 ```json 包裹、前后废话、思考过程里出现的「格式示例」）。

    关键点：思考过程里常写 `{"prompt": "...", "negative_prompt": "..."}` 这种示例，
    所以不能拿第一个 JSON —— 取【最后一个】真正有内容（prompt 不是 "..."）的对象。
    """
    if not text:
        return None
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and _opt_prompt_of(obj):
            return obj
    except Exception:
        pass
    cands = []
    depth, start = 0, -1
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    o = json.loads(raw[start:i + 1])
                    if isinstance(o, dict):
                        cands.append(o)
                except Exception:
                    pass
                start = -1
    good = [o for o in cands if _opt_prompt_of(o) and len(_opt_prompt_of(o)) > 2]
    return good[-1] if good else (cands[-1] if cands else None)


def _opt_prompt_of(obj: dict) -> str:
    v = obj.get("prompt") or obj.get("positive_prompt") or obj.get("positive") or ""
    v = str(v or "").strip()
    if v in ("...", "…", "…(...)", "（略）", ""):
        return ""
    return v


def _strip_thinking(text: str) -> str:
    """去掉模型在正式回答前的思考段（Thinking Process / <think>…</think> / 推理过程）。"""
    t = (text or "").strip()
    if not t:
        return t
    if "<think" in t:
        # 有闭合标签就取闭合之后；只有开标签（被 max_tokens 截断）就等于没有正文
        t = t.split("</think>", 1)[1].strip() if "</think>" in t else ""
    if not t:
        return ""
    for mark in ("Thinking Process:", "Thinking process:", "推理过程:", "思考过程:"):
        if mark in t:
            t = t.split(mark, 1)[1].strip()
            parts = [p for p in t.split("\n\n") if p.strip()]
            if len(parts) > 1:
                t = parts[-1].strip()
    return t


class RemotePromptOptimizer:
    """把「Qwen image 2.1 提示词优化」的本地 GGUF 加载改成远端 LLM 服务。

    本机只发 HTTP：文本 + 参考图(Base64 PNG) → 远端 OpenAI 兼容接口（llama.cpp server /
    LM Studio / vLLM / Ollama 的 /v1/chat/completions）或 Ollama 原生 /api/chat。
    """

    @classmethod
    def INPUT_TYPES(cls):
        ds = default_server()
        llm_names, llm_err = _model_options("llm", ds["url"], ds["token"])
        if not llm_names:
            llm_names = [FAIL_HINT if llm_err else "（服务端没有 GGUF：先配 GGUF 模型目录）"]
        ds_v, _ = _model_options("llm_vision", ds["url"], ds["token"])
        # 三个模型下拉两边共用：服务端的 GGUF + 本机的 GGUF 都放进候选，
        # 名字各不冲突（本机的是相对路径），填哪个会不会被用只取决于「使用远程模型」开关
        local_names, local_visions = _local_gguf_scan()
        llm_all = list(dict.fromkeys(llm_names + local_names))
        # 占位提示（“连接失败 / 没有 GGUF”）排到最后，默认值优先给真模型
        real = [n for n in llm_all if not n.startswith("（") and not n.startswith("(")]
        hint = [n for n in llm_all if n not in real]
        llm_all = real + hint
        if not llm_all:
            llm_all = ["（没有可用 GGUF：服务端配目录 / 本机放 models/LLM）"]
        llm_default = real[0] if real else llm_all[0]
        vis_all = list(dict.fromkeys(ds_v + local_visions))
        return {
            "required": {
                "用户提示词": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
                "使用远程模型": ("BOOLEAN", {"default": True,
                                         "label_on": "远程（服务端 GGUF）",
                                         "label_off": "本地（本机 GGUF）",
                                         "tooltip": "开=用服务端那台机器的 GGUF（本机零依赖）；"
                                                   "关=用本机 models/LLM 里的 GGUF（本机要装 llama-cpp-python）"}),
                "服务端地址": ("STRING", {"default": ds["url"],
                                     "tooltip": "远程模式：RemoteCLIPVAE 服务端地址（8199）；模型是服务端 ComfyUI 里 models/LLM 的 GGUF"}),
                "API Key": ("STRING", {"default": "", "tooltip": "一般留空；服务端开了 X-Auth-Token 校验时填那个 token"}),
                "文生图模型": (llm_all, {"default": llm_default,
                                      "tooltip": "无参考图时用的 GGUF。开关开=服务端 models/LLM 里的；开关关=本机 models/LLM 里的"}),
                "图生图模型": (llm_all, {"default": llm_default,
                                      "tooltip": "接了参考图（图生图/反推）时用的 GGUF；配了「视觉模型」才能看图"}),
                "视觉模型": ([NO_MMPROJ] + vis_all, {"default": NO_MMPROJ,
                                              "tooltip": "mmproj 视觉投影；接图做反推/图生图时要配，纯文本留「不使用」"}),
                "输出语言": (["自动", "中文", "English"], {"default": "自动"}),
                "透明背景": ("BOOLEAN", {"default": False}),
                "种子值": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "种子模式": (SEED_MODES, {"default": "随机"}),
                "卸载模型": (UNLOAD_MODES, {"default": UNLOAD_AUTO,
                                        "tooltip": "远程=服务端 GGUF 的显存策略；本地=跑完是否把本机这份模型放掉。"
                                                   "自动/保持加载=一直留着（下次最快）"}),
            },
            "optional": dict({f"图片{i}": ("IMAGE", {"tooltip": "参考图；接一张自动多一个空接口"})
                              for i in range(1, IMAGE_SLOTS + 1)}),
        }

    RETURN_TYPES = ("STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("提示词", "负向提示词", "是否反推")
    FUNCTION = "optimize"
    CATEGORY = CATEGORY
    DESCRIPTION = ("「Qwen image 2.1 提示词优化」。开关打开=用服务端那台机器的 GGUF（本机零依赖）；"
                   "开关关闭=用本机 models/LLM 里的 GGUF（本机需装 llama-cpp-python）。"
                   "输出：提示词 + 负向提示词 + 是否反推。有图无词=反推 / 有图有词=图生图 / 无图=文生图。")

    # ---- 内部 ---------------------------------------------------------- #
    @classmethod
    def VALIDATE_INPUTS(cls, **kw):
        """启动时固化的是「默认地址扫到的 GGUF 列表」，节点里改过地址/点刷新后列表会变，
        队列校验不要拿旧白名单拦人（真正的错误在执行阶段用中文抛出来）。"""
        return True

    @staticmethod
    def _lang_rule(输出语言: str, user_text: str) -> str:
        lang = (输出语言 or "自动").strip()
        if lang == "中文":
            return "\nprompt 与 negative_prompt 必须用简体中文书写。"
        if lang == "English":
            return "\nprompt and negative_prompt must be written in English."
        if user_text and not re.search(r"[\u4e00-\u9fff]", user_text):
            return "\nWrite prompt and negative_prompt in English (following the user's language)."
        return "\nprompt 与 negative_prompt 用简体中文书写。"

    @staticmethod
    def _post_json(url: str, payload: dict, api_key: str, timeout: float) -> dict:
        import urllib.request
        import urllib.error

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:600]
            raise RuntimeError(f"远端 LLM 返回 HTTP {e.code}：{detail}") from None
        except Exception as e:
            raise RuntimeError(f"连不上远端 LLM 服务 {url}：{e}") from None

    # ---- 本地模式：本机 llama-cpp 直接跑 ---- #
    @staticmethod
    def _local_chat(model_path: str, mmproj: str, system: str, text: str, b64s: list,
                    seed: int, max_tokens: int, temperature: float, n_ctx: int):
        import base64
        import tempfile

        llm = _local_llm_get(model_path, mmproj, n_ctx)
        msgs = [{"role": "system", "content": system}]
        tmp_files = []
        if b64s and mmproj:
            parts = [{"type": "text", "text": text}]
            for b in b64s[:4]:
                try:
                    fd, p = tempfile.mkstemp(suffix=".png")
                    with os.fdopen(fd, "wb") as f:
                        f.write(base64.b64decode(b))
                    tmp_files.append(p)
                    parts.append({"type": "image_url", "image_url": {"url": "file:///" + p.replace("\\", "/")}})
                except Exception:
                    break
            msgs.append({"role": "user", "content": parts})
        else:
            msgs.append({"role": "user", "content": text})

        def call(extra: dict):
            return llm.create_chat_completion(messages=msgs, max_tokens=int(max_tokens),
                                               temperature=float(temperature), seed=int(seed), **extra)

        base_extra = {"response_format": {"type": "json_object"},
                      "chat_template_kwargs": {"enable_thinking": False}}
        resp = None
        for drop in (None, "chat_template_kwargs", "response_format"):
            if drop:
                base_extra.pop(drop, None)
            try:
                resp = call(dict(base_extra))
                break
            except TypeError:
                continue
            except ValueError as e:
                if "response_format" in str(e) or "grammar" in str(e):
                    continue
                raise
        for p in tmp_files:
            try:
                os.remove(p)
            except Exception:
                pass
        if resp is None:
            raise RuntimeError("本地 llama-cpp 调用失败（create_chat_completion 全部分支都报错）")
        content = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        usage = resp.get("usage") or {}
        return content, int(usage.get("completion_tokens") or 0)

    def optimize(self, 用户提示词, 使用远程模型=True, 服务端地址="", APIKey="", 文生图模型="",
                 图生图模型="", 视觉模型=NO_MMPROJ,
                 输出语言="自动", 透明背景=False, 种子值=0,
                 种子模式="随机", 卸载模型=UNLOAD_AUTO, **kw):
        api_key = str(kw.get("API Key", "") or APIKey or "")
        token = _cfg_token()
        manual = str(kw.get("自定义模型名") or "").strip()  # 老工作流里的旧字段，还有就兜底用
        use_remote = bool(使用远程模型)
        # 老工作流里还有这些控件：有就用，没有就走默认值
        iface = str(kw.get("接口类型") or IF_REMOTE)
        模式 = str(kw.get("模式") or MODE_AUTO)
        上下文长度 = int(kw.get("上下文长度") or DEFAULT_CTX)
        温度 = float(kw.get("温度", DEFAULT_TEMP) or DEFAULT_TEMP)
        最大输出长度 = int(kw.get("最大输出长度") or DEFAULT_MAX_TOKENS)
        超时时间 = float(kw.get("超时时间") or DEFAULT_TIMEOUT)
        images = _collect_images(kw)
        user_text = str(用户提示词 or "").strip()
        has_images = bool(images)
        reverse = (模式 == "反推") or ("自动" in 模式 and has_images and not user_text)
        if 模式 == "文生图":
            reverse = False

        if reverse:
            system = SYS_REV
        elif has_images:
            system = SYS_I2I
        else:
            system = SYS_T2I
        system = system + self._lang_rule(输出语言, user_text)
        # Qwen3 系会先吐「Thinking Process」，提示优化只要最终 JSON：直接压掉思考段
        system += ("\n不要输出任何思考过程/推理说明（如 Thinking Process、/no_think 提示），"
                   "第一行就是 {，最后一行就是 }。")
        if 透明背景 and not reverse:
            system += "\n用户要求透明背景：主体之外的环境描述要保证背景是全透明的（alpha=0）。"
        if reverse:
            user_text = "请反推这张图的提示词。" + ("背景要求透明。" if 透明背景 else "")

        seed = (random.randint(0, 0xFFFFFFFF) if 种子模式 == "随机"
                else int(种子值 or 0))

        # 参考图：IMAGE → 每帧一张 Base64 PNG
        b64s = []
        for img in images[:IMAGE_SLOTS]:
            for frame in tensor_to_png_frames(img):
                b64s.append(frame)

        unload_after = 卸载模型 in (UNLOAD_AFTER, UNLOAD_BOTH)
        unload_before = 卸载模型 == UNLOAD_BOTH

        if not use_remote:
            # ---- 本地模式：本机 models/LLM 的 GGUF，用本机 ComfyUI 的 python 直接跑 ----
            # 三栏和远程共用：无图用「文生图模型」，有图用「图生图模型」+「视觉模型」
            local_pick = str((图生图模型 if has_images else 文生图模型) or "").strip()
            if not local_pick or local_pick.startswith("（") or local_pick.startswith("("):
                local_pick = str(kw.get("本地模型") or "").strip()  # 老工作流里的旧字段
            lpath = _local_gguf_path(local_pick)
            if not lpath:
                raise ValueError(
                    f"本地模式找不到这个 GGUF：{local_pick or '(空)'}\n"
                    "先把 GGUF 放进本机 <ComfyUI>/models/LLM（也认 models/llm、models/gguf 等），"
                    "再在「文生图模型 / 图生图模型」里选一个；或者把开关「使用远程模型」打开走服务端。"
                )
            mm_src = str(视觉模型 or "").strip()
            if not mm_src or mm_src.startswith("（") or mm_src.startswith("("):
                mm_src = str(kw.get("本地视觉模型") or "").strip()  # 老工作流里的旧字段
            mm_local = "" if (not mm_src or mm_src.startswith("（") or mm_src.startswith("(")) else _local_gguf_path(mm_src)
            if b64s and not mm_local:
                raise ValueError("本地模式接了参考图，但没选「视觉模型」（mmproj）：本地模型看不了图。\n"
                                 "请把「视觉模型」选成本机对应的 mmproj，或把图片断开走纯文生图。")
            if unload_before:
                _local_llm_free()
            try:
                text, tokens = self._local_chat(lpath, mm_local, system, user_text, b64s, seed,
                                                最大输出长度, 温度, 上下文长度)
            finally:
                if unload_after:
                    _local_llm_free()
            print(f"[提示词优化-本地] model={os.path.basename(lpath)} | mmproj={os.path.basename(mm_local) or '无'} | "
                  f"图 {len(b64s)} 张 | {'反推' if reverse else ('图生图' if has_images else '文生图')} | "
                  f"seed={seed} | {tokens} tok", flush=True)
            data = {"text": text}
        else:
            url = (服务端地址 or "").strip().rstrip("/")
            if not url:
                raise ValueError("请填写服务端地址（或把开关「使用远程模型」关掉，改用本机 GGUF）")
            timeout = float(超时时间 or DEFAULT_TIMEOUT)
            pick = 图生图模型 if has_images else 文生图模型
            if manual:
                model_name = manual
            else:
                model_name = str(pick or kw.get("模型名") or "").strip()
            if not model_name or model_name.startswith("（") or model_name.startswith("("):
                raise ValueError("模型名还是占位符：先在服务端面板配好「GGUF 模型目录」、启动服务，"
                                 "再点一下节点上的「🔄 刷新模型列表」选一个 GGUF。")
            mmproj = "" if str(视觉模型 or "").startswith("（") else str(视觉模型 or "").strip()
            if b64s and not mmproj:
                raise ValueError("接了参考图但没有「视觉模型」（mmproj）：服务端只能用纯文本，看不了图。\n"
                                 "请把「视觉模型」选成对应的 mmproj（例如 mmproj-*.gguf），或把图片断开走纯文生图。")

            if unload_before:
                try:
                    remote_api.rpc(url, token, "llm_unload", {}, timeout=120)
                except Exception:
                    pass

            if iface.startswith("RemoteCLIPVAE"):
                # 走服务端 /rpc → 服务端用它自己 ComfyUI 的 python 加载 <ComfyUI>/models/LLM 里的 GGUF
                try:
                    data = remote_api.rpc(url, token, "llm_chat", {
                        "model": model_name, "mmproj": mmproj,
                        "system": system, "text": user_text, "images": b64s,
                        "max_tokens": int(最大输出长度), "temperature": float(温度),
                        "seed": int(seed), "n_ctx": int(上下文长度 or DEFAULT_CTX),
                        "timeout": timeout, "auto_load": True,
                    }, timeout=timeout + 120)
                except RuntimeError as e:
                    raise RuntimeError(f"{e}\n（远端 GGUF 路径：服务端 config.json 的 llm_model_dir，"
                                       f"留空就是 <ComfyUI>/models/LLM；服务端面板「提示词优化 LLM」里能看到模型清单与实时信息）") from None
                finally:
                    if unload_after:
                        try:
                            remote_api.rpc(url, token, "llm_unload", {}, timeout=120)
                        except Exception:
                            pass
                text = str((data or {}).get("text") or "")
                print(f"[远程提示词优化] RemoteCLIPVAE | model={model_name} | mmproj={mmproj or '无'} | 图 {len(b64s)} 张 | "
                      f"{'反推' if reverse else ('图生图' if has_images else '文生图')} | seed={seed} | "
                      f"{data.get('tokens', 0)} tok / {data.get('tps', 0)} tok/s", flush=True)
            elif iface.startswith("Ollama"):
                content = user_text
                msg = {"role": "user", "content": content}
                if b64s:
                    msg["images"] = b64s
                payload = {
                    "model": model_name,
                    "messages": [{"role": "system", "content": system}, msg],
                    "stream": False,
                    "options": {"temperature": float(温度), "seed": int(seed),
                                "num_predict": int(最大输出长度)},
                }
                data = self._post_json(url + "/api/chat", payload, api_key, timeout)
                text = ((data or {}).get("message") or {}).get("content") or ""
                print(f"[远程提示词优化] Ollama | model={model_name} | 图 {len(b64s)} 张 | seed={seed}", flush=True)
            else:
                parts = [{"type": "text", "text": user_text}]
                for b in b64s:
                    parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}})
                payload = {
                    "model": model_name,
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": parts}],
                    "temperature": float(温度),
                    "max_tokens": int(最大输出长度),
                    "seed": int(seed),
                    "stream": False,
                }
                data = self._post_json(url + "/v1/chat/completions", payload, api_key, timeout)
                try:
                    text = (((data or {}).get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                except Exception:
                    text = ""
                print(f"[远程提示词优化] OpenAI | model={model_name or '(默认)'} | 图 {len(b64s)} 张 | seed={seed}", flush=True)

        if not text.strip():
            raise RuntimeError(f"远端 LLM 没有返回内容：{json.dumps(data, ensure_ascii=False)[:400]}")

        # 思考段一律先剃掉（Qwen3 系会先吐 Thinking Process / <think>）
        text = _strip_thinking(text) if ("Thinking" in text or "思考" in text or "<think" in text) else text
        obj = _opt_json_of(text)
        if isinstance(obj, dict):
            prompt = str(obj.get("prompt") or obj.get("positive_prompt") or obj.get("positive") or "").strip()
            neg = str(obj.get("negative_prompt") or obj.get("negative") or "").strip()
            out_json = json.dumps(obj, ensure_ascii=False, indent=2)
        else:
            clean = text.strip()
            obj2 = _opt_json_of(clean)
            if isinstance(obj2, dict):
                prompt = str(obj2.get("prompt") or obj2.get("positive_prompt") or "").strip()
                neg = str(obj2.get("negative_prompt") or "").strip()
                out_json = json.dumps(obj2, ensure_ascii=False, indent=2)
            else:
                prompt, neg, out_json = clean, "", clean

        if not prompt.strip():
            raise RuntimeError(
                "模型把 token 全用在「思考」上了，没留下正文。\n"
                "换一个不带思考、能直接出 JSON 的 GGUF（或调小「种子值」重跑一次）。"
            )

        if 透明背景 and not reverse and prompt and "透明" not in prompt:
            prompt += "\n背景全透明（alpha=0），不出现地面、墙面与投影。"
        neg = neg.strip()
        return (prompt.strip(), neg, bool(reverse))


# --------------------------------------------------------------------------- #
# 4. 远程 VAE 解码（隐空间 → 图像）
# --------------------------------------------------------------------------- #
class RemoteVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {"tooltip": "接「远程 VAE 加载器」的 vae（类型就是 VAE）"}),
                "latent": ("LATENT", {"tooltip": "接 KSampler 的 LATENT"}),
                "超时时间": ("INT", {"default": 600, "min": 10, "max": 7200, "step": 10}),
                "卸载模型": (UNLOAD_MODES, {"default": UNLOAD_AUTO,
                                        "tooltip": "自动=交给服务端；保持加载=跑完不卸；"
                                                   "运行后/运行前后自动卸载=解码完把 VAE 踢出服务端显存"}),
            },
            "optional": {
                "分块解码": ("BOOLEAN", {"default": False, "tooltip": "大图在服务端显存吃紧时开启"}),
                "分块尺寸": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 64}),
                "重叠": ("INT", {"default": 64, "min": 0, "max": 256, "step": 8}),
                "输出设备": (["cpu", "gpu", "auto"], {"default": "cpu",
                                                     "tooltip": "解出的图像放哪；cpu 最省本机显存"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "decode"
    CATEGORY = CATEGORY
    DESCRIPTION = ("和自带的 VAEDecode 等价，只是把解码放到服务端。"
                   "也可以直接用自带的 VAEDecode + 远程 VAE 加载器。")

    def decode(self, vae, latent, 超时时间, 卸载模型=UNLOAD_AUTO, 分块解码=False, 分块尺寸=512, 重叠=64, 输出设备="cpu"):
        if not isinstance(latent, dict) or "samples" not in latent:
            raise ValueError("latent 需要是 LATENT（含 samples 张量）")
        samples = latent["samples"]
        if isinstance(vae, RemoteVAE):
            vae.timeout = float(超时时间 or vae.timeout)
            before_u, after_u = _unload_plan(卸载模型)
            if before_u:
                _unload_remote(vae.server_url, vae.token, "vae", vae.model, "VAE 解码(运行前)")
            try:
                image = vae.decode_tiled(samples, tile_x=分块尺寸, overlap=重叠) if 分块解码 else vae.decode(samples)
            finally:
                if after_u:
                    _unload_remote(vae.server_url, vae.token, "vae", vae.model, "VAE 解码(运行后)")
            return (_to_device(image, 输出设备),)
        handle = _handle_of(vae)
        if handle is None:
            raise ValueError("vae 输入无效，请接「远程 VAE 加载器」的输出")
        before_u, after_u = _unload_plan(卸载模型)
        url, tok, name = handle["url"], handle.get("token", ""), handle["model"]
        if before_u:
            _unload_remote(url, tok, "vae", name, "VAE 解码(运行前)")
        try:
            data = remote_api.rpc(
                url, tok, "vae_decode",
                {"model": name, "latent": samples.detach().float().cpu(),
                 "tiled": bool(分块解码), "tile_size": int(分块尺寸), "overlap": int(重叠)},
                timeout=float(超时时间 or 600),
            )
            image = png_frames_to_tensor(data["frames"], channels=int(data.get("channels") or 3))
        finally:
            if after_u:
                _unload_remote(url, tok, "vae", name, "VAE 解码(运行后)")
        return (_to_device(image, 输出设备),)


# --------------------------------------------------------------------------- #
# 5. 远程 VAE 编码（图像 → 隐空间）
# --------------------------------------------------------------------------- #
class RemoteVAEEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE", {"tooltip": "接「远程 VAE 加载器」的 vae（类型就是 VAE）"}),
                "图像": ("IMAGE", {"tooltip": "例如 LoadImage 的输出"}),
                "超时时间": ("INT", {"default": 600, "min": 10, "max": 7200, "step": 10}),
                "卸载模型": (UNLOAD_MODES, {"default": UNLOAD_AUTO,
                                        "tooltip": "自动=交给服务端；保持加载=跑完不卸；"
                                                   "运行后/运行前后自动卸载=编码完把 VAE 踢出服务端显存"}),
            },
            "optional": {
                "分块编码": ("BOOLEAN", {"default": False}),
                "分块尺寸": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 64}),
                "重叠": ("INT", {"default": 64, "min": 0, "max": 256, "step": 8}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = CATEGORY
    DESCRIPTION = ("和自带的 VAEEncode 等价，编码在服务端做。"
                   "也可以直接用自带的 VAEEncode + 远程 VAE 加载器。")

    def encode(self, vae, 图像, 超时时间, 卸载模型=UNLOAD_AUTO, 分块编码=False, 分块尺寸=512, 重叠=64):
        if not torch.is_tensor(图像):
            raise ValueError("图像需要是 IMAGE 张量")
        if isinstance(vae, RemoteVAE):
            vae.timeout = float(超时时间 or vae.timeout)
            before_u, after_u = _unload_plan(卸载模型)
            if before_u:
                _unload_remote(vae.server_url, vae.token, "vae", vae.model, "VAE 编码(运行前)")
            try:
                latent = vae.encode_tiled(图像, tile_x=分块尺寸, overlap=重叠) if 分块编码 else vae.encode(图像)
            finally:
                if after_u:
                    _unload_remote(vae.server_url, vae.token, "vae", vae.model, "VAE 编码(运行后)")
            return ({"samples": latent},)
        handle = _handle_of(vae)
        if handle is None:
            raise ValueError("vae 输入无效，请接「远程 VAE 加载器」的输出")
        before_u, after_u = _unload_plan(卸载模型)
        url, tok, name = handle["url"], handle.get("token", ""), handle["model"]
        if before_u:
            _unload_remote(url, tok, "vae", name, "VAE 编码(运行前)")
        try:
            data = remote_api.rpc(
                url, tok, "vae_encode",
                {"model": name, "frames": tensor_to_png_frames(图像),
                 "tiled": bool(分块编码), "tile_size": int(分块尺寸), "overlap": int(重叠)},
                timeout=float(超时时间 or 600),
            )
            latent = data["samples"]
        finally:
            if after_u:
                _unload_remote(url, tok, "vae", name, "VAE 编码(运行后)")
        return ({"samples": latent if torch.is_tensor(latent) else torch.as_tensor(latent)},)


NODE_CLASS_MAPPINGS = {
    "RemoteCLIPLoader": RemoteCLIPLoader,
    "RemoteVAELoader": RemoteVAELoader,
    "RemoteCLIPTextEncode": RemoteCLIPTextEncode,
    "RemoteCLIPImageEncode": RemoteCLIPImageEncode,
    "RemotePromptOptimizer": RemotePromptOptimizer,
    "RemoteVAEDecode": RemoteVAEDecode,
    "RemoteVAEEncode": RemoteVAEEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoteCLIPLoader": "远程 CLIP 加载器 (Remote CLIP Loader)",
    "RemoteVAELoader": "远程 VAE 加载器 (Remote VAE)",
    "RemoteCLIPTextEncode": "远程 CLIP 文本编码 (Remote CLIP Encode)",
    "RemoteCLIPImageEncode": "远程 CLIP 图文编码 (Remote CLIP Encode + Images)",
    "RemotePromptOptimizer": "提示词优化 (Prompt Optimizer, Remote)",
    "RemoteVAEDecode": "远程 VAE 解码 (Remote VAE Decode)",
    "RemoteVAEEncode": "远程 VAE 编码 (Remote VAE Encode)",
}
