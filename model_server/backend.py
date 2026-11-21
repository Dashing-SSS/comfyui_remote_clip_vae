# -*- coding: utf-8 -*-
"""
backend.py —— B 电脑上的模型后端。

职责：
  1. 复用 B 电脑上已安装的 ComfyUI 代码库（comfy.sd / folder_paths），
     这样 CLIP / VAE 的加载与编码方式和 ComfyUI 官方节点完全一致（不会出现
     "独立实现导致 conditioning 数值对不上本机 UNet" 的坑）。
  2. 惰性加载 + LRU 淘汰（默认各只保留 1 个），显存/内存不会被模型堆满。
  3. 提供 3 个原子操作：clip_encode / vae_decode / vae_encode。

导入 comfy 前必须清空 sys.argv：comfy.cli_args 在 import 时就 parse_args()，
我们自己的命令行参数会让它报错 —— 这是很多人搭独立服务端时踩的坑。
"""

from __future__ import annotations

import gc
import math
import os
import sys
import threading
import time
import traceback
from collections import OrderedDict

import torch

import codec
from llm import LLMEngine


class ServerError(Exception):
    """可预期的业务错误，会原样回传给 ComfyUI 节点显示。"""


def _to_cpu(obj):
    """递归把容器里的张量搬到 CPU，避免把 GPU 显存句柄序列化出去。"""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


class Backend:
    def __init__(self, cfg: dict, config_dir: str):
        self.cfg = cfg or {}
        self.config_dir = config_dir
        self.comfy_dir = os.path.abspath(os.path.expanduser(str(self.cfg.get("comfyui_dir") or "")))
        self.models_cfg = self.cfg.get("models") or {}

        max_loaded = self.cfg.get("max_loaded") or {}
        self.max_clip = int(max_loaded.get("clip", 1))
        self.max_vae = int(max_loaded.get("vae", 1))
        self.idle_unload = float(self.cfg.get("idle_unload_seconds", 600) or 0)
        # auto = 交给 ComfyUI 自己决定（有 CUDA 就用 CUDA）；cpu = 强制 CPU（服务端没显卡/显存太小时用）
        self.device = str(self.cfg.get("device") or "auto").strip().lower()

        self._lock = threading.RLock()
        self._loaded = {"clip": OrderedDict(), "vae": OrderedDict(), "clip_vision": OrderedDict()}
        self.max_cv = 1
        # 客户端节点上手动指定的 clip_type（覆盖 config.json 里的默认值）；空 = 不覆盖
        self._clip_override: dict[str, str] = {}
        self._clip_applied: dict[str, str] = {}
        self._last_active = time.time()
        self._comfy = None
        self._models_cache = (0.0, None)
        # 远端 ComfyUI 里的 GGUF（提示词优化节点用）：跑在服务端 ComfyUI 的 python 里。
        # 「外部 LLM 服务地址（llm_external）」这套 v1.7.27 就从面板下线了：留着它会让 GGUF 一个都扫不到，
        # 所以配置里残留的 llm_external 一律忽略（不再走 ExternalLLM）。
        if str(self.cfg.get("llm_external") or "").strip():
            print("[服务端] 已忽略 config.json 里残留的 llm_external —— "
                  "现在 GGUF 由本进程自己的 llama-cpp-python 加载，不再转发到外部服务", flush=True)
            self.cfg.pop("llm_external", None)
        self.llm = LLMEngine(self)

        # 模型根目录（可多个）：直接读这些 models 目录里的 vae / clip / text_encoders / clip_vision / LLM，
        # 按文件夹名自动归类，不用再一个个填「CLIP 目录 / VAE 目录 / GGUF 目录」
        self.model_roots = self._norm_roots(self.cfg.get("model_roots"))
        self._root_scan_info = []
        try:
            self._inject_root_models()
        except Exception as e:
            print(f"[服务端] 扫描模型根目录失败：{e}", flush=True)

        if self.idle_unload > 0:
            t = threading.Thread(target=self._idle_loop, name="idle-unloader", daemon=True)
            t.start()

    # ------------------------------------------------------------------ #
    # 初始化 / ComfyUI 导入
    # ------------------------------------------------------------------ #
    def _import_comfy(self):
        if self._comfy is not None:
            return self._comfy
        if not self.comfy_dir or not os.path.isdir(self.comfy_dir):
            raise ServerError(
                f"comfyui_dir 配置无效：{self.comfy_dir!r}。请在 config.json 里写成 B 电脑上 "
                f"ComfyUI 主目录的绝对路径，例如 D:/ComfyUI 或 /home/me/ComfyUI"
            )
        if self.comfy_dir not in sys.path:
            sys.path.insert(0, self.comfy_dir)

        saved_argv = sys.argv[:]
        sys.argv = [sys.argv[0]]  # comfy.cli_args 会 parse_args()，必须屏蔽我们的参数
        try:
            import comfy.sd
            import comfy.utils
            import comfy.clip_vision
            import comfy.model_management as mm
            import folder_paths
        except Exception as e:
            raise ServerError(
                f"无法从 {self.comfy_dir} 导入 ComfyUI 的 comfy 模块：{e}\n"
                f"请确认 comfyui_dir 指向的是 ComfyUI 源码根目录（里面有 comfy/ 文件夹和 main.py）"
            ) from e
        finally:
            sys.argv = saved_argv

        self._comfy = {"sd": comfy.sd, "utils": comfy.utils, "mm": mm,
                       "folder_paths": folder_paths, "cv": comfy.clip_vision}
        if self.device == "cpu":
            # ComfyUI 自己的 CPU 开关：设了之后所有 load_models_gpu / get_torch_device 都走 CPU，
            # 完全不建 CUDA context（服务端没显卡、或想给别的程序腾显存时用）。
            try:
                mm.cpu_state = mm.CPUState.CPU
                print("[服务端] 计算设备：CPU（已强制关闭 CUDA 路径）", flush=True)
            except Exception as e:
                print(f"[服务端] 警告：强制 CPU 失败：{e}", flush=True)
        return self._comfy

    # ------------------------------------------------------------------ #
    # 模型清单 / 路径解析
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # 重读配置（界面扫描/改目录后不用重启服务）
    # ------------------------------------------------------------------ #
    def reload(self, cfg: dict) -> dict:
        cfg = cfg or {}
        with self._lock:
            self.cfg = cfg
            cd = str(cfg.get("comfyui_dir") or "").strip()
            if cd:
                self.comfy_dir = os.path.abspath(os.path.expanduser(cd))
            self.models_cfg = cfg.get("models") or {}
            ml = cfg.get("max_loaded") or {}
            if ml.get("clip") is not None:
                self.max_clip = int(ml.get("clip", self.max_clip))
            if ml.get("vae") is not None:
                self.max_vae = int(ml.get("vae", self.max_vae))
            if cfg.get("idle_unload_seconds") is not None:
                self.idle_unload = float(cfg.get("idle_unload_seconds") or 0)
            self._models_cache = (0.0, None)          # 目录改了，模型清单缓存作废
            self.model_roots = self._norm_roots(cfg.get("model_roots"))
            try:
                self._inject_root_models()             # 重新扫一遍模型根目录
            except Exception as e:
                print(f"[服务端] 重扫模型根目录失败：{e}", flush=True)
            try:
                self.llm._models_cache = None      # GGUF 清单也一起作废
            except Exception:
                pass
        return {"comfyui_dir": self.comfy_dir,
                "model_roots": self.model_roots,
                "scanned": len(self._root_scan_info),
                "models": {k: len((self.models_cfg.get(k) or {})) for k in ("clip", "vae", "clip_vision")}}

    # ------------------------------------------------------------------ #
    # 模型根目录 → 自动归类
    # ------------------------------------------------------------------ #
    @staticmethod
    def _norm_roots(v) -> list[str]:
        if isinstance(v, str):
            v = [v]
        out, seen = [], set()
        for r in v or []:
            r = str(r or "").strip().strip('"').strip("'")
            if not r:
                continue
            p = os.path.abspath(os.path.expanduser(r))
            if os.path.isdir(p) and os.path.normcase(p) not in seen:
                seen.add(os.path.normcase(p))
                out.append(p)
        return out

    def _inject_root_models(self):
        """扫 model_roots 里的 vae / clip / text_encoders / clip_vision 文件夹，把文件变成模型条目。
        GGUF 不在这里登记：由 LLMEngine 直接从这些根目录下的 LLM 文件夹扫（见 llm._model_roots）。"""
        import config as cfgmod
        self._root_scan_info = []
        if not self.model_roots:
            return
        self.models_cfg.setdefault("clip", {})
        self.models_cfg.setdefault("vae", {})
        self.models_cfg.setdefault("clip_vision", {})

        # 已经登记过的（含手写条目）按「绝对路径」去重，避免同一个文件出现两份
        known = set()
        for kind in ("clip", "vae", "clip_vision"):
            for entry in (self.models_cfg.get(kind) or {}).values():
                try:
                    files = self._entry_files(entry)
                except Exception:
                    continue
                for f in files:
                    known.add(os.path.normcase(self._resolve(f)))
        counts = {"clip": 0, "vae": 0, "clip_vision": 0, "llm": 0, "llm_vision": 0}
        for it in cfgmod.scan_roots(self.model_roots):
            kind = it["kind"]
            counts[kind] = counts.get(kind, 0) + 1
            if kind not in ("clip", "vae", "clip_vision"):
                continue
            key_path = os.path.normcase(it["path"])
            if key_path in known:
                continue
            key = cfgmod.entry_name(it["rel"])
            entries = self.models_cfg[kind]
            if key in entries and os.path.normcase(str(
                    (entries[key].get("files") or [""])[0] if isinstance(entries[key], dict) else entries[key])) != key_path:
                i = 2
                while f"{key}_{i}" in entries:
                    i += 1
                key = f"{key}_{i}"
            entries[key] = {"files": [it["path"]], "note": f"目录扫描（{os.path.basename(os.path.dirname(it['root']))}/models）"}
            known.add(key_path)
            self._root_scan_info.append({"kind": kind, "name": key, "path": it["path"]})
        print(f"[服务端] 模型根目录 {len(self.model_roots)} 个 → 自动登记 CLIP {counts['clip']}、VAE {counts['vae']}、"
              f"clip_vision {counts['clip_vision']}、GGUF {counts['llm']}、mmproj {counts['llm_vision']}", flush=True)

    def list_models(self, ttl: float = 4.0) -> dict:
        hit = getattr(self, "_models_cache", None)
        if hit and hit[1] is not None and time.time() - hit[0] < ttl:
            return hit[1]
        out = {"clip": {}, "vae": {}, "clip_vision": {}}
        for kind in ("clip", "vae"):
            for name, entry in (self.models_cfg.get(kind) or {}).items():
                try:
                    files = self._entry_files(entry)
                except ServerError:
                    out[kind][name] = {"files": [], "exists": False, "note": "配置格式错误"}
                    continue
                out[kind][name] = {
                    "files": files,
                    "resolved": [self._resolve(f) for f in files],
                    "exists": all(os.path.exists(self._resolve(f)) for f in files),
                    "clip_type": (entry or {}).get("clip_type", "stable_diffusion") if kind == "clip" and isinstance(entry, dict) else None,
                    "note": (entry or {}).get("note", "") if isinstance(entry, dict) else "",
                }
        # 视觉编码器（clip_vision）：config.json 里写了的条目 + 目录里现扫到的文件，都列出来给节点下拉
        for name, entry in (self.models_cfg.get("clip_vision") or {}).items():
            try:
                files = self._entry_files(entry)
            except ServerError:
                continue
            out["clip_vision"][name] = {"files": files, "exists": True, "note": "config 条目"}
        for fn in self.list_clip_vision_files():
            key = os.path.basename(fn)
            if key not in out["clip_vision"]:
                out["clip_vision"][key] = {"files": [fn], "exists": True, "note": "目录扫描"}
        # GGUF（给「提示词优化」节点用）：只扫服务端 ComfyUI 的 models/LLM 一类目录
        try:
            gg = self.llm.list_models()
            out.update({"llm": gg.get("llm") or {}, "llm_vision": gg.get("llm_vision") or {}})
        except Exception as e:
            out.update({"llm": {}, "llm_vision": {}, "llm_error": str(e)})
        self._models_cache = (time.time(), out)
        return out

    def clip_vision_dir(self) -> str:
        d = str(self.cfg.get("clip_vision_model_dir") or "").strip()
        if d:
            return os.path.abspath(os.path.expanduser(d))
        return os.path.join(self.comfy_dir, "models", "clip_vision")

    def list_clip_vision_files(self) -> list[str]:
        """扫描 clip_vision 目录里的视觉编码器文件（返回相对该目录的路径）。"""
        root = self.clip_vision_dir()
        if not os.path.isdir(root):
            return []
        exts = (".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin")
        found = []
        for dirpath, _dirs, files in os.walk(root):
            if dirpath[len(root):].count(os.sep) > 1:
                _dirs.clear()
                continue
            for fn in files:
                if fn.lower().endswith(exts) and not fn.startswith("put_"):
                    found.append(os.path.relpath(os.path.join(dirpath, fn), root).replace(os.sep, "/"))
        return sorted(found)

    @staticmethod
    def _entry_files(entry) -> list[str]:
        if isinstance(entry, str):
            return [entry]
        if isinstance(entry, dict):
            if entry.get("files"):
                return [str(x) for x in entry["files"]]
            if entry.get("file"):
                return [str(entry["file"])]
        raise ServerError(f"模型配置格式不对：{entry!r}（应为 {{'files': [...]}} 或字符串）")

    def _resolve(self, p: str) -> str:
        p = os.path.expanduser(str(p).strip())
        if os.path.isabs(p):
            return p
        bases = [os.path.join(self.comfy_dir, "models"), self.comfy_dir, self.config_dir]
        # 界面里填的 CLIP / VAE 模型目录（模型放在 ComfyUI 外面时靠它）
        for key in ("clip_model_dir", "vae_model_dir", "clip_vision_model_dir"):
            d = str(self.cfg.get(key) or "").strip()
            if d:
                bases.append(os.path.expanduser(d))
        for base in bases:
            cand = os.path.join(base, p)
            if os.path.exists(cand):
                return cand
        return os.path.join(self.comfy_dir, "models", p)

    def _model_entry(self, kind: str, name: str):
        entries = self.models_cfg.get(kind) or {}
        if name not in entries:
            raise ServerError(
                f"服务端 config.json 没有名为 {name!r} 的 {kind} 模型。已配置：{sorted(entries.keys())}"
            )
        return entries[name]

    # ------------------------------------------------------------------ #
    # 加载 / 缓存
    # ------------------------------------------------------------------ #
    def get_clip(self, name: str, clip_type: str = ""):
        with self._lock:
            cache = self._loaded["clip"]
            want = (clip_type or self._clip_override.get(name) or "").strip()
            if name in cache:
                applied = self._clip_applied.get(name, "")
                if not want or want == applied:
                    cache.move_to_end(name)
                    self._last_active = time.time()
                    return cache[name]
                # 客户端换了 clip_type：先卸掉旧的再按新类型重载
                print(f"[服务端] CLIP {name} 的 clip_type 由 {applied or 'config'} 改为 {want}，重新加载", flush=True)
                self._free(cache.pop(name))
            c = self._import_comfy()
            entry = self._model_entry("clip", name)
            files = [self._resolve(f) for f in self._entry_files(entry)]
            for f in files:
                if not os.path.exists(f):
                    raise ServerError(f"CLIP 文件不存在：{f}")
            cfg_type = entry.get("clip_type", "stable_diffusion") if isinstance(entry, dict) else "stable_diffusion"
            if not want and str(cfg_type).lower() in ("", "auto", "stable_diffusion"):
                # 没指定 / 占位默认值：按权重内容猜（Qwen-VL 这类文本编码器当 stable_diffusion 加载会用错 tokenizer）
                guess = self._detect_clip_type(files, name)
                if guess:
                    cfg_type = guess
                    print(f"[服务端] 自动识别 clip_type={guess}（{os.path.basename(files[0])} 看起来是 {guess} 文本编码器）",
                          flush=True)
            clip_type = self._clip_type(want or cfg_type)
            print(f"[服务端] 加载 CLIP {name}: {files} (clip_type={clip_type})", flush=True)
            t0 = time.time()
            try:
                clip = c["sd"].load_clip(
                    ckpt_paths=files,
                    embedding_directory=c["folder_paths"].get_folder_paths("embeddings"),
                    clip_type=clip_type,
                    model_options={},
                )
            except TypeError:  # 老版本签名
                clip = c["sd"].load_clip(files, c["folder_paths"].get_folder_paths("embeddings"), clip_type)
            print(f"[服务端] CLIP 加载完成，耗时 {time.time() - t0:.1f}s", flush=True)
            self._evict("clip", self.max_clip)
            cache[name] = clip
            self._clip_applied[name] = want
            return clip

    def get_vae(self, name: str):
        with self._lock:
            cache = self._loaded["vae"]
            if name in cache:
                cache.move_to_end(name)
                self._last_active = time.time()
                return cache[name]
            c = self._import_comfy()
            entry = self._model_entry("vae", name)
            files = [self._resolve(f) for f in self._entry_files(entry)]
            for f in files:
                if not os.path.exists(f):
                    raise ServerError(f"VAE 文件不存在：{f}")
            print(f"[服务端] 加载 VAE {name}: {files}", flush=True)
            sd, metadata = c["utils"].load_torch_file(files[0], return_metadata=True)
            try:
                try:
                    vae = c["sd"].VAE(sd=sd, metadata=metadata)
                except TypeError:
                    vae = c["sd"].VAE(sd)
                if hasattr(vae, "throw_exception_if_invalid"):
                    vae.throw_exception_if_invalid()
            except ServerError:
                raise
            except Exception as e:
                raise ServerError(
                    f"{files[0]} 里没有可识别的 VAE：{e}\n"
                    f"排查：① 该文件是独立 VAE 还是整包 checkpoint？整包 checkpoint 不一定带 VAE；"
                    f"② 请改用一个独立的 VAE 文件（如 models/vae/*.safetensors、ae.safetensors、sdxl_vae.safetensors）"
                ) from e
            self._evict("vae", self.max_vae)
            cache[name] = vae
            return vae

    # ------------------------------------------------------------------ #
    # 视觉编码器（clip_vision）：Redux / IPAdapter 那类图像特征
    # ------------------------------------------------------------------ #
    def _resolve_clip_vision(self, name: str) -> str:
        name = str(name or "").strip()
        if name:
            p = os.path.expanduser(name)
            if os.path.isabs(p) and os.path.isfile(p):
                return p
            entry = (self.models_cfg.get("clip_vision") or {}).get(name)
            if entry is not None:
                files = [self._resolve(f) for f in self._entry_files(entry)]
                if not files:
                    raise ServerError(f"clip_vision 条目 {name!r} 没有文件")
                if not os.path.exists(files[0]):
                    raise ServerError(f"clip_vision 文件不存在：{files[0]}")
                return files[0]
            for cand in (os.path.join(self.clip_vision_dir(), name),
                         os.path.join(self.comfy_dir, "models", name),
                         self._resolve(name)):
                if os.path.isfile(cand):
                    return cand
            raise ServerError(
                f"找不到视觉编码器 {name!r}。\n"
                f"已扫描目录：{self.clip_vision_dir()}（{len(self.list_clip_vision_files())} 个文件）。\n"
                f"提示：clip_vision 是 EVA02_CLIP_L_336 这一类**图像编码器**；"
                f"Qwen3-VL / Qwen2.5-VL 那样的是**文本编码器**，用「Qwen-VL 图文混编」模式，别填在这里。\n"
                f"排查：① 把 clip_vision 文件放进上述目录；"
                f"② 或在 config.json 里加 \"clip_vision_model_dir\" 指向存放目录；"
                f"③ 或直接填该文件的绝对路径"
            )
        files = self.list_clip_vision_files()
        if len(files) == 1:
            return os.path.join(self.clip_vision_dir(), files[0])
        if not files:
            raise ServerError(
                f"没有可用的视觉编码器：{self.clip_vision_dir()} 是空的。\n"
                f"说明：「图片 → 服务端 CLIP 视觉」走的是 clip_vision 模型（如 EVA02_CLIP_L_336_psz14_s6B.pt），"
                f"不是文本编码器——Qwen3-VL / Qwen2.5-VL 这类是「文本编码器」，请改用节点的"
                f"「图片 → 服务端文本编码器（Qwen-VL 图文混编）」模式。\n"
                f"确实要用 Redux/IPAdapter 的话：把 clip_vision 模型放进该目录，"
                f"或在 config.json 写 \"clip_vision_model_dir\"，或在节点「视觉模型」里填绝对路径"
            )
        raise ServerError(
            f"视觉编码器目录里有 {len(files)} 个文件：{', '.join(files[:6])}\n"
            f"请在节点「视觉模型」里填其中一个的名字"
        )

    def get_clip_vision(self, name: str = ""):
        path = self._resolve_clip_vision(name)
        with self._lock:
            cache = self._loaded["clip_vision"]
            key = path
            if key in cache:
                cache.move_to_end(key)
                self._last_active = time.time()
                return cache[key]
            c = self._import_comfy()
            if not os.path.exists(path):
                raise ServerError(f"视觉编码器文件不存在：{path}")
            print(f"[服务端] 加载 clip_vision {path}", flush=True)
            t0 = time.time()
            try:
                cv = c["cv"].load(path)
            except Exception as e:
                raise ServerError(
                    f"加载视觉编码器失败：{path}\n{e}\n"
                    f"排查：该文件是否为 clip_vision 类模型（不是文本 CLIP、也不是整包 checkpoint）"
                ) from e
            print(f"[服务端] clip_vision 加载完成，耗时 {time.time() - t0:.1f}s "
                  f"image_size={getattr(cv, 'image_size', '?')}", flush=True)
            self._evict("clip_vision", self.max_cv)
            cache[key] = cv
            return cv

    def clip_vision_encode(self, model: str, images):
        """images: [B,H,W,C] float 0~1（已由 codec 解码好）。"""
        if not torch.is_tensor(images):
            raise ServerError("images 必须是张量")
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if images.shape[-1] > 3:
            images = images[..., :3]
        with self._lock:
            self._last_active = time.time()
            cv = self.get_clip_vision(model)
            try:
                out = cv.encode_image(images.to(torch.float32))
            except Exception as e:
                raise ServerError(f"视觉编码失败：{e}") from e
            payload = {}
            for k in ("image_embeds", "penultimate_hidden_states", "last_hidden_state"):
                v = out[k] if hasattr(out, "__getitem__") else getattr(out, k, None)
                if torch.is_tensor(v):
                    payload[k] = _to_cpu(v)
            if "penultimate_hidden_states" not in payload:
                raise ServerError("视觉编码器没有返回 penultimate_hidden_states，无法用于 Redux 式连接")
            payload["image_sizes"] = [list(s) for s in (getattr(out, "image_sizes", []) or [])]
            payload["model"] = os.path.basename(self._resolve_clip_vision(model))
            return payload

    # ------------------------------------------------------------------ #
    # 参考图 latent（Qwen/Flux Edit 那类 reference_latents）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fit_pixels(pixels, max_pixels: int = 1048576, multiple: int = 8):
        """按总像素上限等比缩放，宽高对齐到 multiple。"""
        b, h, w, c = pixels.shape
        if max_pixels and max_pixels > 0 and h * w > max_pixels:
            scale = (float(max_pixels) / float(h * w)) ** 0.5
        else:
            scale = 1.0
        nh = max(multiple, int(round(h * scale / multiple)) * multiple)
        nw = max(multiple, int(round(w * scale / multiple)) * multiple)
        if (nh, nw) == (h, w):
            return pixels
        x = pixels.movedim(-1, 1).to(torch.float32)
        x = torch.nn.functional.interpolate(x, size=(nh, nw), mode="bilinear", antialias=True)
        return x.movedim(1, -1).clamp(0.0, 1.0)

    def ref_latent_encode(self, model: str, pixels, max_pixels: int = 1048576, multiple: int = 8):
        if not torch.is_tensor(pixels):
            raise ServerError("pixels 必须是张量")
        if pixels.dim() == 3:
            pixels = pixels.unsqueeze(0)
        if pixels.shape[-1] == 1:
            pixels = pixels.repeat(1, 1, 1, 3)
        pixels = self._fit_pixels(pixels, int(max_pixels or 0), max(8, int(multiple or 8)))
        samples = self.vae_encode(model, pixels)
        return {"samples": samples,
                "height": int(pixels.shape[1]), "width": int(pixels.shape[2])}

    def _detect_clip_type(self, files, name: str = "") -> str:
        """猜 clip_type（不加载权重，几乎零成本）：
        1) 先看 safetensors 里的参数名——Qwen-VL 系带视觉塔（model.visual.* / visual.*）；
        2) 再看文件/条目名——含 qwen + vl / qwen_image 的，就是 Qwen-Image 系文本编码器。
        用 stable_diffusion 加载会拿到 SD1 的 tokenizer，图片永远进不去。"""
        probe = (str(name) + " " + " ".join(os.path.basename(str(f)) for f in files)).lower()
        keys = set()
        try:
            import json as _json
            import struct as _struct
            for f in files:
                if not str(f).lower().endswith(".safetensors") or not os.path.exists(f):
                    continue
                with open(f, "rb") as fh:
                    n = _struct.unpack("<Q", fh.read(8))[0]
                    meta = _json.loads(fh.read(n))
                keys.update(k for k in meta if k != "__metadata__")
        except Exception as e:
            print(f"[服务端] 读取权重参数名失败（不影响启动）：{e}", flush=True)
        if keys and any(k.startswith("model.visual.") or k.startswith("visual.") for k in keys):
            return "qwen_image"  # Qwen-VL 系（含 Qwen-Image 2.1 的 Qwen3-VL-8B）
        if keys and any(k.startswith("conditioner.") for k in keys):
            return "flux"
        if "qwen" in probe and ("vl" in probe or "qwen_image" in probe or "qwen-image" in probe):
            return "qwen_image"
        return ""

    def _clip_type(self, name: str):
        c = self._import_comfy()
        key = str(name or "stable_diffusion").strip().upper().replace("-", "_")
        ct = getattr(c["sd"].CLIPType, key, None)
        if ct is None:
            avail = [x.name.lower() for x in c["sd"].CLIPType]
            raise ServerError(f"未知 clip_type {name!r}，可用：{', '.join(avail)}")
        return ct

    def _evict(self, kind: str, limit: int):
        cache = self._loaded[kind]
        while len(cache) >= max(1, limit):
            old_name, old_obj = cache.popitem(last=False)
            print(f"[服务端] 卸载 {kind} {old_name}（超出 max_loaded={limit}）", flush=True)
            self._free(old_obj)

    def _vram(self):
        """当前显存快照（GB + torch 分配/预留 MB）；没有 CUDA 就返回 None。"""
        try:
            if not torch.cuda.is_available():
                return None
            free, total = torch.cuda.mem_get_info()
            return {"free_gb": round(free / 2**30, 2),
                    "used_gb": round((total - free) / 2**30, 2),
                    "total_gb": round(total / 2**30, 2),
                    "torch_allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 1),
                    "torch_reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 1)}
        except Exception:
            return None

    def _hard_free_gpu(self):
        """真把显存还回去：ComfyUI 的 current_loaded_models 里可能还挂着 CLIP/VAE，
        光把对象从 self._loaded 里 pop 掉，nvidia-smi 里照样占着 —— 必须先让
        model_management 卸掉它自己加载的那份，再强制 empty_cache。"""
        mm = (self._comfy or {}).get("mm")
        if mm is not None:
            try:
                mm.unload_all_models()
            except Exception as e:
                print(f"[服务端] unload_all_models 失败（忽略）：{e}", flush=True)
            try:
                mm.soft_empty_cache(force=True)
            except Exception:
                try:
                    mm.soft_empty_cache()
                except Exception:
                    pass
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass
        self._torch_cache = True

    def _free(self, obj):
        del obj
        gc.collect()
        self._hard_free_gpu()

    def unload(self, kind: str | None = None, name: str | None = None) -> dict:
        before = self._vram()
        freed = []
        with self._lock:
            kinds = [kind] if kind in ("clip", "vae", "clip_vision") else ["clip", "vae", "clip_vision"]
            for k in kinds:
                cache = self._loaded[k]
                for n in list(cache.keys()):
                    if name and n != name:
                        continue
                    obj = cache.pop(n)
                    freed.append(f"{k}:{n}")
                    del obj
            # 不管缓存里有没有 pop 到东西，都要把 ComfyUI 自己那份卸掉（这才是显存真正占着的地方）
            gc.collect()
            self._hard_free_gpu()
            self._last_active = time.time() - self.idle_unload  # 别让空闲线程又跑一遍
        after = self._vram()
        print(f"[服务端] 卸载 {kind or '全部'}{('/' + name) if name else ''} → freed={freed or '（缓存里没有）'} "
              f"显存 {self._vram_text(before)} → {self._vram_text(after)}", flush=True)
        return {"freed": freed, "vram_before": before, "vram_after": after,
                "loaded": {k: list(v.keys()) for k, v in self._loaded.items()}}

    @staticmethod
    def _vram_text(v) -> str:
        if not v:
            return "无 CUDA"
        return f"{v['used_gb']}GB / {v['total_gb']}GB"

    def _idle_loop(self):
        while True:
            time.sleep(max(5.0, self.idle_unload / 4.0))
            if time.time() - self._last_active < self.idle_unload:
                continue
            loaded = [f"{k}:{n}" for k, cache in self._loaded.items() for n in cache]
            if not loaded:
                continue
            print(f"[服务端] 空闲超过 {self.idle_unload:.0f}s，释放 {loaded}", flush=True)
            self.unload()

    # ------------------------------------------------------------------ #
    # 操作：CLIP 文本编码
    # ------------------------------------------------------------------ #
    def clip_encode(self, model: str, text: str, last_layer: int = 0, clip_type: str = "",
                    images=None, vision_pixels: int = 1048576, vae: str = "", resolution: int = 0,
                    no_resize: bool = False, keep_vision=None, prevent_empty_text: bool = False,
                    llama_template: str = ""):
        """images: [ [B,H,W,C] 张量, ... ]，每项是一张参考图。

        有图时完全照 ComfyUI 官方 Qwen-Image-Edit 节点（comfy_extras/nodes_qwen.py）的做法：
        图缩放后同时喂「文本编码器 vision 槽」和「VAE latent 槽」；没有 vae 就 keep_vision=True。
        no_resize=True 表示客户端（原生节点）已经把图缩到 32 的倍数，服务端直接按原样用。
        keep_vision 显式给定时以它为准（原生节点会按自己有没有 reference_latents 决定）。"""
        if not isinstance(text, str):
            raise ServerError("text 必须是字符串")
        with self._lock:
            self._last_active = time.time()
            clip = self.get_clip(model, clip_type)
            if last_layer:
                clip = clip.clone()
                clip.clip_layer(int(last_layer))  # 与 ComfyUI 的 CLIPSetLastLayer 完全一致

            imgs = []
            if images:
                for g in images:
                    if not torch.is_tensor(g):
                        continue
                    if g.dim() == 3:
                        g = g.unsqueeze(0)
                    if g.shape[-1] > 4:
                        g = g[..., :4]
                    for i in range(g.shape[0]):
                        imgs.append(g[i:i + 1].to(torch.float32))
            if imgs:
                res = int(resolution or 0)
                if res <= 0:
                    res = int(round((int(vision_pixels or 1048576)) ** 0.5))
                return _to_cpu(self._qwen_vl_conditioning(
                    clip, text, imgs, vae, res,
                    resize=not bool(no_resize),
                    keep_vision=keep_vision,
                    prevent_empty_text=bool(prevent_empty_text),
                    llama_template=str(llama_template or "")))

            tokens = clip.tokenize(text)
            scheduled = getattr(clip, "encode_from_tokens_scheduled", None)
            if callable(scheduled):
                out = scheduled(tokens)
            else:
                cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
                out = [[cond, {"pooled_output": pooled}]]
            conds = self._canonical_conditioning(out)
        return _to_cpu(conds)

    def _qwen_vl_conditioning(self, clip, text: str, imgs: list, vae_name: str = "", resolution: int = 1024,
                              resize: bool = True, keep_vision=None, prevent_empty_text: bool = False,
                              llama_template: str = ""):
        """照抄 comfy_extras/nodes_qwen.py 的 Qwen-Image-Edit 逻辑：
        参考图按 resolution² 等比缩放到 32 的倍数，同一张图既做 vision 输入又做 VAE latent，
        这样每个 vision slot 正好覆盖 2×2 个 latent。
        resize=False：客户端（原生节点）已经缩好了，直接用。"""
        images_vl, ref_latents = [], []
        vae_model = self.get_vae(vae_name) if vae_name else None
        res = max(0, int(resolution or 0))
        for img in imgs:
            s = img[:1]
            if resize:
                s = s.permute(0, 3, 1, 2)  # B,C,H,W
                if res > 0:
                    ratio = s.shape[3] / s.shape[2]
                    width = round(math.sqrt(res * res * ratio) / 32) * 32
                    height = round(math.sqrt(res * res / ratio) / 32) * 32
                else:
                    width, height = round(s.shape[3] / 32) * 32, round(s.shape[2] / 32) * 32
                width, height = max(32, width), max(32, height)
                if (width, height) == (s.shape[3], s.shape[2]):
                    s = img[:1]
                else:
                    s = self._comfy["utils"].common_upscale(
                        s, int(width), int(height), "lanczos", "disabled").permute(0, 2, 3, 1)
            rgb = s[..., :3]
            if s.shape[-1] > 3:  # 透明图：视觉塔按白底合成，VAE 保留 alpha
                a = s[..., 3:]
                rgb = rgb * a + (1.0 - a)
            images_vl.append(rgb.float())
            if vae_model is not None:
                ref_latents.append(vae_model.encode(s.float()))

        if keep_vision is None:
            # 跟原生 tokenizer 的默认一致（QwenImage21Tokenizer: keep_vision=False）。
            # 之前这里按「服务端没编 latent 就 True」推断，结果原生 2.0 节点（不传 keep_vision、
            # 自己挂 reference_latents）会变成「vision token 留一份 + latent 再挂一份」，双重插入 → 出图糊/重影。
            keep_vision = False
        keep_vision = bool(keep_vision)
        try:
            # 不自己拼 <|im_start|> 模板：Qwen-Image 2.1 的 tokenizer 会按图数重复 vision 槽并记录 image_slots
            tokens = clip.tokenize(text, images=images_vl, keep_vision=keep_vision,
                                   prevent_empty_text=prevent_empty_text,
                                   llama_template=(llama_template or None))
        except TypeError as e:
            raise ServerError(
                f"这个文本编码器不接受图片输入（tokenize 不支持 images=）：{e}\n"
                f"只有 Qwen-VL 系（qwen_image / Qwen3-VL 文本编码器）能这样用。"
                f"若是普通 CLIP / T5，请改用「参考图 → 服务端 VAE 编码」或「图片 → 服务端 clip_vision」模式"
            ) from e
        except Exception as e:
            raise ServerError(f"图文混编失败：{e}") from e

        def _is_img(t):
            if isinstance(t, dict):
                return t.get("type") == "image"
            if isinstance(t, (tuple, list)) and len(t) > 0 and isinstance(t[0], dict):
                return t[0].get("type") == "image"
            return False

        hits = 0
        try:
            groups = tokens.values() if isinstance(tokens, dict) else [tokens]
            hits = sum(1 for grp in groups for row in grp for t in row if _is_img(t))
        except Exception:
            hits = len(images_vl)
        if hits < len(images_vl):
            print(f"[服务端] 警告：只命中 {hits}/{len(images_vl)} 个 vision token —— "
                  f"这个 CLIP 的 tokenizer 不认识 vision 占位符。检查 config.json 里该条目的 "
                  f"\"clip_type\" 是否为 qwen_image（Qwen-Image / Qwen3-VL 系必须用它；"
                  f"填 stable_diffusion 会按 SD1.5 的 tokenizer 处理，图片进不去）", flush=True)

        scheduled = getattr(clip, "encode_from_tokens_scheduled", None)
        if callable(scheduled):
            out = scheduled(tokens)
        else:
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            out = [[cond, {"pooled_output": pooled}]]
        conds = self._canonical_conditioning(out)
        if ref_latents:
            conds = self._append_cond_values(conds, "reference_latents", ref_latents)
            h, w = ref_latents[0].shape[-2] * 16, ref_latents[0].shape[-1] * 16
            print(f"[服务端] 图文混编：{len(images_vl)} 张图 / {hits} 个 vision token / keep_vision={keep_vision}，"
                  f"挂 {len(ref_latents)} 份 reference_latents（采样尺寸建议 {w}x{h}）", flush=True)
        else:
            print(f"[服务端] 图文混编：{len(images_vl)} 张图 / {hits} 个 vision token / keep_vision={keep_vision}"
                  + ("（客户端自带 reference_latents，服务端不再编）" if not keep_vision
                     else "（图像特征直接进条件）"), flush=True)
            if not keep_vision:
                print("[服务端] 注意：keep_vision=False 但没有 reference_latents，"
                      "参考图不会出现在结果里（客户端应把 VAE 编码的 latent 一起挂上条件）", flush=True)
        return conds

    @staticmethod
    def _append_cond_values(conds, key: str, values: list):
        """等价 node_helpers.conditioning_set_values(..., append=True)"""
        out = []
        for c, d in conds:
            d = dict(d or {})
            d[key] = list(d.get(key) or []) + list(values)
            out.append([c, d])
        return out

    @staticmethod
    def _canonical_conditioning(out) -> list:
        """统一成 ComfyUI 的 CONDITIONING 规范：[ [tensor, dict], ... ]"""
        if isinstance(out, dict):
            if "cond" in out or "conditioning" in out:
                out = [out]
            else:
                out = [[out, {}]]
        if torch.is_tensor(out):
            out = [[out, {}]]
        result = []
        for item in out:
            if torch.is_tensor(item):
                result.append([item, {}])
                continue
            if isinstance(item, dict):
                d = dict(item)
                cond = d.pop("cond", None)
                if cond is None:
                    cond = d.pop("conditioning", None)
                if cond is None:
                    raise ServerError(f"CLIP 输出结构无法识别：keys={list(d.keys())}")
                result.append([cond, {k: v for k, v in d.items() if v is not None}])
                continue
            if isinstance(item, (list, tuple)) and len(item) == 2 and torch.is_tensor(item[0]):
                result.append([item[0], dict(item[1] or {})])
                continue
            raise ServerError(f"CLIP 输出结构无法识别：{type(item)!r}")
        if not result:
            raise ServerError("CLIP 编码结果为空")
        return result

    # ------------------------------------------------------------------ #
    # 操作：VAE
    # ------------------------------------------------------------------ #
    def vae_decode(self, model: str, samples, tiled: bool = False, tile_size: int = 512, overlap: int = 64):
        if not torch.is_tensor(samples):
            raise ServerError("latent 必须是张量")
        if samples.dim() == 3:
            samples = samples.unsqueeze(0)
        with self._lock:
            self._last_active = time.time()
            vae = self.get_vae(model)
            ratio = self._vae_ratio(vae)
            expect_ch = int(getattr(vae, "latent_channels", 4) or 4)
            if samples.shape[1] != expect_ch:
                raise ServerError(
                    f"VAE 解码失败：latent 有 {samples.shape[1]} 个通道，但该 VAE 需要 {expect_ch} 个。\n"
                    f"通常说明本机的图像模型与服务端这个 VAE 不是同一代模型（例如 SD1.5 的 4 通道 latent 配了 16 通道的 Flux/Qwen VAE），"
                    f"请换成与本机 UNet 同源的 VAE。"
                )
            samples = samples.to(dtype=self._vae_dtype(vae))
            # 视频 VAE（Wan / Qwen-Image / HunyuanVideo…）要 5 维 [B,C,T,H,W]；单帧时补一个 T=1
            latent_in = samples
            if self._is_video_vae(vae) and samples.dim() == 4:
                latent_in = samples.unsqueeze(2)
            try:
                if tiled:
                    tx = max(64, int(tile_size // ratio))
                    ov = max(8, int(overlap // ratio))
                    try:
                        images = vae.decode_tiled(latent_in, tile_x=tx, tile_y=tx, overlap=ov)
                    except TypeError:
                        images = vae.decode_tiled(latent_in, tile_x=tx, tile_y=tx)
                else:
                    images = vae.decode(latent_in)
            except Exception as e:
                raise ServerError(
                    f"VAE 解码失败：{e}\n排查：服务端显存不足（勾 use_tiled / 调小 tile_size）、"
                    f"或 latent 与该 VAE 不匹配（跨机模型不同源）；视频 VAE 需要 5 维 latent，已尝试自动补 T=1"
                ) from e
            if isinstance(images, dict):  # 兼容极少数自定义 VAE 的返回
                images = images.get("samples", images)
            if torch.is_tensor(images) and images.dim() == 5:  # [B,T,H,W,C] -> 取首帧
                images = images[:, 0]
            if torch.is_tensor(images) and images.dim() == 3:
                images = images.unsqueeze(0)
            self._last_active = time.time()
            return _to_cpu(images)

    def vae_encode(self, model: str, pixels, tiled: bool = False, tile_size: int = 512, overlap: int = 64):
        if not torch.is_tensor(pixels):
            raise ServerError("pixels 必须是张量")
        if pixels.dim() == 3:
            pixels = pixels.unsqueeze(0)
        if pixels.shape[-1] == 1:
            pixels = pixels.repeat(1, 1, 1, 3)
        with self._lock:
            self._last_active = time.time()
            vae = self.get_vae(model)
            pixels = pixels.to(dtype=torch.float32)
            # 视频 VAE 优先按 [B,T,H,W,C]（单帧 T=1）传，失败再退回 4 维
            variants = [pixels]
            if self._is_video_vae(vae) and pixels.dim() == 4:
                variants.insert(0, pixels.unsqueeze(1))
            samples, last_err = None, None
            for pin in variants:
                try:
                    if tiled:
                        tx = max(64, int(tile_size))
                        ov = max(8, int(overlap))
                        try:
                            samples = vae.encode_tiled(pin, tile_x=tx, tile_y=tx, overlap=ov)
                        except TypeError:
                            samples = vae.encode_tiled(pin, tile_x=tx, tile_y=tx)
                    else:
                        samples = vae.encode(pin)
                    break
                except Exception as e:
                    last_err = e
            if samples is None:
                raise ServerError(
                    f"VAE 编码失败：{last_err}\n排查：服务端显存不足（勾 use_tiled）、"
                    f"或该 VAE 与本机图像模型不同源"
                )
            if isinstance(samples, dict):
                samples = samples.get("samples", samples)
            if torch.is_tensor(samples) and samples.dim() == 5:  # [B,C,T,H,W] -> 单帧
                samples = samples[:, :, 0]
            return _to_cpu(samples)

    @staticmethod
    def _is_video_vae(vae) -> bool:
        """Wan / Qwen-Image / HunyuanVideo 这类 VAE 的 downscale_ratio 是 (fn, h, w) 元组。"""
        return isinstance(getattr(vae, "downscale_ratio", None), (tuple, list))

    @staticmethod
    def _vae_ratio(vae) -> int:
        dr = getattr(vae, "downscale_ratio", None)
        if isinstance(dr, (int, float)) and dr >= 1:
            return int(round(float(dr)))
        if isinstance(dr, (tuple, list)):  # 视频 VAE：取空间比例
            for x in dr[1:]:
                if isinstance(x, (int, float)) and x >= 1:
                    return int(x)
        for attr in ("downscale_scale",):
            v = getattr(vae, attr, None)
            if isinstance(v, (int, float)) and v >= 1:
                return int(round(float(v)))
        return 8

    @staticmethod
    def _vae_dtype(vae):
        try:
            return vae.vae_dtype()
        except Exception:
            pass
        for attr in ("dtype", "vae_dtype"):
            v = getattr(vae, attr, None)
            if isinstance(v, torch.dtype):
                return v
        return torch.float32

    # ------------------------------------------------------------------ #
    # 状态
    # ------------------------------------------------------------------ #
    def status(self, light: bool = False) -> dict:
        info = {
            "comfyui_dir": self.comfy_dir,
            "comfy_imported": self._comfy is not None,
            "loaded": {k: list(v.keys()) for k, v in self._loaded.items()},
            "max_loaded": {"clip": self.max_clip, "vae": self.max_vae},
            "idle_unload_seconds": self.idle_unload,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": "cpu" if self.device == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu"),
            "device_name": None,
            "vram": None,
            "torch_allocated_mb": None,
            "torch_reserved_mb": None,
            "models": ({} if light else self.list_models()),
            "llm": self.llm.status(light=light),
        }
        try:
            if torch.cuda.is_available():
                info["device_name"] = torch.cuda.get_device_name(0)
                free, total = torch.cuda.mem_get_info()
                info["vram"] = {"free_gb": round(free / 2**30, 2), "total_gb": round(total / 2**30, 2)}
                info["torch_allocated_mb"] = round(torch.cuda.memory_allocated() / 2**20, 1)
                info["torch_reserved_mb"] = round(torch.cuda.memory_reserved() / 2**20, 1)
            else:
                info["device_name"] = "CPU"
        except Exception:
            pass
        return info

    def handle(self, op: str, params: dict) -> dict:
        """统一入口：返回可被 codec.pack 的 Python 对象。"""
        params = params or {}
        if op in ("ping", "status"):
            return self.status()
        if op == "models":
            return self.list_models()
        if op == "load":
            kind = params.get("kind")
            name = params.get("model") or params.get("name")
            if kind == "clip":
                ctype = str(params.get("clip_type") or "").strip()
                if ctype:
                    self._clip_override[name] = ctype
                self.get_clip(name, ctype)
            elif kind == "vae":
                self.get_vae(name)
            else:
                raise ServerError("load 的 kind 必须是 clip 或 vae")
            return {"loaded": {k: list(v.keys()) for k, v in self._loaded.items()}}
        if op == "unload":
            return self.unload(params.get("kind"), params.get("model"))
        if op == "vae_info":
            vae = self.get_vae(params.get("model", ""))
            return {
                "latent_channels": int(getattr(vae, "latent_channels", 4) or 4),
                "downscale_ratio": self._vae_ratio(vae),
                "model": params.get("model", ""),
            }
        if op == "clip_encode":
            model = params.get("model", "")
            ctype = str(params.get("clip_type") or "").strip()
            if ctype:
                self._clip_override[model] = ctype
            imgs = params.get("images")
            if imgs is None and params.get("image_frames"):
                imgs = [params["image_frames"]]
            if imgs:
                imgs = [codec.png_frames_to_tensor(g) for g in imgs if g]
            conds = self.clip_encode(model, str(params.get("text", "")),
                                     int(params.get("last_layer", 0) or 0), ctype,
                                     images=imgs or None,
                                     vision_pixels=int(params.get("vision_pixels", 1048576) or 0),
                                     vae=str(params.get("vae", "") or ""),
                                     resolution=int(params.get("resolution", 0) or 0),
                                     no_resize=bool(params.get("no_resize", False)),
                                     keep_vision=(None if params.get("keep_vision") is None
                                                  else bool(params.get("keep_vision"))),
                                     prevent_empty_text=bool(params.get("prevent_empty_text", False)),
                                     llama_template=str(params.get("llama_template", "") or ""))
            return {"conditioning": conds, "count": len(conds),
                    "images_used": len(imgs or [])}
        if op == "vae_decode":
            latent = params.get("latent")
            images = self.vae_decode(
                params.get("model", ""), latent,
                bool(params.get("tiled", False)),
                int(params.get("tile_size", 512) or 512),
                int(params.get("overlap", 64) or 64),
            )
            return {
                "frames": codec.tensor_to_png_frames(images),
                "height": int(images.shape[1]),
                "width": int(images.shape[2]),
                "channels": int(images.shape[3]),
            }
        if op == "clip_vision_encode":
            images = params.get("images")
            if images is None and params.get("frames"):
                images = codec.png_frames_to_tensor(params["frames"])
            return self.clip_vision_encode(params.get("model", ""), images)
        if op == "clip_vision_list":
            return {"dir": self.clip_vision_dir(), "files": self.list_clip_vision_files()}
        if op == "ref_latent_encode":
            pixels = params.get("pixels")
            if pixels is None and params.get("frames"):
                pixels = codec.png_frames_to_tensor(params["frames"])
            return self.ref_latent_encode(
                params.get("model", ""), pixels,
                int(params.get("max_pixels", 1048576) or 0),
                int(params.get("multiple", 8) or 8),
            )
        if op == "vae_encode":
            pixels = params.get("pixels")
            if pixels is None and params.get("frames"):
                pixels = codec.png_frames_to_tensor(params["frames"])
            samples = self.vae_encode(
                params.get("model", ""), pixels,
                bool(params.get("tiled", False)),
                int(params.get("tile_size", 512) or 512),
                int(params.get("overlap", 64) or 64),
            )
            return {"samples": samples}
        # ---------------- LLM（提示词优化：用服务端 ComfyUI 里的 GGUF） ---------------- #
        if op == "llm_models":
            return self.llm.list_models(ttl=0.0)
        if op == "llm_status":
            return self.llm.status()
        if op == "llm_env":
            return self.llm.env()
        if op == "llm_uninstall":
            return self.llm.uninstall()
        if op == "llm_bench":
            return self.llm.bench(int(params.get("prompt_tokens", 512) or 512),
                                  int(params.get("gen_tokens", 64) or 64))
        if op == "llm_load":
            return self.llm.load(str(params.get("model") or ""),
                                 str(params.get("mmproj") or ""),
                                 int(params.get("n_ctx", 4096) or 4096),
                                 int(params.get("n_gpu_layers", -1) if params.get("n_gpu_layers") is not None else -1))
        if op == "llm_unload":
            before = self._vram()
            st = self.llm.unload()
            self._hard_free_gpu()
            after = self._vram()
            st["freed"] = ["llm: worker 进程已退出"]
            st["vram_before"] = before
            st["vram_after"] = after
            return st
        if op == "llm_chat":
            images = params.get("images") or []
            if not images and params.get("frames"):
                images = params["frames"]
            data = self.llm.chat(
                str(params.get("model") or ""),
                str(params.get("text") or ""),
                system=str(params.get("system") or ""),
                images=list(images or []),
                max_tokens=int(params.get("max_tokens", 1024) or 1024),
                temperature=float(params.get("temperature", 0.7) or 0.7),
                seed=(None if params.get("seed") is None else int(params.get("seed"))),
                mmproj=str(params.get("mmproj") or ""),
                n_ctx=int(params.get("n_ctx", 4096) or 4096),
                # v1.7.34：timeout 传 None = 不设上限，慢慢生成也等到底
                timeout=(float(params["timeout"]) if params.get("timeout") else None),
                auto_load=bool(params.get("auto_load", True)),
            )
            return data
        raise ServerError(f"未知操作：{op!r}")


class MockBackend(Backend):
    """
    联调用假后端：不加载任何模型，只用随机张量走完整链路。
    用法：python server.py --mock
    作用：先在两台机器上验证网络 / 协议 / 节点接线是否通，再换真模型。
    """

    def __init__(self, cfg: dict, config_dir: str):
        super().__init__(cfg, config_dir)
        self.idle_unload = 0
        from llm import MockLLM
        self.llm = MockLLM(self)

    def _import_comfy(self):
        return {}

    def list_models(self):
        return {
            "clip": {"mock_clip_768": {"files": [], "resolved": [], "exists": True,
                                      "clip_type": "stable_diffusion", "note": "SD1.5 形状的假 CLIP"}},
            "vae": {"mock_vae": {"files": [], "resolved": [], "exists": True,
                                "clip_type": None, "note": "8x 下采样的假 VAE"}},
            "clip_vision": {"mock_clipvision.safetensors": {"files": [], "resolved": [],
                                                            "exists": True, "note": "假视觉编码器"}},
            "llm": {"mock_llm.gguf": {"files": ["mock_llm.gguf"], "exists": True, "note": "假 GGUF"}},
            "llm_vision": {"mock_mmproj.gguf": {"files": ["mock_mmproj.gguf"], "exists": True,
                                                "note": "假 mmproj"}},
        }

    def get_clip(self, name, clip_type=""):
        return object()

    def get_vae(self, name):
        return object()

    def clip_encode(self, model, text, last_layer=0, clip_type="", images=None, vision_pixels=1048576,
                    vae="", resolution=0, no_resize=False, keep_vision=None, prevent_empty_text=False,
                    llama_template=""):
        n = len(images or [])
        ctx = torch.randn(1, 77 + 64 * n, 768, dtype=torch.float32)
        out = [[ctx, {"pooled_output": torch.randn(1, 1280, dtype=torch.float32)}]]
        if vae and n:
            out = self._append_cond_values(out, "reference_latents",
                                           [torch.randn(1, 16, 40, 40) for _ in range(n)])
        return out

    def vae_decode(self, model, samples, tiled=False, tile_size=512, overlap=64):
        b, c, h, w = samples.shape
        return torch.rand(b, h * 8, w * 8, 3, dtype=torch.float32)

    def vae_encode(self, model, pixels, tiled=False, tile_size=512, overlap=64):
        b, h, w, c = pixels.shape
        return torch.randn(b, 4, h // 8, w // 8, dtype=torch.float32)

    def list_clip_vision_files(self):
        return ["mock_clipvision.safetensors"]

    def get_clip_vision(self, name=""):
        return object()

    def clip_vision_encode(self, model, images):
        b = images.shape[0]
        return {"image_embeds": torch.randn(b, 81, 64, dtype=torch.float32),
                "penultimate_hidden_states": torch.randn(b, 729, 64, dtype=torch.float32),
                "last_hidden_state": torch.randn(b, 729, 64, dtype=torch.float32),
                "image_sizes": [], "model": "mock_clipvision.safetensors"}

    def ref_latent_encode(self, model, pixels, max_pixels=1048576, multiple=8):
        pixels = self._fit_pixels(pixels, max_pixels, max(8, multiple))
        return {"samples": self.vae_encode(model, pixels),
                "height": int(pixels.shape[1]), "width": int(pixels.shape[2])}

    def status(self):
        s = super().status()
        s["mock"] = True
        s["device_name"] = "MOCK"
        s["models"] = self.list_models()
        return s
