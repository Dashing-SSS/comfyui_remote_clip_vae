# -*- coding: utf-8 -*-
"""
config.py —— 配置读写 + 环境探测 + 模型自动扫描。

被 1_检查环境并生成配置.bat 调用，也可以自己跑：

    python config.py --show
    python config.py --comfy-path "F:\\ComfyUI_Mie_xxx\\ComfyUI" --python "F:\\...\\python_embeded\\python.exe" --write --add-models
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CFG = os.path.join(HERE, "config.json")
MODEL_EXTS = ((".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".gguf", ".bin"))
SCAN_SUBDIRS = ("vae", "text_encoders", "clip", "clip_vision", "diffusion_models", "unet", "checkpoints")
# 旧式 clip/ 目录里混着 clip_vision（视觉编码器），当文本编码器加载会直接报错
VISION_HINTS = ("eva02", "eva-clip", "sigclip", "clip_vision", "clipvision", "vision")


def clean_path(v) -> str:
    """把拖进输入框的 "D:\\...\\python.exe" 变成干净的路径（Windows 拖拽会自动加引号）。"""
    s = str(v or "").strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    s = s.strip().strip("'").strip()
    if s.startswith("\\\\?\\"):
        s = s[4:]
    return s


def is_vision(rel: str) -> bool:
    name = os.path.basename(rel).lower()
    return any(h in name for h in VISION_HINTS)


def load(path: str = DEFAULT_CFG) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # 手改/拖拽带进来的引号统一清掉，后端按干净路径去找模型
        if isinstance(cfg.get("comfyui_dir"), str):
            cfg["comfyui_dir"] = clean_path(cfg["comfyui_dir"]).replace("\\", "/")
        for k in ("clip_model_dir", "vae_model_dir", "clip_vision_model_dir", "llm_model_dir"):
            if isinstance(cfg.get(k), str):
                cfg[k] = clean_path(cfg[k])
        for k in ("llm_python",):
            if isinstance(cfg.get(k), str):
                cfg[k] = clean_path(cfg[k])
        # config.json 里没写 ComfyUI 目录时，退用 1_ / 2_ / 面板记下的 _comfy.txt
        if not cfg.get("comfyui_dir"):
            rec = os.path.join(os.path.dirname(os.path.abspath(path)), "_comfy.txt")
            if os.path.isfile(rec):
                with open(rec, "r", encoding="utf-8", errors="replace") as f:
                    d = clean_path(f.read())
                if d:
                    cfg["comfyui_dir"] = d.replace("\\", "/")
        return cfg
    return {"server": {"host": "0.0.0.0", "port": 8199, "token": "", "max_request_mb": 256},
            "comfyui_dir": "", "clip_vision_model_dir": "", "max_loaded": {"clip": 1, "vae": 1},
            "idle_unload_seconds": 600, "models": {"clip": {}, "vae": {}, "clip_vision": {}}}


def save(cfg: dict, path: str = DEFAULT_CFG) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def safe_name(filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0]
    base = re.sub(r"[^0-9A-Za-z_.-]+", "_", base).strip("_").lower()
    return base or "model"


def scan(comfy_dir: str) -> dict:
    """扫描 <comfy_dir>/models 下的模型文件，路径统一成 '/' 分隔的相对路径。"""
    models_root = os.path.join(comfy_dir, "models")
    found = {k: [] for k in SCAN_SUBDIRS}
    for sub in SCAN_SUBDIRS:
        root = os.path.join(models_root, sub)
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            depth = dirpath[len(root):].count(os.sep)
            if depth > 2:
                _dirs.clear()
                continue
            for fn in files:
                if fn.lower().endswith(MODEL_EXTS) and not fn.startswith("put_"):
                    rel = os.path.relpath(os.path.join(dirpath, fn), models_root).replace(os.sep, "/")
                    if sub == "clip" and is_vision(rel):
                        # 旧式 clip/ 里混着的视觉编码器：不当文本编码器，改归到 clip_vision
                        found["clip_vision"].append(rel)
                        continue
                    found[sub].append(rel)
    return found


def auto_models(comfy_dir: str) -> dict:
    """根据扫描结果猜一份能直接用的 models 清单。"""
    found = scan(comfy_dir)
    clip, vae, cv = {}, {}, {}

    def pick(sub, patterns):
        for rel in found[sub]:
            name = os.path.basename(rel).lower()
            if all(re.search(p, name) for p in patterns):
                return rel
        return None

    t5 = pick("text_encoders", [r"t5"]) or pick("clip", [r"t5"])
    clip_l = pick("text_encoders", [r"clip_?l"]) or pick("clip", [r"clip_?l"])
    clip_g = pick("text_encoders", [r"clip_?g"]) or pick("clip", [r"clip_?g"])
    if t5 and clip_l:
        clip["flux_auto_clip"] = {"files": [t5, clip_l], "clip_type": "flux",
                                  "note": "自动识别：T5 + CLIP-L，适用于 Flux"}
    if clip_l and clip_g:
        clip["sdxl_auto_clip"] = {"files": [clip_l, clip_g], "clip_type": "stable_diffusion",
                                  "note": "自动识别：CLIP-L + CLIP-G，适用于 SDXL"}
    if not clip:
        for rel in (found["text_encoders"] + found["clip"])[:4]:
            # 不写死 clip_type：服务端会在加载时按权重内容/文件名自动识别（Qwen-VL 必须走 qwen_image）
            clip[safe_name(rel)] = {"files": [rel],
                                    "note": "自动识别：clip_type 由服务端在加载时判断"}
    for rel in found["vae"][:6]:
        vae[safe_name(rel)] = {"files": [rel], "note": "自动识别"}
    if not vae:
        for rel in found["checkpoints"][:2]:
            vae[safe_name(rel) + "_vae"] = {"files": [rel],
                                            "note": "自动识别：整包 checkpoint 当 VAE 源（不一定含 VAE）"}
    for rel in found["clip_vision"][:4]:
        cv[safe_name(rel)] = {"files": [rel], "note": "自动识别：视觉编码器（图片→CLIP 视觉 模式用）"}
    return {"clip": clip, "vae": vae, "clip_vision": cv}


def merge_models(cfg: dict, auto: dict) -> int:
    """把自动识别的条目合并进现有配置，已存在的路径不重复添加。"""
    added = 0
    cfg.setdefault("models", {}).setdefault("clip", {})
    cfg["models"].setdefault("vae", {})
    cfg["models"].setdefault("clip_vision", {})
    for kind in ("clip", "vae", "clip_vision"):
        existing_paths = set()
        for entry in (cfg["models"][kind] or {}).values():
            if isinstance(entry, dict):
                for f in entry.get("files", []):
                    existing_paths.add(str(f).replace("\\", "/").lower())
        for name, entry in (auto[kind] or {}).items():
            files = {f.lower() for f in entry["files"]}
            if files & existing_paths:
                continue
            cfg["models"][kind].setdefault(name, entry)
            added += 1
            existing_paths |= files
    return added


def summary(cfg: dict) -> str:
    lines = []
    comfy = clean_path(cfg.get("comfyui_dir")) or "(未设置)"
    lines.append(f"comfyui_dir : {comfy}   [{'OK' if os.path.isdir(comfy) else '缺失'}]")
    srv = cfg.get("server") or {}
    lines.append(f"listen      : {srv.get('host', '0.0.0.0')}:{srv.get('port', 8199)}"
                 f"   token={'有' if srv.get('token') else '无'}")
    for kind in ("clip", "vae", "clip_vision"):
        entries = (cfg.get("models") or {}).get(kind) or {}
        lines.append(f"{kind:11s} : {len(entries)} 个条目")
        for name, entry in entries.items():
            files = entry.get("files", []) if isinstance(entry, dict) else [str(entry)]
            ok = all(os.path.exists(f if os.path.isabs(f) else os.path.join(comfy, "models", f)) for f in files)
            lines.append(f"   - {name:28s} [{'OK' if ok else '文件缺失'}] {' + '.join(files)}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--comfy-path", dest="comfy_path", default=None)
    ap.add_argument("--python", default=None, help="把 ComfyUI 的 python 路径记录到 model_server/_python.txt")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--token", default=None)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--add-models", dest="add_models", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    cfg = load(args.config)
    if args.comfy_path:
        path = args.comfy_path.strip().strip('"')
        if not os.path.isdir(path):
            print(f"[ERROR] 目录不存在：{path}")
            return 2
        if not os.path.isfile(os.path.join(path, "comfy", "sd.py")):
            print(f"[ERROR] 这个目录里没有 comfy\\sd.py：{path}")
            print("        请指向包含 comfy 文件夹的 ComfyUI 源码根目录")
            return 2
        cfg["comfyui_dir"] = clean_path(path).replace("\\", "/")
        print(f"[OK] comfyui_dir = {cfg['comfyui_dir']}")
        found = scan(path)
        for k in SCAN_SUBDIRS:
            if found[k]:
                print(f"     扫描 {k}: {len(found[k])} 个文件（例：{found[k][0]}）")
        if args.add_models:
            auto = auto_models(path)
            added = merge_models(cfg, auto)
            print(f"[OK] 自动写入模型条目：{added} 个")
    if args.port is not None:
        cfg.setdefault("server", {})["port"] = args.port
    if args.token is not None:
        cfg.setdefault("server", {})["token"] = args.token

    if args.python:
        py = os.path.normpath(args.python.strip().strip('"'))
        if not os.path.isfile(py):
            print(f"[ERROR] python 不存在：{py}")
            return 2
        with open(os.path.join(HERE, "_python.txt"), "w", encoding="utf-8") as f:
            f.write(py)
        print(f"[OK] 已记录 ComfyUI 的 python：{py}")

    if args.write:
        save(cfg, args.config)
        print(f"[OK] 已写入 {args.config}")

    if args.show or not args.write:
        print("-" * 62)
        print(summary(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
