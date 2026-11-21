# comfyui_remote_clip_vae（ComfyUI 侧插件）

装到出图那台电脑的 `<ComfyUI>/custom_nodes/` 下。把它整体拷过去，重启 ComfyUI 即可。
通常直接运行上一层的 `3_安装插件到ComfyUI.bat` 更省事（会自动写入服务器地址）。

节点分类：**`comfyui_remote_clip_vae`**

## 节点

| 节点 | 输出 | 说明 |
|---|---|---|
| 远程 CLIP 加载器 (Remote CLIP Loader) | `clip` | 句柄：服务端地址 + 模型名，本机不占显存 |
| 远程 VAE 加载器 (Remote VAE) | `vae` | 同上 |
| 远程 CLIP 文本编码 (Remote CLIP Encode) | `正向条件` / `负向条件` | 一个节点出两路 CONDITIONING，直接接 KSampler；**只有文本输入，不接图片** |
| 远程 CLIP 图文编码 (Remote CLIP Encode + Images) | `正向条件` / `负向条件` | 文本 + 参考图（`图片1…图片10` 接一张自动多一个空槽）；接了 `vae` 就同时编 `reference_latents` |
| 远程 VAE 解码 (Remote VAE Decode) | `图像` | 服务端解码，回传 PNG |
| 远程 VAE 编码 (Remote VAE Encode) | `latent` | img2img / 重绘 |

两个「加载类」节点字段顺序一致：**服务端显卡 / 计算设备 / 服务端地址 / 模型**
（CLIP 加载器多一个「模型类型」），接着是「卸载模型」和整宽按钮 **「🔄 刷新模型列表」**。

- **服务端显卡**：只读，点刷新自动填服务端**显卡型号**（只有型号，不带 CUDA/ROCm 这类计算方式，也不带显存）；
  服务端是纯 CPU 时显示「（无显卡，纯 CPU 计算）」，连不上显示「（连不上服务端）」
- **计算设备**：只读，点刷新自动识别远端到底在用**显卡**还是 **CPU** 算（值就是「显卡」/「CPU」）；
  以前这个位置是让你手选 `auto / cpu / gpu`，v1.7.32 起取消手选，完全自动
- **模型**：服务端 `config.json` 里的条目名；连不上时是「（连接失败，检查服务端）」，ComfyUI 照常能启动
- **模型类型**：`自动（按文件名判断）` = 用服务端 `clip_type`；也可手动指定 `flux` / `sd3` / `qwen_image` …（仅 CLIP 加载器）
- **v1.7.32 起所有节点的「手动模型名」已彻底删掉**（加载器 v1.7.24 起就没有，文本编码 / 图文编码也在这一版删掉）：
  模型只从「🔄 刷新模型列表」拉下来的下拉里选；老工作流里残留的那个值一律忽略，不会报错

### 文本编码节点（v1.7.12 起只有文本）

- `正向提示词` / `负向提示词` 各一个多行框，输出 `正向条件` / `负向条件` 两路，直接进 KSampler / 采样器
- 可选输入只有 `clip`（接「远程 CLIP 加载器」）；原来的 `device`（auto/cpu/gpu）已删除（v1.7.32），
  条件张量一律跟原生节点一致放在本机模型设备上，不用再选
- **v1.7.7~v1.7.11 那套 `图片模式` + `图片1…图片32` + `vae` / `视觉模型` / `参考图像素` 已整套移除**，
  节点上不会再出现图片接口，前端也不会自动增槽。需要图文混编请改用 ComfyUI 原生
  `TextEncodeQwenImage21`（配 `qwen3vl_8b_bf16`），本包的加载器照样能给它供 `clip` / `vae`。

### 图文编码节点「远程 CLIP 图文编码」（v1.7.14 起，v1.7.15 上限 10 张）

- 显示名 `远程 CLIP 图文编码 (Remote CLIP Encode + Images)`，字段与文本编码一致，可选输入多一个 `vae`
- `参考图最大像素`（v1.7.35 起）：**单位百万像素（MP）**，默认 1.00（≈1000×1000），只缩不放、宽高对齐 32；
  `0` = 保持原尺寸。老工作流里那个「参考图像素」(1048576 这种) 仍然认，会直接按像素数走
- `图片1…图片10`：**接一张自动长一个槽**（前端 `refresh.js` 动态加，最多 10）
- 接了 `vae` → 参考图同时进视觉槽并编 `reference_latents`（`keep_vision=False`，等同原版 2.0 编辑 Plus）
- 不接 `vae` → 只走视觉槽（`keep_vision=True`）

### 提示词优化节点「远程提示词优化」（v1.7.16 起，v1.7.17 改走服务端 GGUF）

- 输入（v1.7.33 现状）：`用户提示词` / `使用远程模型` / `服务端地址` / `API Key` / `文生图模型` / `图生图模型` /
  `视觉模型` / `参考图最大像素`(MP) / `输出语言` / `透明背景` / `种子值`（右侧带「固定值/递增值/递减值/随机值」）/
  `卸载模型` + `图片1…图片10`
- **v1.7.33：「种子值」和「种子模式」合成一个控件**：就是 ComfyUI 原生的 seed + control 样式
  （`control_after_generate`）——数值框右边一个按钮，点开是 固定值 / 递增值 / 递减值 / 随机值，默认「随机值」
- 输出：`提示词` / `负向提示词` / `是否反推`（有图无词 = 反推；有图有词 = 图生图；无图 = 文生图）
- `接口类型` 三选一（默认第一项）：`RemoteCLIPVAE 服务端 (用远端 ComfyUI 里的 GGUF)` / OpenAI 兼容 / Ollama 原生
- 选第一项时，模型在**服务端机器的 `<ComfyUI>/models/LLM`**，由服务端那份 ComfyUI 的 python 加载；
  服务端面板能看「LLM 运行实时信息」、能一键装 llama-cpp-python

## 文件

| 文件 | 作用 |
|---|---|
| `nodes.py` | 7 个节点（CLIP 加载 / VAE 加载 / 文本编码 / 图文编码 / 提示词优化 / VAE 解码 / VAE 编码） |
| `api.py` | 本机代理路由 `/remote_clipvae/probe`（浏览器跨域会被挡，刷新按钮走它） |
| `web/refresh.js` | 「🔄 刷新模型列表」按钮、显卡栏只读、改动地址自动重探（无图片槽逻辑） |
| `codec.py` | 张量与图像的序列化协议，**必须与 `model_server/codec.py` 完全一致** |
| `remote_api.py` | 访问服务端的 HTTP 客户端，只用标准库 |
| `settings.py` + `default_server.json` | 默认服务器地址（`3_` 脚本会写入） |

## 默认服务器地址

`default_server.json`：

```json
{ "url": "http://192.168.1.20:8199", "token": "" }
```

也可用环境变量覆盖：`REMOTE_CLIPVAE_SERVER`、`REMOTE_CLIPVAE_TOKEN`。

## 注意

- `REMOTE_CLIP` / `REMOTE_VAE` 只是字典句柄（地址 + 模型名），**不占显存**
- 下拉框的模型名默认来自服务端 `config.json`；服务端重扫后点「🔄 刷新模型列表」即刻生效，不用重启 ComfyUI
- 服务端没启动时下拉框显示「（连接失败，检查服务端）」，ComfyUI 仍能正常启动
- 修改 `codec.py` 后必须同步 `model_server/codec.py`，否则协议对不上
- 如果浏览器缓存了旧版 `refresh.js`，按 `Ctrl+F5` 强刷一次 ComfyUI 页面
