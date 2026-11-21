# RemoteCLIPVAE —— 局域网双机 ComfyUI 分布式（CLIP 文本编码 + VAE 编解码放另一台机）

```
出图电脑：ComfyUI 主程序                      服务端电脑：本包的服务端
┌──────────────────────────────┐   HTTP    ┌────────────────────────────────┐
│ Load Diffusion Model (UNet)  │   JSON    │  CLIP 文本编码器（T5 / CLIP-L/G）│
│ KSampler ────────────────────┼─ prompt ─▶│  VAE（编码 + 解码）              │
│ 远程 CLIP 文本编码 ◀──────────┼─ COND ────│  直接调用 ComfyUI 的 comfy.sd    │
│ 远程 VAE 解码 ◀───────────────┼─ PNG ─────│  惰性加载 + LRU + 空闲释放        │
└──────────────────────────────┘           └────────────────────────────────┘
```

- 出图电脑的显存**全部留给图片生成模型**，不再放 CLIP 和 VAE
- 文本编码结果是纯张量，跨机传输**无损**（float32 原样，不压 8bit）
- 图像走 PNG 压缩（1024×1024 约 1~2 MB），千兆局域网单张往返 20~40 ms
- 服务端**不另写一套 CLIP 实现**，而是调用 服务端电脑 ComfyUI 的 `comfy.sd`，所以和 本机 UNet 完全同源
---

## 本次更新 v1.7.36 —— 修掉「vae 与 clip 指向不同服务端」这个硬报错

报错原文：

```
ValueError: 「vae」与「clip」指向不同服务端，请用同一台服务端的 CLIP / VAE 加载器
```

**为什么会报**：图文编码节点真正干活的只有 **CLIP 那一台机器**，VAE 只是把「模型名」丢给那台机器去加载，
两边地址根本不需要字符串完全相等。v1.7.35 之前用的是 `rstrip("/")` 字面比较，所以下面这些**同一台机器**也被判成"不同"：

| 会被误判的写法 | 举例 |
|---|---|
| 一边带 `http://` 一边不带 | `http://111.180.124.155:8199` vs `111.180.124.155:8199` |
| 末尾多一个 `/` | `http://x:8199/` vs `http://x:8199` |
| 前后有空格 / 大小写不同 | ` HTTP://x:8199 ` |
| VAE 加载器的地址还留着 `<未填>` 占位 | `<请填服务端地址>` |

**现在怎么处理**：
1. 地址先规范化（去 `http(s)://`、去末尾 `/`、去空格、转小写、去 `:80/:443`）再比；只是格式不同 → 直接放行。
2. VAE 加载器地址是空 / `<…>` / `（…）` 占位 → 自动沿用 CLIP 的服务端，不再报错。
3. 真是指向另一台机器 → **不拦了**，只在日志里提示一行，然后照 CLIP 那台去加载同名 vae：
   ```
   [远程CLIP图文编码] 注意：「远程 VAE 加载器」填的是 http://127.0.0.1:8199，CLIP 用的是 http://111.180.124.155:8199；
   本次仍按 CLIP 的服务端加载 vae=「qwen_image_vae.safetensors」。若那边没有这个 vae，把 VAE 加载器的地址改成同一个（或留空）即可。
   ```
   那台机器扫不到这个 vae 时，服务端会明确报「找不到 vae 模型 xxx」，比现在这个含糊的 ValueError 好定位。

### 你这边怎么弄（任选一步就能跑）

1. 把「远程 VAE 加载器」的 **服务端地址** 改成和 CLIP 完全相同（推荐，最省事）；
2. 或者那个 VAE 加载器的地址 **直接留空**（VAE 会自动跟着 CLIP 走）；
3. 更新到 v1.7.36 后，即使两边写法不一样也不会再被这个 ValueError 卡住。

> 提醒：VAE 模型必须能在 **CLIP 那台机器的模型目录** 里被扫到（同名），图文编码是那台机器算完再回传条件的。

### 本版实测

```
_norm_url 规范化比较:  http://111.180.124.155:8199/ == 111.180.124.155:8199  → True
                       " HTTP://111.180.124.155:8199 " == http://111.180.124.155:8199 → True
                       http://111.180.124.155:8199 != http://127.0.0.1:8199 → False（真不同，只提示不报错）
模拟 clip=111.180.124.155:8199 / vae=127.0.0.1:8199 → 打印提示后照常发起请求（不再 ValueError）
py_compile(nodes.py) = 0
```

---

## 本次更新 v1.7.35 —— 「远程 CLIP 图文编码」的参考图尺寸也改成百万像素（MP）

和「提示词优化」统一：原来那栏是 `参考图像素`（INT，默认 1048576 这种绝对像素数，不好读），
现在改成一模一样的那一套：

```
参考图最大像素   [ −  1.00  + ]    ← 单位：百万像素（MP）
```

- 默认 **1.00 MP**（≈1000×1000），只缩不放、宽高对齐 32；`0` = 保持原尺寸
- 换算：`总像素 = MP × 1,000,000` → 服务端 `resolution = √总像素`，再按 32 对齐缩放
  （1.00 → 1000²，0.4 → 632²≈400K，2.0 → 1414²≈2M）
- 这栏同时决定**文本编码器 vision 槽**和 **VAE reference_latents** 的缩放（还是官方那套保证 2×2 slot）
- **老工作流兼容**：残留的 `参考图像素`（1048576 这种像素数）仍然会被读，数值不走样
- 日志里会打：`[远程CLIP图文编码] 图 2 张 | 参考图最大像素=1MP（1000000 px）| vae=xxx`

### 本版实测

```
RemoteCLIPImageEncode required: … / 参考图最大像素 / 卸载模型
参考图最大像素 opts: {"default": 1.0, "min": 0.0, "max": 32.0, "step": 0.05}
py_compile(nodes.py) = 0
```
---

## 本次更新 v1.7.34 —— 底层超时限制全部取消（客户端 + 服务端 + LLM 子进程三层都没有上限）


上一版只是把「超时时间」这一栏从节点上删掉了，**底层还在按老值砍**（CLIP/VAE 120s、600s，
提示词优化 160s 一路传到服务端 LLM 子进程）。这一版把三层全改成"无限等"：

| 位置 | 以前 | 现在 |
|---|---|---|
| `remote_api.rpc` / `health` / `unload` / `llm_unload` | 120 ~ 600s | `timeout=None`（阻塞直到服务端回话） |
| `RemoteCLIP.timeout` / `RemoteVAE.timeout` | 120 / 600 | `None` |
| 提示词优化 → 服务端 `llm_chat` | 节点 160s + rpc 280s | 两层都 `None`，`params.timeout=None` |
| 服务端 `llm.py::_call` | `Event.wait(600)` 抛"LLM 子进程超时" | `Event.wait(None)`，一直等 worker 回 |
| 服务端 `llm.py::chat` / `backend.py` llm_chat | `timeout=600` | `None` |
| 外部 llama-server 客户端 `LLMServerClient` | 600 | `None` |

于是：大 CLIP 首次加载、超大图 VAE 解码、AMD 那台机视觉编码 2 分半、LLM 慢慢吐 2000 token，
都**不会中途被砍**。要停就手动在 ComfyUI 里点「取消」。

> 代价说明：如果服务端真的卡死/断网，节点会一直等（不再自己报"超时"）。取消队列即可。
> 只保留两个例外（都是秒级健康探测，不影响出图）：启动时枚举模型清单 `list_models` 4s、
> 前端「🔄 刷新模型列表」的 probe 5s。

### 本版实测

```
rpc.timeout default    = None
health.timeout default = None
unload / llm_unload    = None
RemoteCLIP.timeout attr = None
RemoteVAE.timeout attr  = None
optimizer 已无固定超时 ✓
VAE 解码 已无固定超时 ✓
py_compile(nodes.py / remote_api.py / llm.py / backend.py) = 0；exe --smoke = OK
```
---

## 本次更新 v1.7.33 —— 所有节点的「超时时间」删掉；「种子值 + 种子模式」合成原生 style 的一个控件

### 1）「超时时间」整栏删除（7 个节点全删）

| 节点 | 内部固定超时 |
|---|---|
| 远程 CLIP 加载器 / 远程 VAE 加载器 | 5s（连服务端的 health 探测） |
| 远程 CLIP 文本编码 / 远程 CLIP 图文编码 | 120s |
| 远程 VAE 解码 / 远程 VAE 编码 | 600s |
| 提示词优化 | `DEFAULT_TIMEOUT`（160s） |

时间都够用（服务端首次加载大 CLIP 也在这之内）；老工作流里残留的 `超时时间` 值仍然会被读（`kw.get("超时时间")`），
所以旧图不会因为少一栏而报错。要改默认值就改代码里的常量，不用再占节点一行。

### 2）「种子值」+「种子模式」合成一个控件（ComfyUI 原生 seed style）

以前是两栏：`种子值`（INT）+ `种子模式`（随机 / 固定）。现在只留一栏，跟图二那种一模一样：

```
种子值  [ − 0 🎲 + ]   ← 右侧那个蓝色按钮点开 = 固定值 / 递增值 / 递减值 / 随机值
```

实现就是 ComfyUI 自己的 `"control_after_generate": True`（和 KSampler 的 `seed` 同一套前端逻辑）：
选「随机值」→ 每次排队后自动换种子；「固定值」→ 一直用这个数；「递增值 / 递减值」→ 每次 +1 / -1。

### 3）本版实测（`INPUT_TYPES()` 实测，7 个节点全过）

```
RemoteCLIPLoader      required: 服务端显卡 / 计算设备 / 服务端地址 / 模型 / 模型类型 / 卸载模型
RemoteVAELoader       required: 服务端显卡 / 计算设备 / 服务端地址 / 模型 / 卸载模型
RemoteCLIPTextEncode  required: 正向 / 负向 / 服务端显卡 / 计算设备 / 服务端地址 / 模型 / 模型类型 / 卸载模型
RemoteCLIPImageEncode required: 同上 + 参考图最大像素(MP)
RemotePromptOptimizer required: 用户提示词 / 使用远程模型 / 服务端地址 / API Key / 文生图模型 / 图生图模型 /
                               视觉模型 / 参考图最大像素 / 输出语言 / 透明背景 / 种子值 / 卸载模型
RemoteVAEDecode       required: vae / latent / 卸载模型        optional: 分块解码 / 分块尺寸 / 重叠 / 输出设备
RemoteVAEEncode       required: vae / 图像 / 卸载模型          optional: 分块编码 / 分块尺寸 / 重叠

种子值 opts: {"default": 0, "min": 0, "max": 4294967295, "control_after_generate": true, ...}
ALL OK
```

> 字段又少了两栏（`超时时间`、`种子模式`），老工作流的控件值还是会错位——照样建议把节点重新拖一遍。
---

## 本次更新 v1.7.32 —— 「手动模型名」全部删掉；`device` 不再手选，改成自动识别远端在算的是显卡还是 CPU；显卡栏只留型号

### 1）所有节点的「手动模型名」删掉

模型只有一个来源：点 **「🔄 刷新模型列表」** 后，「模型」下拉里的条目。文本编码 / 图文编码节点上那栏也删了
（加载器在 v1.7.24 已经删过）。老工作流里残留的 `手动模型名` 值**一律忽略**，不会报错、也不会被采用。

### 2）`device`（auto / cpu / gpu）改成「计算设备」只读栏，自动识别

以前两个编码节点末尾有个 `device` 下拉让你选条件张量放哪。现在：

| 以前 | 现在 |
|---|---|
| `device`：`auto` / `cpu` / `gpu` 手选 | `计算设备`：只读，点刷新自动从服务端读回，值只有 **显卡** / **CPU** |

判定依据是服务端 `/health` 里的 `device` + `device_name`：`cpu` 或压根没报到显卡型号 → **CPU**，否则 → **显卡**。
条件张量一律跟原生节点一致（本机模型设备），不用再选。

### 3）「服务端显卡」只显示型号

原来显示 `NVIDIA GeForce RTX 4070 12.0GB · CUDA`（型号 + 显存 + 计算方式）。现在只留型号：

- 有显卡：`AMD Radeon RX 5700 XT`（就这一串，不带 · CUDA / · ROCm，也不带显存）
- 纯 CPU 机器：`（无显卡，纯 CPU 计算）`
- 连不上：`（连不上服务端）`

### 4）本版实测（服务端 111.180.124.155）

```
probe clip -> {'ok': True, 'device': 'AMD Radeon RX 5700 XT', 'compute': '显卡'}  models=4
probe vae  -> {'ok': True, 'device': 'AMD Radeon RX 5700 XT', 'compute': '显卡'}  models=5
probe llm  -> {'ok': True, 'device': 'AMD Radeon RX 5700 XT', 'compute': '显卡'}  models=2
```

节点字段（实测 `INPUT_TYPES()`）：

```
RemoteCLIPLoader       required: 服务端显卡 / 计算设备 / 服务端地址 / 模型 / 超时时间 / 模型类型 / 卸载模型
RemoteVAELoader        required: 服务端显卡 / 计算设备 / 服务端地址 / 模型 / 超时时间 / 卸载模型
RemoteCLIPTextEncode   required: 正向提示词 / 负向提示词 / 服务端显卡 / 计算设备 / 服务端地址 / 模型 / 超时时间 / 模型类型 / 卸载模型
                       optional: clip
RemoteCLIPImageEncode  required: … / 参考图最大像素(MP) / 卸载模型   optional: clip / vae / 图片1…图片10
```

> 字段变多了（多出「计算设备」），老工作流的控件值会错位，**建议重新拖一个节点出来**，
> 或者把节点上这几个下拉重新选一遍再保存。
---


## 本次更新 v1.7.31 —— 模型只填「模型根目录」：按文件夹名自动归类 vae / clip / clip_vision / LLM；面板直接告诉你「谁在 CPU 里算」

### 1）配置区只剩一行：模型根目录（可加多个）

以前要分别填「CLIP 模型目录」「VAE 模型目录」「GGUF 模型目录」，填错一个节点里就空一片。现在改成：

```
┌ 模型根目录 ────────────────────────────────────┐
│ F:\Comfy-Desktop\ComfyUI-Shared\models        │  ← 可以加多行
│ D:\models2                                     │
└────────────────────────────────────────────────┘
      [添加目录…] [移除选中] [自动找]
```

规则（只按**文件夹名**判断，不看文件名）：

| 文件夹名叫 | 归到 |
|---|---|
| `vae` | VAE |
| `clip` / `text_encoder` / `text_encoders` | 文本编码器（CLIP） |
| `clip_vision` / `clipvision` | 视觉编码器（clip_vision） |
| `llm` / `gguf` | 提示词优化的 GGUF；文件名带 `mmproj` / `vision` / `vit` 的自动进「视觉投影」 |
| 其它（`loras`、`checkpoints`、`unet`、`diffusion_models`…） | 整棵跳过，不扫 |

- 一个根目录可以加多个；扫到同名文件会在名字后加 `_2`
- `diffusers<xxx>/vae/diffusion_pytorch_model.safetensors` 这种也算 VAE，条目名会自动带上父目录名（`vae_diffusion_pytorch_model`），不会出现一堆 `model`
- 填了根目录后，老的 `clip_model_dir` / `vae_model_dir` / `llm_model_dir` 不用再填；面板保存时会把它们从配置里删掉
- 服务端启动 / `/reload` 时会重新扫一遍，**加了模型不用重启**，点一下「扫描并加入模型」或「重读配置」就行
- 原来手写的 `models.clip/vae` 条目保留，扫描只会补没登记过的文件

配置写法（config.json，等价的两种）：

```json
"model_roots": ["F:/Comfy-Desktop/ComfyUI-Shared/models"]
```

```json
"model_roots": [
  "F:/Comfy-Desktop/ComfyUI-Shared/models",
  "D:/ComfyUI/models"
]
```

### 2）状态栏会说「多少张量还在内存里算」

加载完 GGUF 后，服务端从 ggml 的加载日志里把 `Vulkan0 compute buffer size` 和 `CPU compute buffer size` 抓出来，状态行多一段：

```
…… 上卡: offloading 31 repeating layers to GPU    缓冲: 显存 5.6G / 内存 4.4G    ⚠ 模型主体在显卡，但还有 4.4 GB 张量在内存里算（显存缓冲 0.3 GB）。带图做提示词优化时，视觉投影（mmproj）多半也在 CPU，这就是任务管理器里 GPU 利用率很低的原因。想快：① 优化时不带图（只用文生图 GGUF）；② 换 f16 版 mmproj（bf16 在 Vulkan 上容易回落 CPU）；③ 图先缩到 1024 以内再喂
```

### 3）你截图那台 AMD 机的结论（RX 5700 XT + Vulkan）

- `offloading 31 repeating layers to GPU` + 专用显存 7.4/8.0 GB → **模型权重确实在显卡上**，不是 CPU 版在跑
- `GPU 利用率 1%` + `CPU compute buffer size = 4484.10 MiB` + 日志里的 `encoding image 1/1 to input buffer (nx=1408, ny=1920)` → **慢在视觉编码**：1440×1920 的原图先在 CPU 里编码，`mmproj-bf16.gguf` 在 Vulkan 后端（RDNA1 没有 cooperative matrix）上几乎不参与算
- `flash attention is disabled` 是 Vulkan 后端的正常现象，不是配置错

### 4）抓到你那份 worker 日志后的定位（2026-11-22，带图那次实测）

你那次的日志里这几行是决定性的：

```
mtmd_backends: using Vulkan0 backend
warmup: WARNING: the mmproj graph uses unsupported operators by the backend
warmup:          the performance will be suboptimal
warmup:          list of unsupported ops (backend=Vulkan0):
warmup:          MUL_MAT: type = f32, ne = [8464 8464 16 1]
warmup:         SOFT_MAX: type = f32, ne = [8464 8464 16 1]
mtmd_context:    Vulkan0 compute buffer size =   285.29 MiB
mtmd_context:        CPU compute buffer size =  4484.10 MiB
encode: copying image 1/1 to input buffer (nx=1408, ny=1920)
image slice encoded in 148705 ms          ← 2 分 28 秒，全在 CPU
prompt eval time = 154292.50 ms / 2797 tokens (18.13 tokens per second)
        eval time =   6911.13 ms /  322 runs  (46.59 tokens per second)
```

**结论（这次不是猜的）**：

- 语言模型 31 层全在 Vulkan 卡上，生成 46.6 tok/s —— 这部分没问题
- 慢的是**视觉编码器**：mmproj 的注意力算子（`MUL_MAT f32 8464×8464×16`、`SOFT_MAX` 同形）在 Vulkan 后端**没有实现**，llama.cpp 只能把整张图的计算退回 CPU。CPU compute buffer 给到 4484 MiB 就是这块
- 一张 1408×1920 的原图 = **2640 个视觉 token**，f32 注意力对 token 数是平方级，于是单张图 148 秒。任务管理器里 GPU 利用率 1% 就是这么来的（那两分半卡在 CPU 上，卡基本闲着）

**这版的处理：缩图放到节点上，按「百万像素」调**

「提示词优化」节点原来没有这个设置，现在在参数区加了一项 **`参考图最大像素`（单位 MP，默认 0.4）**：

- 接进来的参考图在**本机节点这一侧**先等比缩小（只缩不放，宽高对齐 32，bicubic），再编码成 PNG 发给服务端 —— 服务端不再二次缩图（`llm_image_max_side` 保留但默认 0）
- 控制台会打一行：`[参考图] 1408×1920（2.70MP）→ 544×736（0.40MP）`，然后是 `[远程提示词优化] … 图 1 张（上限 0.4MP）`
- `0` = 不缩（原图直发）；觉得反推细节掉了就调到 `0.8 ~ 1.0`
- 实测尺寸对照（1408×1920 原图 = 2.70MP = 2640 视觉 token）：

| 参考图最大像素 | 实际发给模型的尺寸 | 视觉 token | 视觉编码耗时（该 AMD 机估算） |
|---|---|---|---|
| 0（不缩） | 1408×1920 | 2640 | 148 秒（实测） |
| 0.2 | 384×512 | 192 | 约 1 秒 |
| **0.4（默认）** | **544×736** | **391** | **约 2~3 秒** |
| 0.8 | 768×1056 | 792 | 约 10 秒 |
| 1.0 | 864×1152 | 972 | 约 18 秒 |

原因还是上面那段：mmproj 的 f32 注意力在 Vulkan 后端没有实现，会整段退回 CPU，耗时随 token 数平方涨；所以缩图是这台机唯一便宜的解法。


三条能立刻见效的做法（按性价比排）：

1. **节点上「参考图最大像素」保持 0.4**（本版默认，最稳）
2. **优化时不带图**：只挂「文生图模型」（`Qwen-Image-2.1-PE-T2I.Q5_K_S.gguf`），图那一路不接 → 预填量从 2797 tokens 掉到几百
3. **换 f16 版 mmproj**（`...mmproj-f16.gguf`）替掉 bf16：f32/bf16 混算在 Vulkan 上同样会回落，f16 通常少踩一点

> RX 5700 XT 是 gfx1010，AMD 官方 Windows 轮子只覆盖 gfx1030+，装 HIP 版通常起不来；这台就走 Vulkan。真正的根治要等 llama.cpp 给 Vulkan 补上 f32 大矩阵乘（日志里它让你去报 issue，就是这个意思）。

### 5）本版实测（本机）

```
[服务端] 模型根目录 1 个 → 自动登记 CLIP 3、VAE 6、clip_vision 0、GGUF 4、mmproj 2
list_models: clip 3 / vae 6 / clip_vision 0 / llm 1 / llm_vision 1
/reload → model_roots 1 个、scanned 9
_parse_buffers("Vulkan0 … 285.29 MiB / CPU … 4484.10 MiB") → (285.29, 4484.1)
节点缩图（fit_max_pixels，1408×1920 原图 2.70MP）：0.2→384×512(192tok)、0.4→544×736(391tok)、0.8→768×1056(792tok)、1.0→864×1152(972tok)，0=原样 2640tok
```

（本机 `D:\Comfy-Desktop\ComfyUI\ComfyUI\models` 下 GGUF 在 `LLM\GGUF\Qwen\<型号>\*.gguf`，是 4 层深，所以扫描深度从 3 放到 5，否则正好扫不到——你 F: 那几个目录也是这个结构。）

---

## 上一版 v1.7.30 —— 「到底是谁在算」说清楚：环境诊断 + 一键卸载 + 真跑分量速

你服务端那台 AMD 机我直接连上去测了，先把**事实**摆出来（`/llm_status` 原样）：

```
backend=vulkan   llama_cpp=0.3.37   offloaded="offloading 31 repeating layers to GPU"
device_name=AMD Radeon RX 5700 XT
短提示：prompt eval 84.9 tok/s，生成 48.0 tok/s
你那次真实请求（带 1 张图，2797 个 token）：prompt eval 18.07 tok/s → 一次 175 秒
```

也就是说：**生成确实在显卡上跑**（48 tok/s，纯 CPU 做不到这个数），
**卡住你的是"预填"那一段**——大段提示 + 图片编码在 Vulkan 后端上掉到 18 tok/s，
175 秒里绝大部分时间花在这。RX 5700 XT（gfx1010）没有 cooperative matrix，
Vulkan 的大 batch 矩阵乘只能走慢路径，这是卡本身的限制，不是没上卡。

这一版把"判断"交到你手里，三件事：

### 1）「环境诊断（装了几版 llama）」

点一下就把这个 ComfyUI 的 python 里的真实情况打出来：

- import 到的到底是哪份 `llama_cpp`（绝对路径 + 版本）
- pip 里**登记了几份** llama-cpp-python（>=2 份就是有重复安装）
- `llama_cpp\lib` 里到底带了哪些原生库 —— 有 `ggml-vulkan.dll`/`ggml-cuda.dll` 才是 GPU 版，
  只有 `ggml-cpu*.dll` 就是 **CPU 版**
- site-packages 里有没有**第二个 llama_cpp 目录盖在上面**（源码编译装过最容易留这个，导入的可能是那份 CPU 版）
- `C:\Windows\System32\vulkan-1.dll` 在不在（缺了它，Vulkan 版会静默回落到 CPU）
- torch 自带的 GPU 库、以及有没有人设过 `GGML_VK_VISIBLE_DEVICES` 之类环境变量

结论会直接写成一句人话：`CUDA 版（走显卡）` / `vulkan 版（走显卡）` / `CPU 版（只在内存里算）`。

### 2）「卸载 llama-cpp-python」

你之前装过 CPU 版，最怕的就是"两份并存、导入到旧那份"。这个按钮做两遍：

1. `pip uninstall -y llama-cpp-python`（再补一刀 `llama_cpp_python`）
2. 再扫 site-packages，把 pip 删不掉的**残留目录**（`llama_cpp\`、`llama-cpp*`、`*.dist-info`）
   统统删掉，删完再 import 一次确认真没了

会先停掉服务、杀掉 LLM worker（否则 `.pyd` 被占用删不掉）。**卸载完必须再点
「装 Vulkan 版（AMD·免SDK）」把带 GPU 后端的版本装回来**，否则 GGUF 加载不了。

> 实测（本机临时 venv）：先正常卸掉 `0.4.7+cu130` → `removed=[]`、`still_importable=[]`；
> 再手工造一个假的 `llama_cpp` 残留目录 → 一键卸完自动 `rmtree` 掉，`still_importable=[]`。

### 3）「GPU 跑分（预填/生成）」

让服务端真跑一次（512 token 预填 + 64 token 生成），读 llama.cpp 自己的
`llama_perf_context_print` 行，给出两个数：

- **PP（预填）tokens/s**：长提示、带图时主要看这个
- **生成 tokens/s**：一般 30+ 就是显卡在算；低于 10 基本就是 CPU 版

本机 RTX 5070 实测（同一个 `Qwen-Image-2.1-PE-T2I.Q5_K_S.gguf`）：
`PP 2484.6 tok/s · 生成 85.8 tok/s · offloading 31 repeating layers to GPU → 结论：显卡在算（生成速度正常）`。

### 你现在怎么做（针对"计算还在 CPU"）

1. 面板点「环境诊断」。
   - 如果写 `CPU 版` 或提示有 **残留 llama_cpp 目录 / 多份登记** → 点「卸载 llama-cpp-python」，
     再点「装 Vulkan 版（AMD·免SDK）」，完了点「加载 GGUF 到显存」+「GPU 跑分」复核。
   - 如果写 `vulkan 版（走显卡）` 且跑分生成 ≥30 tok/s → **其实已经在卡上算了**；
     慢在预填（尤其带图）。这时能提速的只有换后端：点「装 HIP 版（AMD·免SDK）」
     （走 AMD 官方预编译 wheel，同样不用 SDK），装完再跑一次分对比。
2. 想确认"有没有真的用显卡"：**别看任务管理器**——Windows 任务管理器的 GPU 占用统计
   对 Vulkan/ROCm 计算不可靠；只看「GPU 跑分」的两个数字和 `上卡: offloading N layers to GPU`。

> 路线不变：GGUF 还是跑在服务端那个 ComfyUI 的 python 里、和 CLIP/VAE 共用 8199 一个端口，
> 不另开 llama 端口；卸载/重装只是换 `llama_cpp` 的构建（cpu / vulkan / hip），不改这套架构。

---

## 上一版 v1.7.29 —— 修你实机上的两个故障：状态栏一直是「—」、GGUF 扫不到

你截的两张图我都能对上原因，都是**服务端这一侧**的，不是你装错、也不是 AMD 卡的问题：

### 故障一：面板「运行状态」全显示 —（不更新）

面板每 3 秒 GET 一次 `/health`，超时设的是 **4 秒**。而你服务端这台机 `/health` 每次要 **4.09–4.15 秒**
（我直接从公网 `111.180.124.155:8199` 连过去测了多次，稳定这个数）——因为 `/health` 里塞了完整模型清单，
要去 `F:\Comfy-Desktop\ComfyUI-Shared\models` 这个共享盘上 stat 上千个文件。

结果就是：**每次都刚好超时 → 返回 None → `apply_health` 永远进不去 → 标签停在默认的「—」。**
服务本身一点问题没有，是"响应比超时慢 90 毫秒"。

改法：

- `app.py`：`/health` 支持 **`?light=1`**（`/`、`/status` 同）。轻量模式下 `models` 返回 `{}`，
  不再扫盘；`/models` 反而改成 `ttl=0` 强制拿最新清单。
- `backend.py`：`status(light=False)`；`list_models(ttl=4.0)` 加了 4 秒缓存，连着请求不会反复扫盘。
- `llm.py`：`LLMEngine.status(light=...)`——轻量时连 GGUF 目录都不枚举（`MockLLM` / `ExternalLLM` 同样接受 `light` 参数，不再 TypeError）。
- `gui.py`：`_http_get(path, timeout=4)` 加形参；`_poll_health` 改打 `**/health?light=1`**，超时放到 **8 秒**。

你机上实测该修好后 `/health?light=1` 只剩 **890 字节量级**（改造前 4007 字节），耗时从 4 秒级掉到毫秒级。

### 故障二：节点扫得到 CLIP/VAE，扫不到 GGUF

两个叠加的原因：

1. **你机上的 `config.json` 里还留着 v1.7.26 写进去的 `"llm_external": "http://127.0.0.1:8300/v1"`。**
   只要这一行在，`Backend` 就会走 `ExternalLLM`——它的 `models` 恒为空，而 v1.7.27 起面板已经没有
   「外部 LLM 服务地址」这个输入框了，可旧值被回填、又被 `_collect_cfg` 原样写回，于是**永久自锁**：
   GGUF 清单永远是空，插件端就提示"服务端没扫到 GGUF"。
2. 即使忽略 external，旧代码也只扫 `<comfyui_dir>/models/LLM*`。**你的模型实际在
   `F:\Comfy-Desktop\ComfyUI-Shared\models\...`（隔壁目录），不在 ComfyUI 自己的 models 下**，所以照样扫不到。

v1.7.29 三层一起清 + 自动发现共享目录：

- `backend.py`：`__init__` 直接 `pop` 掉 `llm_external`，只建 `LLMEngine`，并打印
  `[服务端] 已忽略 config.json 里残留的 llm_external …`。
- `config.py::load()`：读配置时就丢掉这一项（`[配置] 已丢弃残留的 llm_external …`）。
- `gui.py`：不再回填 `var_llm_ext`，`_collect_cfg` 无条件 `pop` —— 老配置**不用你手改**，新包自己清。
- `llm.py`：`_model_roots()` 从 ComfyUI 目录**向上两层**枚举兄弟目录，凡是含 `models` 的都算
  （`F:\Comfy-Desktop\ComfyUI-Shared` 就是这么被找到的）；`llm_dirs()` 再在每个根下找
  `LLM / llm / llm_gguf / LLM_gguf / gguf / text_encoders_llm` 里真实存在的那个，按 `normcase` 去重
  （Windows 上 `LLM` 和 `llm` 是同一个目录，不会出现两份）。面板的「GGUF 模型目录」下拉用同一套逻辑。
- 清缓存：`/reload` 会同时清掉 CLIP/VAE 与 GGUF 的清单缓存。

修好后 `F:\Comfy-Desktop\ComfyUI-Shared\models\LLM` 自动出现在「GGUF 模型目录」里，
`.gguf` 进「GGUF 模型」、`mmproj-*.gguf` 进「视觉模型（mmproj）」，不用手选。

### 你现在要做的（三步）

1. 服务端那台 AMD 机：**先点「停止服务」**，把旧的 `RemoteCLIPVAE服务端.exe` 换成新包里的同名文件
   （`config.json` 可以留着，里面的 `llm_external` 新包会自己清掉——想干净也可以直接删掉这一行）。
2. 再「启动服务」，打开面板等 3 秒：状态栏应变成 `运行中（PID …）· 端口 8199 · 已运行 Ns`，
   设备行出 `AMD Radeon RX 5700 XT`，显存条有数。
3. 看「GGUF 模型目录」是否已经指向 `F:\Comfy-Desktop\ComfyUI-Shared\models\LLM`。
   若还是空，说明你的 `.gguf` 放在别的路径 → 在那个下拉里手动选一次，点「刷新模型列表」，
   服务端会 `/reload` 重读，插件端节点下拉立刻就有。
   （浏览器直接验：`http://111.180.124.155:8199/models` 的 `llm` 里能看到 `xxx.gguf` 就说明通了。）

### 本版实测（本机 NVIDIA + 假目录复刻你的 F: 结构，ComfyUI venv python）

| 项目 | 结果 |
|---|---|
| 残留 `llm_external` 时后端类型 | `LLMEngine`（不再是 ExternalLLM） |
| `llm_dirs()` | `…\Comfy-Desktop\ComfyUI-Shared\models\LLM`（一条，无 `llm` 重复项） |
| `status(light=True)` | 0.07s，`models` 为 `{}` |
| `status()` 全量 | 0.001s 命中缓存，`llm=['Qwen3.5-9B-Q4.gguf']`、`llm_vision=['mmproj-Qwen3.5-9B-f16.gguf']` |
| HTTP `GET /health?light=1` | 0.069s / **890 B**（改造前同机 2534 B 且含全量清单） |
| HTTP `GET /models` | 0.001s / 942 B，`llm` 与 `llm_vision` 均有条目 |
| `py_compile` | 6 个文件全过；`config.example.json` JSON 合法 |

> 说明：GGUF 仍然跑在**服务端那个 ComfyUI 的 python** 里（`llm_worker.py` 子进程 import `llama_cpp`），
> 和 CLIP/VAE **共用 8199 一个端口**，不另开 llama 端口——这条路线没变。

---

## 上一版 v1.7.28 —— AMD 卡不用装任何 SDK：直接上官方预编译 Vulkan wheel

v1.7.27 那两个按钮（`装 Vulkan 版` / `装 HIP 版`）前提是"这台机能编译"——要 Vulkan SDK 或 ROCm/HIP SDK，
再加上 MSVC C++ 和 CMake。你机器上这些都没有，所以一点就秒失败。现在改成**根本不编译**：

- llama-cpp-python 官方 wheel index 除了 `cuXXX`，还有两条 AMD 通道：`whl/vulkan` 和 `whl/hip-radeon`。
  里面是 `llama_cpp_python-0.3.37-py3-none-win_amd64.whl`——**CPython 3.8+ 通用**（你那个 3.13.12 / cp313 直接用），
  包内自带 `ggml-vulkan.dll`，不需要 SDK、不需要编译器、不需要 ROCm。
- 点 `装 Vulkan 版（AMD·免SDK）` 后做的事：抓 index 页面 → 选最新 wheel → **多线路断点续传下载 42MB**
  → 装进你那个 ComfyUI 的 python → 校验 `llama_cpp\lib` 里真出现 `ggml-vulkan`，再报"后端 = vulkan"。
  四条线路依次是：GitHub 直连、`gh-proxy.com`、`ghfast.top`、`ghproxy.net`；每拉 8MB 就主动断开重连（国内长连接必断），
  下了多少存成 `.part`，**再点一次就从断点接着下**。下好的 wheel 缓存在 `%LOCALAPPDATA%\RemoteCLIPVAE\wheels`。
- `装 HIP 版（AMD·免SDK）` 同理，走 `hip-radeon` 通道。AMD 卡一般用 Vulkan 就够，HIP 只是备选。
- 两条路都只有"下载失败 **且** 本机确实装了 SDK"时才退回去现场编译；连 SDK 都没有时给明确提示 + 手轮子的直链，
  不再提 llama-server / 外部服务地址那些已下线的说法。
- `安装 llama-cpp-python`（普通那条）在 ROCm torch 的机器上也改成先走 Vulkan 预编译 wheel，不再往 AMD 机上装 cuXXX 轮子。

架构没动：GGUF 仍然由服务端那个 python 拉起 `llm_worker.py` 加载，**和 CLIP/VAE 共用 8199，不另开端口**。

| 你的情况 | 点哪个 | 前置 |
|---|---|---|
| AMD 显卡（含 RX 5700 XT / gfx1010），不想装 SDK | **装 Vulkan 版（AMD·免SDK）** | 无（联网下 42MB wheel） |
| AMD 显卡，想试 HIP 后端 | **装 HIP 版（AMD·免SDK）** | 无（同上，走 hip-radeon 通道） |
| NVIDIA | `安装 llama-cpp-python` | 无（自动匹配 cu 预编译 wheel） |

装完点 `检测计算后端`，显示 `计算后端: VULKAN（ggml-vulkan）` 就是真上卡了；
再 `加载 GGUF 到显存`（`n_gpu_layers=-1` 全部层上卡），状态行会写 `后端: vulkan`，设备名是你的 AMD 卡。

> 注意：这台是 NVIDIA 测试机，本地 5070 上验证的是"wheel 能下、能装进 cp313、能被探测成 vulkan 后端"；
> 真机 RX 5700 XT 上的出字速度要以你那边实测为准（gfx1010 走 Vulkan 通常比同一张卡的 CPU 快数倍）。

### 实测记录（本机 cp313 空 venv，就是按你服务端的环境复刻的）

```
选中：llama_cpp_python-0.3.37-py3-none-win_amd64.whl
下载 40MB → 4 条线路轮流、断了 3 次、每次都从断点续上 → 42,201,883 字节下完，zip 校验通过
pip install → llama-cpp-python 0.3.37（cp313 装 py3-none 包，没有任何 "not a supported wheel"）
后端探测：{"backend":"vulkan","libs":[...ggml-vulkan.dll...],"devices":["NVIDIA GeForce RTX 5070 (NVIDIA)"],"offload":true}
llm_worker：ping ok / load 3.3s / 生成 96 tokens ≈ 9.8 tok/s / unload ok（同时修掉了 unload 参数不匹配的老 bug）
```


## 上一版 v1.7.27 —— GGUF 就跑在服务端自己这个进程里，**不另开 llama 端口**

先说清楚架构（这也是它本来该有的样子）：GGUF **不是**另起一个服务、另占一个端口。
服务端（监听 8199 的那个进程）会用它自己那个 ComfyUI 的 python 拉起 `llm_worker.py`，
llama-cpp-python 就在里面加载 GGUF；提示词优化节点发的每个请求，都是 `POST http://<服务端IP>:8199/rpc`
（`op=llm_chat`）走过去。**和 CLIP / VAE 完全同一个端口、同一份配置、同一个 `/health`。**

所以「装完还是 CPU」跟端口无关，只跟那个 python 里的 llama-cpp-python 是 CPU 版有关。
这一版把这件事彻底改了：

- 面板去掉了「外部 LLM 服务地址 / 下载 llama-server / 启动 llama-server」这几行（不再引导你另开 8300 端口）；
  只留一条说明：**GGUF 由服务端自己的 python 加载，走同一个 8199 端口**。
- 「提示词优化 LLM」区块现在是：`加载 GGUF 到显存` / `卸载 GGUF` / `安装 llama-cpp-python` / `装本地 wheel…`
  + **`装 Vulkan 版（AMD 显卡）`** / **`装 HIP 版（AMD ROCm）`** / `检测计算后端`。
- 识别 ROCm：如果那个 python 的 torch 是 `+rocm`（`torch.version.hip` 有值），
  `安装 llama-cpp-python` 会直接告诉你「没有 HIP 预编译 wheel，cuXXX 装了也只能 CPU」并**不再乱装 cu 轮子**。
- 本地 wheel 会先核对 Python 版本：`cp314` 的 wheel 往 `cp313` 的 python 里装，现在直接拦下来提示换版本
  （你日志里那个 `llama_cpp_python-0.4.7+cu132-cp314-...whl` + `Python 3.13.12（cp313）` 就是这个坑）。

### AMD 显卡怎么让它真的上卡（v1.7.28 起：不用装 SDK）

| 你的 python | 该点哪个 | 前置 |
|---|---|---|
| 有显卡驱动，没装任何 SDK（你目前这台就是这样） | **装 Vulkan 版（AMD·免SDK）** | 无，只联网下 42MB 预编译 wheel |
| 想试 HIP 后端 | **装 HIP 版（AMD·免SDK）** | 无，走官方 `hip-radeon` 预编译 wheel |
| 上面的 wheel 下不来、但本机真装了 SDK | 同上按钮（下载失败后会自动退回现场编译） | Vulkan SDK / ROCm HIP SDK + MSVC + CMake |

（v1.7.27 里"必须装 SDK 才能编"的说法已被 v1.7.28 的预编译 wheel 取代。）

编译是**一次性**的（5~20 分钟），装完 `llama_cpp\lib` 里会出现 `ggml-hip.dll` / `ggml-vulkan.dll`；
`检测计算后端` 显示 `计算后端: HIP（ggml-hip）` 或 `VULKAN（ggml-vulkan）` 就说明成了。
之后 `加载 GGUF 到显存` → 状态行里 `后端: hip`，`n_gpu_layers=-1`（全部层上卡），提示词优化就走 8199 用显卡算。

> 状态行会同时显示 `设备`（ROCm torch 下就是 AMD 卡名）和 `后端`；如果装的还是 CPU 版，
> 会追一行 `⚠ 这个 llama-cpp-python 是 CPU 版…`，一眼能分辨。

`config.json` 里的 `llm_external` 仍然保留（想接现成的 llama-server 可以手动填），但面板不再引导，默认不填。

实测（本机 CUDA 机）：`py_env_info` 现在回 `tag/cuda/hip/devices`（本机 `cp313 / 13.0 / 空 / RTX 5070`）；
cp314 wheel 装进 cp313 python → 提前拦下并提示；`llm.py` / `gui.py` / `backend.py` 编译通过；
`RemoteCLIPVAE服务端.exe` 重打后 `--smoke` 全 OK。

## 上一版 v1.7.26 —— AMD 显卡（RX 5700 XT）终于能用显卡跑提示词优化

**为什么「安装 llama-cpp-python」装完还是 CPU**：那一步只会找 CUDA 的预编译 wheel
（JamePeng / abetlen 的 index），找不到就退到纯 CPU 版。RX 5700 XT 是 **gfx1010（Navi10 / RDNA1）**，
Windows 上没有 CUDA —— 所以**装不出带 GPU 的 llama-cpp-python**，必须用 Vulkan 或 HIP 后端从源码编一次。
v1.7.26 加了 Vulkan 编译入口和「检测计算后端」；v1.7.27 又补上 HIP 路线，并把多余的 llama-server 端口去掉了。

## 上一版 v1.7.25 —— 「🗑 卸载模型」真能还显存了（含 exe 重打）

**为什么之前点了没用**：服务端把 CLIP/VAE 对象从自己的缓存里删掉就完事了，但 ComfyUI 的
`comfy.model_management` 里还挂着同一份模型（`current_loaded_models`），PyTorch 的缓存分配器也没归还，
所以 `nvidia-smi` 照旧显示占着。**删缓存 ≠ 卸显存**。

现在服务端 `unload` 做三件事：

1. 从 `_loaded` 缓存里删掉对应条目；
2. **无条件**调 `model_management.unload_all_models()`（这才是把 ComfyUI 那份模型真正放开的地方；
   以前只有“缓存里没东西”时才调，所以正常情况永远走不到）；
3. `synchronize` + `torch.cuda.empty_cache()` + `ipc_collect()` + `soft_empty_cache(force=True)`，让分配器也归还。

**LLM（提示词优化）**：卸载不再只是让 worker `del` 模型 —— 直接杀掉 worker 进程。因为进程里的 CUDA 上下文
本身就会占一截显存（几百 MB），只 del 模型，`nvidia-smi` 里那块还在；杀进程才彻底干净，下次用会自动重启 worker。

**能看到结果**：卸载接口现在回 `vram_before` / `vram_after`，节点上的弹框会显示
`服务端显存：1.9 GB → 0.4 GB（共 12 GB）`，控制台也会打一行 `[远程卸载] … | 显存 XGB → YGB`。

> 这次改的是**服务端代码**（`backend.py` / `llm.py`），所以 `RemoteCLIPVAE服务端.exe` 已重新打包，
> 直接用新 exe 覆盖旧的即可（配置 `config.json` 不用动）。

实测（本机 RTX 5070 12GB，强制走我们新的释放路径）：
`allocated 400MB → 0MB`、`reserved 400MB → 0MB`、整卡 `used 9.22GB → 1.17GB`（剩下的 1.17GB 是桌面/其他进程）。

## 上一版 v1.7.24 —— 两个加载器删掉「手动模型名」

「远程 CLIP 加载器」和「远程 VAE 加载器」不再有 `手动模型名` 这一栏：模型只从 `模型` 下拉选（下拉由 `🔄 刷新模型列表` 从服务端拉）。

- 现在这两栏的顺序：**服务端显卡 / 服务端地址 / 模型 / 超时时间 /（CLIP 加载器多一项 模型类型）/ 卸载模型** + 整宽刷新按钮。
- 老工作流兼容：执行侧仍然接 `手动模型名`（万一有值就优先用）；只是画布上不再显示，也不再参与新节点。
- 「远程 CLIP 文本编码 / 图文编码」两个节点**仍保留** `手动模型名`（服务端列不出清单时靠它手工点名），需要一起删的话说一声。
- 服务端列不出模型时，报错文案改成：先在服务端面板选好目录并扫描 → 点「🔄 刷新模型列表」→ 在「模型」下拉里选一个。

## 上一版 v1.7.23 —— 本地模式也用「文生图模型 / 图生图模型 / 视觉模型」这三栏

之前本地模式是另外两栏「本地模型 / 本地视觉模型」，切过去等于要重新选一遍、还容易选错。现在**取消那两栏**，两边共用同一套三栏，只是候选来源跟着开关换：

| 开关 | 文生图模型 / 图生图模型 | 视觉模型 | 额外显示 |
|---|---|---|---|
| 开（远程） | 服务端 `<ComfyUI>/models/LLM` 的 GGUF | 服务端 mmproj | 服务端地址 / API Key / 🔄刷新 / 🗑卸载 |
| 关（本地） | **本机** `<ComfyUI>/models/LLM` 的 GGUF | 本机 mmproj（只有一个时自动选上） | 无（地址与按钮自动收起） |

- 切到本地时前端会打本机新路由 `GET /remote_clipvae/local_models`（扫 `models/LLM`、`models/llm`、`models/gguf` 等），把三个下拉的候选换成本机文件；按文件名自动分流：带 `t2i` 的进「文生图模型」、带 `i2i/edit/img2img` 的进「图生图模型」，其余两边都列。
- **两边各自记住自己的选择**：来回拨开关，远程选的服务端 GGUF 和本地选的本机 GGUF 互不覆盖。
- 本机 `models/LLM` 里一个 GGUF 都没有时，下拉直接显示中文提示（含扫描过的目录），不会让你白跑一次出图。
- 执行侧：无图用「文生图模型」、有图用「图生图模型」+「视觉模型」，和远程完全同一套判定（有图无词=反推 / 有图有词=图生图 / 无图=文生图）。

## 上一版 v1.7.22 —— 提示词优化：双输出口 + 本地/远程开关

**1. `json` 输出口拆成两个 STRING 口**

| 输出 | 内容 |
|---|---|
| **提示词** | 正向提示词（纯文本，不含任何 json 引号） |
| **负向提示词** | 负向提示词（模型没给就是空串，透明背景不会污染它） |
| **是否反推** | 接了图但没写词 = True |

所以现在可以直接 `提示词 → CLIP 文本编码(正向)`、`负向提示词 → CLIP 文本编码(负向)`，不用再拆 json 字符串。

**2. 新增「使用远程模型」开关**

| 开关 | 谁跑 LLM | 摸哪个目录 | 需要的栏位 |
|---|---|---|---|
| **开（远程，默认）** | 服务端 ComfyUI 的 python 拉 `llm_worker.py` 子进程 | 服务端 `<ComfyUI>/models/LLM` | 服务端地址 / API Key / 文生图模型 / 图生图模型 / 视觉模型（+ 刷新、卸载按钮） |
| **关（本地）** | 本机 ComfyUI 的 python 直接 `llama_cpp.Llama` | 本机 `<ComfyUI>/models/LLM` | 本地模型 / 本地视觉模型 |

- 开关一拨，**另一半栏位在画布上立刻收起**（不会误填、不会报「模型名还是占位符」）；工作流里存的值不会丢，拨回来还在。
- 本地模式也认绝对路径：`本地模型` 下拉里没有的，直接把 `.gguf` 全路径填进去也认（按 basename 兜底搜 `models/LLM` 等目录）。
- 本地模式接了图但没选「本地视觉模型」（mmproj）→ 直接中文报错，不会静默降级成纯文本。
- 本机没装 llama-cpp-python 时，报错会写明：在服务端面板点「安装 llama-cpp-python」，或本机 `pip install llama-cpp-python`（CUDA 版用面板里的 JamePeng wheel）。
- 本地模式的「卸载模型」四档走**本机**：卸载 = `del` + `gc` + `torch.cuda.empty_cache()`；「运行前后自动卸载」= 跑前先放掉上一个，跑完再放。
- 本地模式跑完会在控制台打 `[提示词优化-本地] model=... | mmproj=... | 图 n 张 | 文生图/图生图/反推 | seed | tok`。

## 上一版 v1.7.21 —— 每个远程节点都加「卸载模型」下拉（4 档）+ 「🗑 卸载模型」按钮

| 节点 | 卸载谁 |
|---|---|
| 远程 CLIP 加载器 | 服务端那份 CLIP |
| 远程 VAE 加载器 | 服务端那份 VAE |
| 远程 CLIP 文本编码 | 编码用的那份 CLIP（连了 clip 端口就是上游那份） |
| 远程 CLIP 图文编码 | 同上 |
| 远程 VAE 解码 / 远程 VAE 编码 | 上游「远程 VAE 加载器」那份 VAE（地址和模型从连线自动取） |
| 提示词优化 | 服务端的 GGUF（llama-cpp 子进程退出） |

四档语义（和截图一致，默认「自动」）：

| 档位 | 行为 |
|---|---|
| **自动** | 不主动卸，交给服务端按需加载 / LRU / 空闲释放 |
| **保持加载** | 明确不卸（服务端一直留着，下次调用最快） |
| **运行后自动卸载** | 这个节点跑完就把对应模型踢出服务端显存（失败也照样卸，放在 `finally` 里） |
| **运行前后自动卸载** | 跑之前先卸一次（清掉上一个占显存的东西），跑完再卸一次 |

点「🗑 卸载模型」按钮是**立刻**让服务端卸掉该节点当前选中的那个模型，弹框会回显服务端释放了哪些（`（服务端本来就没加载）` 表示本来就空）。
服务端新增 `unload{kind, model}`、CLIP/VAE 侧走 `/rpc`，前端走本机代理 `GET /remote_clipvae/unload?url=&kind=&model=`。
token 已不在节点上：这个代理路由会自动补 `default_server.json` 里的 token。

## 上一版 v1.7.20 —— 「远程 CLIP 文本编码 / 图文编码」：`clip` 端口一连，本节点的模型字段自动收起

- 「远程 CLIP 文本编码 (Remote CLIP Encode)」和「远程 CLIP 图文编码 (Remote CLIP Encode + Images)」
  左边那个 **`clip` 输入端口**连了东西，就**不再用本节点自己的模型设置**（服务端显卡 / 服务端地址 / 模型 /
  手动模型名 / 模型类型 这 5 栏会在画布上自动收起，值也不参与计算），编码完全交给连过来的那个 clip：
  - 连的是「远程 CLIP 加载器」→ 用它带的地址、模型、clip_type（本节点里填的地址/模型一律忽略）
  - 连的是本机原生 CLIP（CLIPLoader / DualCLIPLoader 等）→ 本机直接用它的 `tokenize + encode_from_tokens_scheduled`，
    图文编码节点会把 `图片1…图片10` 以 `images=` 传给它（有 vae 就 `keep_vision=False`，没 vae 就 `True`，和 Qwen-Image 节点一致）
- 拔掉连线，5 栏立刻恢复显示，回到「用本节点自己的字段」的老行为；工作流里存下的这些值不会丢。

## 上一版 v1.7.19 —— 全节点删除 `token` 与「自定义模型名」

- 5 个使用服务端的节点都去掉了 `token` 栏，改成内部自动取 `default_server.json` / 侧边栏设置里的 token（没填就是空）。
- 「提示词优化」去掉 `自定义模型名`（两个模型下拉 + 刷新按钮已够用）；`手动模型名` 保留，作为服务端扫不到模型时的兜底。

## 上一版 v1.7.18 —— 「提示词优化」节点按参考节点重排，并加 服务端地址 / API Key / 刷新模型列表

节点长相现在和你截图里的「Qwen image 2.1 提示词优化」一致，从上到下：

```
图片1 … 图片10（接一张自动多一个空接口）
用户提示词
服务端地址
API Key
🔄 刷新模型列表   ← 整宽按钮，插在 API Key 正下方
文生图模型
图生图模型
视觉模型
输出语言        自动 / 中文 / English
透明背景        ← 开关
种子值
种子模式        随机 / 固定
卸载模型        自动 / 保持加载 / 运行后自动卸载 / 运行前后自动卸载
```

- **不再有**「接口类型 / 模式 / 上下文长度 / 温度 / 最大输出长度 / 超时时间」这些栏目，默认值直接固定：
  走服务端 GGUF（`接口类型=RemoteCLIPVAE`）、`n_ctx=4096`、`温度=0.7`、`最大输出=2048`、`超时=600s`。
  老工作流里带着这些旧控件也照样能跑（有值就用，没值就用默认）。
- **有图无词 = 反推**、**有图有词 = 图生图**（用「图生图模型」+「视觉模型」）、**无图 = 文生图**（用「文生图模型」），自动判断，不用再选「模式」。
- 接了图却没选「视觉模型」（mmproj）→ 执行阶段直接中文报错拦住，不会静默丢图。
- 「卸载模型」= 服务端显存策略：`自动` 按需加载/换模型才换；`保持加载` 跑完不卸；`运行后自动卸载` / `运行前后自动卸载` 省显存但下次要重新加载（走服务端 `llm_unload`）。
- 「文生图模型」「图生图模型」两个下拉由服务端 `models/llm` 一起刷新；「视觉模型」读 `models/llm_vision`（`（不使用 / 纯文本）` + mmproj 列表）。
- 服务端扫不到模型时，用 optional 里的「自定义模型名」手填 GGUF 文件名即可覆盖（这也让 OpenAI/Ollama 老用法还有退路）。

## 上一版 v1.7.17 —— 「提示词优化」改成直接用**服务端 ComfyUI 里的 GGUF**（模型不出服务端）

v1.7.16 是把提示词丢给远端的 Ollama / LM Studio / llama.cpp server；这一版改成**服务端自己加载 GGUF**：
模型文件就放在服务端机器的 `<ComfyUI>/models/LLM`，由**服务端那份 ComfyUI 的 python**（已带 torch 和 llama-cpp-python）加载，
本机和出图电脑都不用装任何东西，也不用手动开 Ollama。CLIP / VAE / clip_vision 的老接口完全不受影响。

### 1. 节点「接口类型」现在是三选一

| 接口类型 | 谁在算 | 需要什么 |
|---|---|---|
| **RemoteCLIPVAE 服务端 (用远端 ComfyUI 里的 GGUF)** ← 默认 | 服务端 exe 拉子进程，用 `<ComfyUI>/models/LLM` 里的 GGUF | 服务端 python 里有 `llama-cpp-python`（面板一键装） |
| OpenAI 兼容（vLLM / LM Studio / llama.cpp server） | 远端 HTTP 服务 | 一个 `/v1/chat/completions` 服务 |
| Ollama 原生 | 远端 Ollama | `ollama serve` + 已 pull 的模型 |

- 「模型名」下拉框会直接从**服务端**拉 GGUF 列表（文件名带 `mmproj` / `vision` / `vit` 的自动归到「视觉模型」）
- 「视觉模型」下拉 = `（不使用 / 纯文本）` + 服务端的 mmproj 列表；选 `（不使用 / 纯文本）` 就是纯文本优化
- 模型放在别的盘 → 服务端面板「GGUF 模型目录」填路径；或 `config.json` 写 `llm_model_dir`
- 节点里改完地址 / 点刷新按钮，下拉会立刻重拉；队列校验不再拿启动时的旧列表拦人（`VALIDATE_INPUTS` 一律放行，真错误在执行阶段用中文报）

### 2. 服务端 exe 面板新增「提示词优化 LLM」

> 就是双击 `RemoteCLIPVAE服务端.exe` / `0_启动图形界面.bat` 打开的**那个窗口**（tkinter 控制面板），
> 在「服务配置」里多了 `GGUF 模型目录` 一行，在「运行状态」下面多了一个 `提示词优化 LLM（GGUF，跑在服务端这台机上）` 区块。

- **GGUF 模型目录** 输入框（默认 `<ComfyUI>/models/LLM`，也自动认 `llm / gguf / LLM_gguf / text_encoders_llm`）
- 两个列表：**GGUF 文本模型** / **视觉投影 mmproj**，选中即高亮
- 按钮：`加载 GGUF 到显存`、`卸载 GGUF`、`安装 llama-cpp-python`（对 ComfyUI 的 python 跑 pip，逐行日志）
- 上下文长度 `nctx`（默认 4096）
- **LLM 运行实时信息** 三行：
  - `状态 / 模型 / mmproj / 上下文 / handler / 加载耗时 / llama-cpp 版本 / 设备`
  - `生成中：tokens · tok/s · 秒 · 图数 · 时间｜Python｜设备`（每 8 个 token 刷新一次）
  - `错误 / 提示`

HTTP 侧多了 `GET /llm_status`（给面板和前端轮询），RPC 侧多了 `llm_models / llm_status / llm_load / llm_unload / llm_chat`。

### 3. 服务端放模型、装依赖

1. GGUF 丢进 `<ComfyUI>/models/LLM`（视觉用的 mmproj 也丢同一个目录）
2. llama-cpp-python 装在**跑服务端的那个 python** 上：
   - 启动服务端用 **`4_用ComfyUI的Python启动服务端.bat`**（推荐，ComfyUI 的 venv 一般已经装好 llama-cpp）
   - 或面板点「安装 llama-cpp-python」：会自动读这个 python 的 **cp 版本** 和 **torch 的 CUDA 版本**，
     去 `JamePeng/llama-cpp-python` 的 release 里挑**对应的预编译 wheel**（如 `cp313 + cu130 → v0.4.7-cu130-win`），
     `--no-deps` 装进去，**不走源码编译**（Windows 上编译几乎必定失败）；
     匹配不到再退回 abetlen 官方 index（同样只接受预编译 wheel，最后退 CPU 版）
   - 公司网络下不了 GitHub → 手动去
     <https://github.com/JamePeng/llama-cpp-python/releases> 下 `cp版本-cu版本-win_amd64.whl`，
     再点面板上的 **「装本地 wheel…」**
   - exe 内置的 python 装不了（onefile 里没有编译环境）
3. 面板里点「加载 GGUF 到显存」，状态变「已加载」即成功（首次 6GB 模型约 5~8 秒；RTX 5070 实测 4.45s / 7.8s）
4. 回 ComfyUI，节点「模型名」选刚加载的那个，跑一次 —— 日志里会打印 tokens / tok/s

### 4. 实测（本机 RTX 5070 12GB，真 GGUF）

| 项 | 结果 |
|---|---|
| 模型 | `local-agent-quality.gguf` 6.02GB + `mmproj-Qwen3.5-9B…BF16.gguf` 0.86GB |
| 加载 | 4.45s / 7.8s（面板按钮 / RPC 直接调），`llama-cpp-python 0.4.3+cu130` |
| 纯文本 | 3.1s，225 tok，73.77 tok/s |
| 带图反推 | 2.7s，192 tok，71.91 tok/s（正确描述成「抽象数字像素艺术」） |
| 面板链路 | `/api/llm_load` → 12s 后 `/api/state` 显示 `state=已加载`、`device_name=RTX 5070`、`pid`、`stderr` 全对 |
| ComfyUI 提交 | `/object_info` 字段顺序正确、`/prompt` → `node_errors: {}`、`/history` status success |

**注意**：这一版对模型输出加了 `response_format={"type":"json_object"}` 语法约束（不支持的版本自动去掉重试），
所以模型不会再吐「思考过程」或半截非 JSON；个别「agent 类」小模型仍会原样复述你的提示词，换个提示词优化专用的 GGUF（如 Qwen-Image-2.1-PE）效果最好。

---

## 上一版 v1.7.16 —— 新增「提示词优化 (Prompt Optimizer, Remote)」：模型跑在远端，本机零依赖

照你给的「Qwen image 2.1 提示词优化」节点做的，但**不再在本机加载 GGUF**：节点只发 HTTP，
模型在远端机器上（Ollama / llama.cpp server / LM Studio / vLLM 都行），本机不用装 llama-cpp-python，
服务端 exe 一行没改。

### 1. 节点长这样

```
                       ┌─────────────────────────────────────────────┐
图像1 ─────────────────▶  用户提示词   [多行文本框]                    │
图像2 ─────────────────▶  服务端地址   http://10.10.10.2:8080        │
…（接一张自动多一个）      接口类型     OpenAI 兼容 / Ollama 原生        │──▶ prompt（优化后的提示词）
图像10 ◀────────────────  模型名       qwen3-vl-8b                    │──▶ json（模型原始 JSON）
                          API Key      （留空不鉴权）                  │──▶ 是否反推（BOOLEAN）
                          模式         自动 / 文生图 / 图生图 / 反推
                          输出语言     自动 / 中文 / English
                          透明背景     false
                          种子值 0 / 种子模式 随机
                          温度 0.70 / 最大输出长度 1024 / 超时时间 180
                       └─────────────────────────────────────────────┘
```

- 图片接口 `图片1…图片10`，**接一张自动多一个空接口**（和「远程 CLIP 图文编码」同一套前端逻辑）
- 三个输出和原节点一致：`prompt` / `json` / `是否反推`
- 走的是 OpenAI 兼容 `POST /v1/chat/completions`（文本 + `image_url` 的 Base64 PNG），
  选 `Ollama 原生` 则走 `POST /api/chat`（`messages.images`）

### 2. 模式怎么判（和原代码一致）

| 情况 | 实际模式 |
|---|---|
| 有图 + 用户提示词为空 | 反推（看图写提示词） |
| 有图 + 有用户提示词 | 图生图（改写编辑指令） |
| 无图 | 文生图 |
| 手动选「文生图 / 图生图 / 反推」 | 按你选的来 |

`输出语言=自动` 时跟随用户提示词的语言（提示词为空/全中文 → 中文）；`透明背景` 打开时
system prompt 里追加「背景全透明、保留 alpha」，并在结果末尾补一句透明背景要求。

### 3. 远端要准备什么

只要一个能跑 Qwen3-VL / Qwen-Image-PE 之类视觉模型的服务，任选其一：

```powershell
# llama.cpp server（推荐，模型就是原来那个 GGUF + mmproj）
llama-server -m Qwen-Image-2.1-PE-I2I.gguf --mmproj mmproj.gguf --host 0.0.0.0 --port 8080
# 节点填：服务端地址 http://<服务端IP>:8080，接口类型 OpenAI 兼容，模型名留空或随意

# Ollama
ollama serve
ollama pull qwen3-vl:8b
# 节点填：http://<服务端IP>:11434，接口类型 Ollama 原生，模型名 qwen3-vl:8b

# LM Studio：Developer → Start Server（默认 1234），接口类型 OpenAI 兼容
```

GGUF 原来放本机的话，把它拷到远端那台机器就行（本节点不再读本机 `models/LLM`）。

### 4. 不认 OpenAI 兼容的老接口怎么办

节点会直接把远端返回的报错抛到 ComfyUI 控制台（`远端 LLM 返回 HTTP 4xx：<body>`），
常见原因：模型名空着、地址少写 `http://`、防火墙没开端口、服务端只开了 `/api/generate`。

### 5. 实测（本机，mock 远端 LLM 8198）

- 5 组用例全过：文生图 / 图生图+透明背景 / 反推(2 图) / Ollama 原生 / 单节点多帧（batch=2 → 2 张图）
- 断言的请求内容：`model=qwen3-vl-8b`、`seed=1234`（固定种子）、`text part=请反推这张图的提示词。`、
  `image_url` 为 `data:image/png;base64,…`、system prompt 随模式切换
- `/object_info`：required 13 项 + optional 11 项（`图片1..图片10` + `token`），共 7 个节点注册
- `py_compile` / `node --check` 通过；服务端未改，无需重打 exe

---

## 上一版 v1.7.15 —— 图文编码节点的图片上限：32 → 10

「远程 CLIP 图文编码」的图片接口上限从 32 张改成 **10 张**（`nodes.py` 的 `IMAGE_SLOTS = 10`、
`web/refresh.js` 的 `IMG_MAX = 10`），行为不变：接一张自动多一个空槽，最多到 `图片10` 就不再长。

- 旧工作流里如果接了 11 张以上的图，多出来的连线会失效，需要重接；10 张以内完全不受影响
- 服务端没改，`RemoteCLIPVAE服务端.exe` 沿用 v1.7.13/1.7.14 那个

---

## 上一版 v1.7.14 —— 新增「远程 CLIP 图文编码 (Remote CLIP Encode + Images)」

原来的「远程 CLIP 文本编码」继续保持不变（纯文本、四项输入、两路条件）。
新增一个**并列的独立节点**，专门用来「文字 + 参考图」：

**`Remote CLIP 图文编码 (Remote CLIP Encode + Images)`**

| 项 | 说明 |
|---|---|
| 正向提示词 / 负向提示词 | 和原节点一样，出 `正向条件` / `负向条件` |
| `clip`（可选） | 接「远程 CLIP 加载器」 |
| `vae`（可选） | 接「远程 VAE 加载器」；**接上就会同时编 reference_latents**（Qwen-Image-Edit 的做法） |
| 参考图最大像素 | **百万像素(MP)**，默认 1.00（≈1000×1000）；图会按它等比缩放到 32 的倍数；0 = 原尺寸 |
| `图片1 … 图片32` | **接一张自动多出一个空接口**，拔掉自动收回去，最多 32 张 |

两种用法自动区分：
- 接了 `vae` → 服务端把同一张图一边进文本编码器 vision 槽、一边编成 `reference_latents` 挂到条件上，`keep_vision=False`；
- 没接 `vae` → 只当 vision 输入，`keep_vision=True`。
- 一张图都不接 → 自动退化成纯文本编码（和原节点等价）。

对照原生节点：它等价于 `TextEncodeQwenImage21` 的「正向/负向」两路输出（但**不输出 latent**，
所以你原来用「空Latent图像 (SD3)」的做法不用改）。原生的 2.1 节点是把参考图塞进 `image_1…image_16`；
我们这个节点用的是 ComfyUI 老式动态输入 + 前端增槽（`web/refresh.js` 的 `syncImgSlots`），
老版本 ComfyUI 也能用，且「显卡只读 / 刷新模型列表 / 地址改后自动重探」全部继承。

### 实测（本机 ComfyUI 0.44.2 + mock 服务端）

- `nodes.py` `py_compile` 通过；`refresh.js` `node --check` 通过
- `/object_info/RemoteCLIPImageEncode`：required = 正向提示词 / 负向提示词 / 服务端显卡 / 服务端地址 / 模型 / 模型类型 / 参考图最大像素(MP)；
  optional = `clip` / `vae` / `token` / `device` + `图片1…图片32`（共 36 项，32 个图片槽），输出 `正向条件 / 负向条件`
- 提交 `RemoteCLIPLoader + RemoteVAELoader + 2×EmptyImage → 图片1/图片2 → 图文编码 → PreviewAny`：
  `node_errors: {}`、`status: success`，条件的 extra 里出现 **reference_latents** 和 `pooled_output`（说明 vae 那一路生效了）

---

## 上一版 v1.7.13 —— 修「本地 CLIP 出图正常、远程 CLIP 出图糊/重影」

真机现象：同一张工作流（原生 `文本编码 Qwen Image 编辑 Plus` + 参考图 + 远程 VAE），
把 CLIP 换成「加载 CLIP（本地）」图就正常，换成「远程 CLIP 加载器」就变成灰白叠影。
对比下来是**远程路径把提示词丢了**，并且 `keep_vision` 用了错的默认值。两处都修了：

### 1. 提示词被吃掉（真正的元凶）

`RemoteCLIP.tokenize()` 把文本塞在 `{"__remote_clip__": {"text": ...}}` 里，
而 `_text_of()` 只查顶层 `__remote_clip_text__` / `text` —— 于是 `encode_from_tokens_scheduled`
取到的文本是 **空字符串**。服务端收到「空提示词 + 2 张参考图」，
Qwen-Image 只能把参考 latent 硬平均着铺满整幅图 → 就是你看到的灰白、半透明、穿不上衣服。

修法：
- `tokenize()` 返回值同时带上顶层 `__remote_clip_text__`；
- `_text_of()` 增加「先读内层 payload.text」的分支；
- 实测「把图一的衣服穿到图二身上」原样送达服务端（`sent text match: True`）。

### 2. `keep_vision` 默认值不对（第二种叠影来源）

`comfy/text_encoders/qwen_image21.py` 里 tokenizer 的签名是
`tokenize_with_weights(..., keep_vision=False, ...)` —— **默认 False**。
原生「编辑 Plus」节点不传这个参数，所以本地是 False：vision token 被换成 `image_slots`，
参考信息只由节点自己挂的 `reference_latents` 提供，一份不多一份不少。

服务端 v1.7.9 起的推断是「本地没编 latent 就 keep_vision=True」，于是远程变成
**vision token 留一份 + reference_latents 再挂一份**，图像信息插了两次 → 叠影/糊。

修法：`_qwen_vl_conditioning` 在 `keep_vision is None` 时直接用 tokenizer 的原生默认 **False**，
不再自己推断；原生 2.1 节点显式传的值照旧原样透传。

### 3. 这次不用换节点

上一版建议你改用 `TextEncodeQwenImage21`，但你在本机用 **2.0 的「编辑 Plus」+ SD3 空 latent**
跑出来的图是正常的 —— 说明你这套 `qwen_image_2.1` UNet 就是吃 16 通道 latent 的，
那就别再动拓扑了，**只需要把服务端 exe 和插件换成这一版**。

### 4. 实测（本机）

- `nodes.py` / `backend.py` `py_compile` 通过
- 单测：`tokenize` 后 `_text_of` 返回完整原文（66 字符），发给服务端的 payload：
  `text` 与原文完全一致、`images` 2 张、`no_resize=True`、`llama_template` 透传、
  `keep_vision` 不再被强塞（交给服务端按默认 False 处理）
- `RemoteCLIPVAE服务端.exe` 重新打包并 `--smoke` 通过

### 5. 换新后怎么确认生效

跑一次远程 CLIP 的工作流，服务端日志应出现：

```
[服务端] 图文混编：2 张图 / 2 个 vision token / keep_vision=False（客户端自带 reference_latents，服务端不再编）
```

只要 `keep_vision=False` 且提示词不是空的，出图就应该和本地 CLIP 一致。
如果还是糊，把这段日志和 `[服务端] 加载 CLIP ... (clip_type=...)` 那行发我。

---

## 上一版 v1.7.12 —— 「远程 CLIP 文本编码」退回纯文本版（服务端一律不动）

### 1. 节点回到 v1.7.6 的样子

`Remote CLIP 文本编码 (Remote CLIP Encode)` 上**不再有任何图片接口**：

- 去掉 `图片模式` 下拉
- 去掉 `图片1 … 图片32` 这一堆 IMAGE 输入
- 去掉 `vae` / `视觉模型` / `参考图像素` 三个可选输入

现在只剩：正向提示词、负向提示词、服务端显卡（只读）、服务端地址、模型、超时时间、
手动模型名、模型类型 + 可选 `clip` / `token` / `device`，输出照旧「正向条件 / 负向条件」两路。

### 2. 前端脚本同步收口

`web/refresh.js` 里那套「接一张图就自动多出一个空槽」（`syncImgSlots` / `onConnectionsChange` 增删槽）
一并删掉，避免对着已经没有 `图片N` 输入的节点做无意义增删；刷新模型列表 / 显卡只读 / 地址改后自动重探全部保留。

### 3. 服务端 / 其他节点：一行代码都没改

`model_server`（`app.py backend.py codec.py config.py panel.py gui.py`）与
`RemoteCLIPLoader` / `RemoteVAELoader` / `RemoteVAEDecode` / `RemoteVAEEncode` 全部保持 v1.7.11 原样，
所以包里的 `RemoteCLIPVAE服务端.exe` 仍是上版那个，**无需重新打**（v1.7.11 的 clip_type 自动识别照常生效）。
服务端的 `clip_encode` 依然保留 images / keep_vision / 图文混编能力，只是客户端节点不再发图。
之前用旧工作流存过「文本编码节点接了图」的，打开后那几个图片连线会被丢掉，需要重连到原生编码节点。

### 4. 实测（本机）

- `nodes.py` `py_compile` 通过；`refresh.js` `node --check` 通过
- 临时 ComfyUI(8196) + mock 服务端(8197)：`/object_info/RemoteCLIPTextEncode`
  → required = 正向提示词 / 负向提示词 / 服务端显卡 / 服务端地址 / 模型 / 超时时间 / 手动模型名 / 模型类型；
  optional = `clip` / `token` / `device`（**无任何 IMAGE 输入**）
- 提交 `RemoteCLIPLoader → RemoteCLIPTextEncode → PreviewAny` 图：`node_errors: {}`、`status: success`

---

## 上一版 v1.7.11 —— 根因抓到了：`clip_type=STABLE_DIFFUSION`（Qwen-VL 被当成 SD1.5 加载）

### 1. 你日志里那一行是关键

```
[服务端] 加载 CLIP qwen3vl_8b_bf16: [...qwen3vl_8b_bf16.safetensors] (clip_type=CLIPType.STABLE_DIFFUSION)
[服务端] 警告：只命中 0/2 个 vision token
```

`clip_type` 是 `stable_diffusion` → ComfyUI 用 **SD1.5 的 CLIP/tokenizer** 去加载这个 Qwen3-VL 8B 文件，
tokenizer 根本不认 `<|image_pad|>`，图当然进不去，条件里 0 个 vision token。
之前面板「扫描」会把新加的 CLIP 条目统一写成 `"clip_type": "stable_diffusion"`，就是这个来源。

### 2. 修复：clip_type 自动识别（不再写死 stable_diffusion）

- 服务端加载 CLIP 时，如果条目没写 / 写的是占位默认 `stable_diffusion`，会**按权重内容 + 文件名**判断：
  - safetensors 参数名里有 `model.visual.*` / `visual.*`（Qwen-VL 系带视觉塔）→ `qwen_image`
  - 有 `conditioner.*` → `flux`
  - 文件名/条目名含 `qwen` 且含 `vl` / `qwen_image` → `qwen_image`
  - 都不匹配 → 保持原值（真 SD1.5 不受影响）
- 日志会明确告诉你：`[服务端] 自动识别 clip_type=qwen_image（qwen3vl_8b_bf16.safetensors 看起来是 qwen_image 文本编码器）`
- 0 个 vision token 时的警告也改成了直接指路（提示去核对 clip_type）
- 面板扫描 / config 自动生成**不再写死** `clip_type`，交给加载时判断

### 3. 你这次要做的

1. 服务端换新 exe（`RemoteCLIPVAE服务端.exe`），重新启动；
2. 再跑一次，日志应出现 `自动识别 clip_type=qwen_image` 和
   `图文混编：2 张图 / 2 个 vision token / keep_vision=False，挂 2 份 reference_latents`；
3. 如果还想手动钉死，也可以在 `server/config.json` 里给该条目加 `"clip_type": "qwen_image"`。

⚠️ 另外两条仍然成立（上一版说过）：
- 你的节点是 **Qwen-Image 2.0** 的 `TextEncodeQwenImageEditPlus`（vision 缩 384²、latent 缩 1024²），
  配 `qwen3vl_8b_bf16`（2.1）时槽位和 latent 对不齐，即使图进去了也会有糊感 —— 请改用
  原生 **`TextEncodeQwenImage21`**，或用我们的「远程 CLIP 文本编码」+ Qwen-VL 图文混编模式；
- **别用 SD3 的空 Latent** 给 Qwen-Image（通道数/1:16 缩放都不对），用编辑节点自带的 `latent` 输出。

### 4. 实测（本机）

- 识别逻辑单测：`qwen3vl_8b_bf16.safetensors` → `qwen_image`；`clip_l.safetensors` → 空（不动）；
  伪造 `model.visual.*` 头的 safetensors → `qwen_image`；伪造 SD1.5 头 → 空
- `backend.py / config.py / panel.py / gui.py` 全部 `py_compile` 通过

---
## 本次更新 v1.7.10 —— 修「用原生文本编码节点 → 出图糊」：images / keep_vision 被丢掉了

### 1. 根因

你在工作流里用的是 ComfyUI **原生**节点「文本编码 Qwen Image 编辑 Plus」
（`TextEncodeQwenImageEditPlus`），它内部是这样调我们给的 RemoteCLIP 代理的：

```python
tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
conditioning = clip.encode_from_tokens_scheduled(tokens)
```

而 v1.7.9 的 `tokenize()` 只留了文本，`images=` / `llama_template=` / `keep_vision=` **全被丢掉**，
于是服务端拿到的是「纯文字、没有图」的请求：

- vision 槽全是空的 → tokenizer 里没有 image token、`image_slots` 也没有；
- 但 VAE 那一路是正常的（`vae.encode()` 确实把参考 latent 编出来了）；
- 结果：DiT 收到一堆参考 latent，却不知道它们该插在序列的哪里 → 只能硬塞 → 出图糊/重影。

### 2. 修复

- `RemoteCLIP.tokenize(text, images=…, keep_vision=…, prevent_empty_text=…, llama_template=…)`
  现在把 **图片（PNG 编码，带实例内缓存，同一批图只编一次）**、`keep_vision`、`llama_template` 一起发给服务端；
- `encode_from_tokens_scheduled()` 发现是带图的 token 对象时，原样转发，并带 `no_resize=True`
  （原生节点已经把图缩好了，服务端不再二次缩放，避免尺寸被改）；
- 服务端 `clip_encode` 新增 `no_resize` / `keep_vision` / `prevent_empty_text` / `llama_template`：
  走的是 `clip.tokenize(text, images=…, keep_vision=…, llama_template=…)`，和原生节点完全一致。

服务端日志会打：`图文混编：N 张图 / N 个 vision token / keep_vision=False，挂 N 份 reference_latents`。

### 3. 还有一半原因：这个节点是 Qwen-Image **2.0** 的

`TextEncodeQwenImageEditPlus` 的缩放口径是 **vision 图缩到 384²，参考 latent 缩到 1024²**（2.0 的写法）。
你的 `qwen3vl_8b_bf16` 在 ComfyUI 里走的是 **Qwen-Image 2.1** 的 tokenizer
（`QWEN3VL_8B` → `qwen_image21.QwenImage21Tokenizer`），2.1 的要求是
**vision 和 latent 用同一套缩放，且都是 32 的倍数**，这样每个 vision slot 正好覆盖 2×2 个 latent。
384² 的 vision 配 1024² 的 latent，槽位和 latent 对不齐，画面就会糊、重影。

所以配 2.1 的模型，请用下面任一种：

| 用法 | 说明 |
|---|---|
| 原生 `TextEncodeQwenImage21`（节点名「文本编码 Qwen Image 21」） | 它是按 2.1 写的：vision/latent 统一缩放，还输出 positive/negative/latent 三路 |
| 我们的「远程 CLIP 文本编码」+ `图片模式 = 图片 → 服务端文本编码器（Qwen-VL 图文混编）` | 服务端内部已经按 2.1 官方口径缩放，接上「远程 VAE 加载器」就自动挂 reference_latents |

另外：**别用 SD3 的「空 Latent 图像」给 Qwen-Image**（通道数和 1/16 的缩放都对不上），
用上面这些节点自带的 `latent` 输出，尺寸按参考图来。

### 4. 实测（本机）

- `py_compile` 通过，已同步到本机 custom_nodes
- 直接调代理（模拟原生节点写法）：`clip.tokenize(带 "Picture 1: <|vision_start|>…" 的文本, images=[图1, 图2], llama_template=…)`
  → `tokenize` 载荷含 `images/keep_vision/llama_template/prevent_empty_text`，
  `encode_from_tokens_scheduled` 返回 `(1, 205, 768)`（77 + 64×2，**说明两张图真的到了服务端**）
- mock 服务端 + 临时 ComfyUI（8196）跑**原生节点** `TextEncodeQwenImageEditPlus`
  （clip=RemoteCLIPLoader、vae=RemoteVAELoader、image1/image2=EmptyImage）：
  `node_errors: {}`、`status: success`
- 真 Qwen3-VL 权重本机没有，最终清晰度要你在 10.10.10.2 上确认

---
## 本次更新 v1.7.9 —— 修正：Qwen-Image 编辑的图片走「文本编码器」，按官方 QwenImage21 节点实现

### 1. 上一版为什么会警告

v1.7.8 里我自己拼了 `<|im_start|>` 对话模板，还自己判断「图有没有变成 vision token」。
你那个 `qwen3vl_8b_bf16` 在 ComfyUI 里走的其实是 **Qwen-Image 2.1 的 tokenizer**
（`comfy/sd.py`：`clip_type=qwen_image` 且 `te_model=QWEN3VL_8B` → `qwen_image21.QwenImage21Tokenizer`），
它的机制和 2.0 不一样：

- 模板由 tokenizer 自己生成，多图时它会把 `<|vision_start|><|image_pad|><|vision_end|>` 按图数重复；
- 编码后如果**没有** VAE latent，需要 `keep_vision=True` 把 vision token 留在条件里；
- 如果**有** VAE latent，需要 `keep_vision=False` —— vision token 被摘掉，换成 `image_slots`（告诉 DiT 第几个位置插参考 latent）。

上一版既手拼模板又没传 `keep_vision`，还用了错误的 token 结构去判断，所以打了一条误报警告。

### 2. 现在的做法：完全照抄官方节点

`comfy_extras/nodes_qwen.py`（Qwen-Image 2.1 的参考图文本编码节点）就是这么干的，服务端现在一模一样：

1. 每张参考图按 `resolution²` 等比缩放到 **32 的倍数**（`resolution = √参考图像素`，默认 1024²）；
2. 同一张图既给文本编码器（vision 槽），又在服务端用 VAE 编一份 latent；
3. `clip.tokenize(text, images=…, keep_vision=len(ref_latents)==0, prevent_empty_text=True)`；
4. 有 latent 就 `reference_latents` 追加进条件（和官方 `conditioning_set_values(..., append=True)` 等价）。

所以现在：

- **接了「远程 VAE 加载器」** → 条件里带 `image_slots` + `reference_latents`，DiT 把参考 latent 拼进序列（**官方编辑用法**）
- **没接 VAE** → `keep_vision=True`，图像特征直接留在文本条件里（轻量用法）

服务端日志会直接告诉你走了哪条、以及建议的采样尺寸：

```
[服务端] 图文混编：2 张图 / 2 个 vision token，挂 2 份 reference_latents（采样尺寸建议 1184x896）
[服务端] 图文混编：2 张图 / 2 个 vision token，未接 VAE → keep_vision=True（图像特征直接进条件）
```

采样尺寸请按这条日志的建议来（官方节点也强调：采样尺寸偏离参考图尺寸，编辑会跑偏）。

### 3. 参数变化（节点上不用动）

- 「参考图最大像素」(MP) 含义不变（默认 1.00 = 1000²≈1MP），现在它就是官方的 `resolution`，同时决定文本编码器和 VAE 的缩放，保证每个 vision slot 正好覆盖 2×2 个 latent。
- 图文混编不再需要你在节点里手动挂 VAE 输出——「远程 VAE 加载器」接上就自动带上（服务端会自己加载那个 VAE）。

### 4. 实测（本机）

- `py_compile` 通过；`nodes.py` / `codec.py` 已同步到本机 custom_nodes
- 逻辑单测（伪 clip/vae 替身）：两图 + VAE → tokenize 收到 `keep_vision=False`、条件含 `image_slots` + 2 份 `reference_latents`；不接 VAE → `keep_vision=True`
- mock 服务端 + 临时 ComfyUI（8196）：`图片模式=Qwen-VL 图文混编` + VAE 接上 → `node_errors: {}`、`status: success`，客户端日志 `附带 ['reference_latents']`
- 真 Qwen3-VL 权重本机没有，tokenizer 的真实路径需要你在 10.10.10.2 上再跑一次

### 5. 顺带

- 服务端控制台/网页面板里的「A 端 / B 端」字样统一改成「客户端 / 服务端」。

---
## 本次更新 v1.7.7 —— 文本编码支持输入图片（自动增槽）+ 服务端图片解码提速

### 1. 「远程 CLIP 文本编码」多了图片接口，接一张自动长一个

- 节点新增 **`图片模式`** 下拉（默认「关闭（只编码文字）」，不接图时完全和以前一样）：

| 图片模式 | 服务端干什么 | 适合的模型 |
|---|---|---|
| 关闭（只编码文字） | 图片被忽略，只编码文字 | 所有普通文生图模型 |
| 参考图 → 服务端 VAE 编码（Qwen/Flux Edit 式） | 每张图在服务端用 VAE 编码成 latent，挂到条件的 `reference_latents` 上 | Qwen-Image-Edit、Flux Kontext 这类编辑模型 |
| 图片 → 服务端 CLIP 视觉（Redux 式） | 每张图在服务端用 clip_vision 编码，挂到条件的 `clip_vision_output` 上 | Flux Redux、IPAdapter 这类参考图模型 |

- 图片接口是**动态增槽**：节点上只显示 `图片1` 一个空接口，**每接上一张图，后面立刻自动多出一个空接口**
  （最多 32 个，拔掉图后多余的空槽会自动收回去）。新建节点/载入工作流都会自动整理成「接了几张 + 1 个空槽」。
- 配合两个新字段：
  - `vae`（可选输入）：图片模式 = VAE 编码时接「远程 VAE 加载器」的 vae——编码在服务端做，本机不加载 VAE。
  - `视觉模型`：图片模式 = CLIP 视觉时用，填服务端 clip_vision 目录里的文件名；**留空 = 自动用目录里的第一个**。
  - `参考图最大像素`：百万像素(MP) 上限（默认 1.00 ≈ 1MP，服务端按它等比缩放；0 = 原尺寸）。
- 服务端新增第三类模型目录 **clip_vision**：默认扫 `<comfyui_dir>/models/clip_vision`，
  放在别处就在 `config.json` 写 `"clip_vision_model_dir": "D:/models/clip_vision"`。
  `1_检查环境并生成配置.bat` 现在也会扫这个目录（`models/clip/` 里混着的 `EVA02_*` 这类视觉编码器
  会被自动归到 clip_vision，不再被当成文本编码器）。

实测（临时 ComfyUI 8196 + 服务端 mock 8197，`EmptyImage → RemoteCLIPTextEncode(图片1/图片2) → PreviewAny`）：

| 模式 | 结果 |
|---|---|
| 参考图 → VAE 编码 | `node_errors: {}`、`status_str: success`，日志 `参考图1 → 服务端 latent (1,4,64,64)`，条件额外键 `reference_latents`（2 条 latent） |
| 图片 → CLIP 视觉 | 同上，日志 `图片1 → 服务端视觉特征 (1,81,64)`，条件额外键 `clip_vision_output`（image_embeds 2×81×64 + penultimate 2×729×64） |
| 关闭 | 图片被忽略，条件里只有 `pooled_output` |

### 2. 服务端图片解码/编码重写（快 + 稳）

旧实现是逐帧串行 `base64.b64decode → Image.open → np.stack → 整体 astype(float32)`，
帧一多就明显卡在 CPU 上。新版 `codec.py`：

- **线程池并行**多帧解码/编码（`max(2, min(8, CPU 核数))` 个线程），两端 `codec.py` 仍逐字节一致；
- `np.asarray` 零拷贝 + 先按 uint8 堆叠，再一次性 `from_numpy().float().div_(255)`，少一次大张量拷贝；
- 支持灰度 / LA / P / PA / CMYK / YCbCr / 16bit（`I;16` → 8bit）/ float 图，
  多帧通道数或尺寸不一致时自动用最近邻统一（不再因为一张图格式不同就整条失败）；
- `Image.open` 一律 `with` 显式关闭，Encoding 用 `compress_level=3`（比默认 6 快、体积几乎一样）。

实测（4 帧 1024×1024）：**编码 0.131s / 解码 0.038s，往返最大误差 0.00196（< 1/255）**，
单帧 selftest（灰度、RGBA）全过。

---

## 本次更新 v1.7.4 / v1.7.5 / v1.7.6 —— 节点按截图改版 + 修校验 + 直连自带节点

- 节点标题：`远程 CLIP 加载器 (Remote CLIP Loader)` / `远程 VAE 加载器 (Remote VAE)` /
  `远程 CLIP 文本编码 (Remote CLIP Encode)`；分类统一为 **`comfyui_remote_clip_vae`**，
  节点脚标也是它。**全文不再出现「A 端 / B 端」字样**，服务端日志里的 `[B]` 也改成了 `[服务端]`。
- 字段顺序按截图：服务端显卡 → 服务端地址 → 模型 → 超时时间 → 手动模型名（→ 模型类型），
  最后一行整宽按钮 **「🔄 刷新模型列表」**。
- 「服务端显卡」改为**只读灰字**：点刷新时自动填成 `显卡型号 显存 · 计算方式`（如 `RTX 4070 12.0GB · CUDA`），
  连不上显示「（连不上服务端）」，手改会被还原。
- 文本编码节点一个节点出**正向条件 / 负向条件两路**，直接接 KSampler（省掉原来的两个编码节点）。
- 「模型类型」现在真的生效：选 `flux` / `sd3` / `qwen_image` 时会把 clip_type 传给服务端，
  服务端发现类型变了会自动卸载旧的再按新类型重装。
- 附带保留 `远程 VAE 解码` / `远程 VAE 编码`（同风格命名，去掉了 vae 句柄没有消费者的死路）。

### v1.7.5 修「无效输入：模型」这个报错

现象：下拉里明明识别到了模型，点运行却红条报 **无效输入 · 远程 VAE 加载器 - 模型 / 远程 CLIP 文本编码 - 模型**。

原因：ComfyUI 在**启动那一刻**就固化了每个下拉框的合法值列表（`INPUT_TYPES` 的返回值），
而「模型」是点「🔄 刷新模型列表」时从服务端实时拉的——新名字不在启动时那份快照里，
后端队列校验就判 `value_not_in_list`，跟模型本身能不能用无关。

修法（两条）：
1. 三个节点都加了 `VALIDATE_INPUTS(模型=..., 模型类型=...)`，**跳过这份白名单校验**，
   只拦明显是占位符的值（`（连接失败，检查服务端）` 之类），剩下的交给执行阶段报清楚的中文错误；
   顺手兼容了旧版标签写法（`flux（Flux 的 t5xxl / clip_l）` → `flux`）。
2. 下拉清单加了本地缓存 `model_cache.json`：只要成功拉过一次，ComfyUI 重启后即使服务端还没开，
   下拉里也是真名字，不会再退回占位符。

实测：起一个临时 ComfyUI（服务端故意不开，启动时下拉只有占位符），把 `模型` 填成启动列表里
根本没有的 `my_flux_vae.safetensors` 直接 `/prompt` 提交——返回 `node_errors: {}` 正常排队，
之后的失败是执行阶段「连不上服务端」的中文报错，不再是「无效输入」。

### v1.7.6 加载器输出改成真 `CLIP` / `VAE` 类型，可以直连自带节点

之前「远程 CLIP 加载器」输出的是自定义类型 `REMOTE_CLIP`，只能接本套件自己的节点，
接不了 ComfyUI 自带的 `CLIPTextEncode`。现在改成输出**鸭子类型的代理对象**，
`RETURN_TYPES` 就是 `CLIP`（VAE 加载器同理，就是 `VAE`）：

| 加载器 | 输出类型 | 现在能直接接 |
|---|---|---|
| 远程 CLIP 加载器 | `CLIP` | 自带 `CLIPTextEncode`、`CLIPSetLastLayer`、`CLIPTextEncodeSDXL`… |
| 远程 VAE 加载器 | `VAE` | 自带 `VAEDecode`、`VAEEncode`、`VAEDecodeTiled`、`VAEEncodeTiled`、`VAEEncodeForInpaint`… |

代理对象实现的是原生节点真正会调的那些方法（`tokenize` / `encode_from_tokens_scheduled` /
`encode_from_tokens` / `clip_layer` / `clone`，`decode` / `encode` / `decode_tiled` / `encode_tiled` /
`spacial_compression_decode` / `throw_exception_if_invalid` …），内部全部转成 RPC 发给服务端，
本机不加载任何模型。所以两种用法随便挑：

```
用法一（和原生一模一样）：
  远程 CLIP 加载器 ── clip ─→ CLIPTextEncode(正向) ─→ KSampler positive
                          └→ CLIPTextEncode(负向) ─→ KSampler negative
  远程 VAE 加载器  ── vae  ─→ VAEDecode ─→ PreviewImage / SaveImage

用法二（少连几根线）：
  远程 CLIP 文本编码（正向/负向一个节点出两路）→ KSampler
  远程 VAE 解码 / 编码（等价于自带 VAEDecode / VAEEncode）
```

实测（真 ComfyUI，服务端 mock）：`远程 VAE 加载器 → 原生 VAEDecode → SaveImage` 真的写出了
13.7KB 的 PNG；`远程 CLIP 加载器 → 原生 CLIPSetLastLayer → 原生 CLIPTextEncode` 提交后
`node_errors: {}`（类型/连线校验通过），编码结果 1×77×768 + pooled 1×1280。
本套件自己的编码/解码节点在「接 / 不接 clip 句柄」两种情况下也都跑通。


---

## 一、目录结构

```
RemoteCLIPVAE/
  RemoteCLIPVAE服务端.exe       ★★ 双击就开（Windows 桌面程序，不需要装 python）
  0_启动图形界面.bat            ★ 图形控制面板：左边配置/状态，右边黑底实时日志
  1_检查环境并生成配置.bat     探测 ComfyUI 目录/Python，扫描模型，生成 config.json
  2_启动服务端.bat             用上一步记录的 Python 启动（可加 --mock 联调）
  3_安装插件到ComfyUI.bat      把 comfyui_remote_clip_vae 拷进 custom_nodes，并写入服务器地址
  4_用ComfyUI的Python启动服务端.bat  指定 ComfyUI 目录启动（run_headless.py 方式，日志留在控制台）
  5_自测.bat                   不启动 ComfyUI，直接跑一遍 CLIP→VAE 全链路

  comfyui_remote_clip_vae/     ← 装到 出图电脑 custom_nodes 的插件
    __init__.py                节点注册 + WEB_DIRECTORY
    nodes.py                   5 个节点
    api.py                     本机代理路由 /remote_clipvae/probe（刷新按钮用）
    web/refresh.js             「🔄 刷新模型列表」按钮、显卡栏只读
    codec.py                   张量/图像序列化协议（与 model_server/codec.py 完全一致）
    remote_api.py              HTTP 客户端（只用标准库）
    settings.py                默认服务器地址读取
    default_server.json        服务器地址（3_ 脚本会帮你写）
    model_cache.json           ← 自动生成：上次成功拉到的模型清单（重启后下拉不丢）

  model_server/                ← 在 服务端电脑运行
    app.py                     HTTP 服务（/health /models /rpc）
    backend.py                 CLIP/VAE 加载、编解码、LRU、空闲释放
    config.py                  配置读写 + 模型自动扫描
    codec.py                   同上（两端必须逐字节一致）
    config.json                唯一需要改的配置
    config.example.json        Flux/SDXL/SD1.5/SD3/Qwen 的写法参考
    gui.py                     ★ 桌面版控制面板（tkinter，只用标准库）
    panel.py                   ★ 网页面板（没有 tkinter 时的自动替换形态）
    _python.txt                ← 面板/1_ 生成：记住 ComfyUI 的 python（2_/4_/5_ 直接复用）
    _comfy.txt                 ← 面板/1_ 生成：记住 ComfyUI 目录（4_ 直接复用）
    run_headless.py            --backend comfy --comfy-path ... 兼容入口
    selftest.py                5_ 脚本调用的自测
```

---

## 二、图形界面（推荐先跑这个）

**最省事的用法：直接双击 `RemoteCLIPVAE服务端.exe`**（Windows 桌面窗口，自带 tkinter 运行时，
机器上有没有 python、装没装 ComfyUI 都无所谓，能打开）。打开后在界面里填四样东西：

| 设置项 | 填什么 |
|---|---|
| ComfyUI 路径 | 服务端电脑上 ComfyUI 的源码根目录（能看到 `comfy` 文件夹和 `main.py` 的那层），点「浏览...」选 |
| 本机 ComfyUI 的 Python | **选完上一行就自动填好**：按这个 ComfyUI 目录找 `python_embeded\python.exe` / `.venv\Scripts\python.exe`（含上一层目录的 `.venv`），并优先挑里面带 torch 的那个；找不到再手动「浏览...」 |
| CLIP 模型目录 | 文本编码器所在文件夹，例如 `<ComfyUI>\models\text_encoders`（也可以选 `models\clip`，或 ComfyUI 外面的任何目录） |
| VAE 模型目录 | VAE 所在文件夹，例如 `<ComfyUI>\models\vae` |
| 视觉编码器目录（`clip_vision`） | 只有用「图片 → CLIP 视觉」模式才需要：默认 `<ComfyUI>\models\clip_vision`；放在别处就在 `config.json` 写 `"clip_vision_model_dir": "D:/models/clip_vision"`，界面里不需要选 |

> **CLIP / VAE 两栏只认你填的目录**：改「ComfyUI 路径」不会去动这两栏，扫描也只扫这两个目录里的模型文件
> （不会再从 `<ComfyUI>\models` 下面另外猜一遍）。
> **不用手点扫描**：改 CLIP / VAE 目录（含「浏览...」选）后立刻扫描，列表马上出现条目
> （`clip/` 里的 `EVA02_CLIP_*` 这类视觉编码器会自动跳过，避免加载时报错）。
> 服务已经开着时再扫描/加目录，界面会自动让服务端重读 `config.json`（走 `/reload`），**不用重启服务**。

**exe 是单文件、自带服务端**：`app.py` / `backend.py` / `codec.py` / `config.py` 都已经打进 exe 里，
运行时解到临时目录再交给 ComfyUI 的 python 跑。所以：

- 把 `RemoteCLIPVAE服务端.exe` **单独拖到任何文件夹（甚至桌面）都能直接双击用**，不需要 `model_server` 目录在旁边
- 配置写在 **exe 同一目录**的 `config.json`（外加 `gui_state.json`，记录上次填的四个路径）；
  万一 exe 放在不可写的位置（如 `C:\Program Files`），会自动改用 `%APPDATA%\RemoteCLIPVAE\config.json`
- 想确认自带的服务端是不是完整：`RemoteCLIPVAE服务端.exe --smoke`，退出码 0 且同目录生成 `_smoke.txt`，
  里面会写 `app.py/codec/backend : OK` 和 `srv_dir` 的实际位置

（包里的 `model_server` 目录保留给 `2_` / `4_` / `5_` 这些 bat 和网页面板那条路线用，**exe 用不到它**。）

填完点「启动服务」即可。全部设置存在 `model_server\config.json`，包里**没有任何写死的路径**，换机器就是重新填这四个框。

也可以双击 `0_启动图形界面.bat`（有 exe 时它直接帮你打开 exe），出来的就是同一个控制面板：左边「服务配置 / 运行状态 / 服务端 CLIP 模型 / 服务端 VAE 模型」，
右边「本机日志 / 服务端日志（实时）」两个页签 + 黑底实时控制台 + 清屏。

| 区域 | 内容 |
|---|---|
| 服务配置 | 监听地址、端口（**计算设备不在界面里选**：一律自动，NVIDIA / AMD / Intel 都交给 ComfyUI+torch 判断，状态栏会显示实际用的是哪个）、CLIP 模型目录、VAE 模型目录、ComfyUI 路径、本机 ComfyUI 的 Python、勾选「只读这两个目录」；按钮 `启动服务` `停止服务` `清空显存(卸载模型)` `刷新模型列表` `打开目录` |
| 运行状态 | 计算方式、加载后端、设备（显卡名）、显存 已用/总量、GPU 利用率、CPU、torch 已用、内存；显存与 GPU 利用率两条进度条实时刷新 |
| 服务端 CLIP / VAE 模型 | 两个列表（扫描出来的条目，可 Ctrl 多选），`预加载选中的` 直接把模型装进 服务端显存，底部显示当前已加载 |
| 日志 | `本机日志`（面板自己记的启动/失败原因）+ `服务端日志（实时）`（服务端进程 stdout/stderr 一行行进来，带颜色） |

两种形态，自动挑：

- 本机 python **有可用的 tkinter** → 桌面窗口（`model_server/gui.py`；有 `pythonw.exe` 就用它启动，不带黑框）
- **没有 tkinter**（整合包 `python_embeded` 常被精简掉）→ 本地网页面板（`model_server/panel.py`），
  用 Chrome / Edge 的 `--app=` 打开成独立窗口；面板监听 `127.0.0.1:8190`（被占用自动 +1，最多试 20 次），
  关掉它留下的黑框窗口 = 服务端一起停

> 界面用什么 python 跑无所谓：**服务端永远是用 ComfyUI 的 python 起子进程**（对应面板里「本机 ComfyUI 的 Python」和「优先用 ComfyUI 自带的 Python」两项）。

---

## 三、5 步跑起来

### 第 1 步：服务端电脑上 `1_检查环境并生成配置.bat`

- 把 **ComfyUI 源码根目录**拖进窗口（那个里面能看到 `comfy/` 文件夹和 `main.py` 的目录，
  整合包路径通常是 `...\ComfyUI_Mie_xxx\ComfyUI_Mie_xxx\ComfyUI`）
- 脚本会自动找到 ComfyUI 自带的 `python_embeded\python.exe`
- 然后扫描 `models\` 并写入 `config.json`，同时打印扫描到的 VAE / 文本编码器
- 顺带生成两个小文件：`model_server\_python.txt`（记住 python 路径）、`model_server\_comfy.txt`
  （记住 ComfyUI 目录）。后面 2_/4_/5_ 会自动读它们，不用再拖一次路径

> 本包**故意不建 venv**：服务端必须用 ComfyUI 自己的 Python，因为要 `import comfy.sd`。
> 用独立 venv 会报 `No module named 'comfy_aimdo'` 或者能跑但数值对不上。

### 第 2 步：`2_启动服务端.bat`

```
==============================================================
 服务端模型服务已启动
 监听：http://0.0.0.0:8199
 本机节点里填：http://192.168.3.2:8199
==============================================================
```

> v1.6.2 起 `2_` 不再无脑抓 PATH 上的 `python`：它会先探测 `import numpy, torch`，
> 跑不通就打印 `[skip] ...` 跳过。PATH 上的 Microsoft Store 版 python 常装着残缺的 torch
> （报 `ModuleNotFoundError: No module named 'torchgen'`），那种 python 绝对不能跑服务端。
> 如果它一个可用的都没找到，会直接提示你三种做法（0_ 面板 / 1_ 生成配置 / 4_ 指定目录），
> 也可以当场把 ComfyUI 目录拖进去，它会自己挑 `python_embeded` / `python_embedded` / `python` / `.venv` / `venv`。
**先建议 `2_启动服务端.bat --mock`**：不加载真模型、返回随机张量，几十秒就能验证网络和节点接线。
启动时如果提示"以下模型文件的路径不存在"，照着改 `config.json` 即可。

### 第 3 步：出图电脑上 `3_安装插件到ComfyUI.bat`

- 把 出图电脑的 `custom_nodes` 目录拖进窗口
- 脚本会问服务器地址（服务端窗口顶部那行 `http://192.168.x.x:8199`），直接写进 `default_server.json`
- 重启 ComfyUI，节点菜单里出现分类 **`comfyui_remote_clip_vae`**

### 第 4 步（可选）：`4_用ComfyUI的Python启动服务端.bat`

和第 2 步是同一个服务端，只是改成"指定 ComfyUI 目录 + `run_headless.py --backend comfy --comfy-path`"的写法，
适合把服务端和 ComfyUI 装在不同盘、第 1 步没记录到 Python 的情况。

用法三种都行：

```
4_用ComfyUI的Python启动服务端.bat                       ← 直接回车，用第 1 步记下的目录
4_用ComfyUI的Python启动服务端.bat "F:\xxx\ComfyUI"      ← 当场指定目录
4_用ComfyUI的Python启动服务端.bat --mock                ← 只带参数，目录照样从 _comfy.txt 读
```

### 第 5 步：`5_自测.bat`

不启动 ComfyUI，直接按节点背后的逻辑跑：`/health` → CLIP 编码 → VAE 解码 → VAE 编码。
看到 `[OK] 全链路通过` 就说明两台机器、协议、模型全部就绪。

---

## 四、工作流怎么接

```
原生：  CheckpointLoaderSimple ─┬─ CLIP ─→ CLIPTextEncode ─────────┐
                               ├─ VAE  ─→ VAEDecode ─→ SaveImage  │
                               └─ MODEL ──────────────→ KSampler ─┘

改造：  Load Diffusion Model（只加载 UNet）────────────────→ KSampler
        远程 CLIP 加载器 → 远程 CLIP 文本编码 ───────────────→ KSampler(positive / negative 两路)
        远程 VAE 加载器 → 远程 VAE 解码 ────────────────────→ PreviewImage / SaveImage
```

⚠️ **最容易踩的坑**：本机如果继续用 `CheckpointLoaderSimple` / `Load Checkpoint`，
它会把 checkpoint 里自带的 CLIP 和 VAE **一起加载进 本机的显存** —— 那就没省到。
真正省显存的用法是 本机用 `Load Diffusion Model` 加载独立的 unet / diffusion_models 文件
（`flux1-dev.safetensors`、SDXL 的 unet 版等），CLIP 和 VAE 全部走 服务端。
如果手上只有整包 checkpoint，先用 ComfyUI 的"拆包/unet 提取"把 unet 单独存出来。

### 节点清单（分类：`comfyui_remote_clip_vae`）

| 节点 | 输入（从上到下） | 输出 | 说明 |
|---|---|---|---|
| **远程 CLIP 加载器** (Remote CLIP Loader) | 服务端显卡 / 服务端地址 / 模型 / 超时时间(5) / 手动模型名 / 模型类型 / 🔄 刷新模型列表 | `clip`（类型 **CLIP**） | 句柄式代理对象，本机不占显存；可直接接自带 CLIPTextEncode / CLIPSetLastLayer |
| **远程 VAE 加载器** (Remote VAE) | 服务端显卡 / 服务端地址 / 模型 / 超时时间(5) / 手动模型名 / 🔄 刷新模型列表 | `vae`（类型 **VAE**） | 同上（没有「模型类型」）；可直接接自带 VAEDecode / VAEEncode / VAEDecodeTiled |
| **远程 CLIP 文本编码** (Remote CLIP Encode) | 正向提示词 / 负向提示词（两个多行框）/ 服务端显卡 / 服务端地址 / 模型 / 超时时间(120) / 手动模型名 / 模型类型 / 🔄 刷新模型列表（可选输入 `clip`） | `正向条件` `负向条件` | 一个节点同时出正负两路 CONDITIONING，直接接 KSampler；不接 clip 也行，用它自己的字段 |
| **远程 CLIP 图文编码** (Remote CLIP Encode + Images) | 正向提示词 / 负向提示词 / 服务端显卡 / 服务端地址 / 模型 / 模型类型 / 参考图最大像素(MP)（可选 `clip` `vae` + `图片1..图片10`，接一张自动增槽） | `正向条件` `负向条件` | 图片既进 vision 槽又（接了 vae 时）编 `reference_latents`；`keep_vision` 自动按有无 vae 决定 |
| **提示词优化** (Prompt Optimizer, Remote) | 用户提示词 / 服务端地址 / 接口类型 / 模型名 / API Key / 模式 / 输出语言 / 透明背景 / 种子值 / 种子模式 / 温度 / 最大输出长度 / 超时时间（可选 `图片1..图片10`，接一张自动增槽） | `prompt` `json` `是否反推` | 不在本机加载 GGUF，直接调远端 LLM（OpenAI 兼容 / Ollama 原生），本机零依赖 |
| 远程 VAE 解码 (Remote VAE Decode) | vae（类型 VAE）/ latent / 超时时间(600) / 分块解码 / 分块尺寸 / 重叠 / 输出设备 | `图像` | 等价自带 VAEDecode，只是把解码放服务端 |
| 远程 VAE 编码 (Remote VAE Encode) | vae（类型 VAE）/ 图像 / 超时时间(600) / 分块编码 / 分块尺寸 / 重叠 | `latent` | 等价自带 VAEEncode，编码在服务端做 |

参数要点：

- **服务端显卡**：只读灰字，点「🔄 刷新模型列表」时自动填服务端型号 + 显存，连不上显示「（连不上服务端）」
- **模型**：服务端 config.json 里的条目名；连不上服务端时下拉是「（连接失败，检查服务端）」，不会让 ComfyUI 起不来
- **手动模型名**：填了优先于下拉（服务端条目名，不依赖网络也能跑老工作流）
- **模型类型**：`自动（按文件名判断）` = 用服务端 config.json 里的 `clip_type`；Flux / SD3 / Qwen 对不上时可手动指定，服务端会按新类型重新装模型
- **超时时间**：加载器是连接检查超时（秒）；文本编码 / VAE 是单次调用超时，首次装大 CLIP 要留足（默认 120 / 600）
- **分块解码 / 编码**：服务端显存不够解大图时打开，走 `decode_tiled/encode_tiled`
- **输出设备**：VAE 解码默认 `cpu`（最省本机显存，SaveImage/PreviewImage 都支持）；文本编码 `auto` 与原生一致

---

## 五、config.json

| 字段 | 说明 |
|---|---|
| `server.host` / `server.port` | 监听地址与端口，默认 `0.0.0.0:8199` |
| `server.token` | 可选鉴权；填了之后 本机节点要填同一个 |
| `comfyui_dir` | **必改**，ComfyUI 源码根目录（含 `comfy/`） |
| `device` | `auto`（默认）/ `cuda`（强制显卡）/ `cpu`（强制纯 CPU）。**界面里已经没有这个选项了**，想强制就在 config.json 里手写；不写就是 auto，AMD / Intel / NVIDIA 显卡都交给 ComfyUI 自己判断 |
| `max_loaded.clip` / `.vae` | 同时驻留的模型数，超出按最久未用淘汰，默认各 1 |
| `idle_unload_seconds` | 空闲多少秒自动释放模型，0 = 不释放 |
| `models.clip` | 键名随便起（本机下拉框显示它）；`files` 可用多个文件（Flux 的 T5+CLIP-L）；`clip_type` 如 `flux`/`stable_diffusion`/`sd3`/`qwen_image` |
| `models.vae` | 同上，通常单个文件 |

`files` 写相对路径时按 `<comfyui_dir>/models/` 查找，也可以写绝对路径。
`config.example.json` 里有 Flux / SDXL / SD1.5 / SD3 / Qwen-Image 的现成写法。

---

## 六、接口（可直接 curl 调试）

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/health` | 设备、显存、已加载模型、可用模型清单 |
| GET | `/models` | 模型清单 |
| GET | `/reload` | 重读 `config.json`（界面扫描/改模型目录后自动调，不用重启服务） |
| POST | `/rpc` | `{"op": "...", "params": {...}}` |

`op` 取值：`clip_encode` / `vae_decode` / `vae_encode` / `vae_info` / `load` / `unload` / `status` / `models`

```powershell
curl.exe -X POST http://127.0.0.1:8199/rpc -H "Content-Type: application/json" -d '{\"op\":\"unload\"}'
```

---

## 七、带宽与耗时（千兆局域网，实测）

| 操作 | 数据量 | 网络耗时 | 备注 |
|---|---|---|---|
| 文本编码 SD1.5（77×768） | ~0.2 MB | 5~10 ms | |
| 文本编码 Flux（512×4096+pooled） | ~8 MB | 60~90 ms | 显存吃紧可 `device=cpu` |
| VAE 解码 1024² | PNG 1.5~2 MB | 20~40 ms | 服务端显存紧就开 `use_tiled` |
| VAE 编码 1024² | PNG 1.5~2 MB | 20~40 ms | 同上 |

时间大头是 服务端**首次装模型**（SDXL CLIP 约 15~60 s，Flux 的 T5-XXL 更久），之后走缓存。

---

## 八、已验证环境与实测数据

本包在下面这套环境里跑过（真模型、非模拟）：

- 服务端：Windows + ComfyUI Mie 整合包（Python 3.13、torch 2.10.0+cu130、RTX 5070）
- 实测：`context = 1×77×768`、`pooled_output = 1×768`（从 checkpoint 抽出的 CLIP）
- `flux2_vae`（4 维，128 通道，16× 下采样）：解码 → `IMAGE 1024×1024×3`；编码 → `LATENT 128×64×64`
- `qwen_image_vae`（5 维视频 VAE，16 通道）：自动补 `T=1`，解码 → `512×512`，编码 → `16×64×64`
- 协议自测：张量往返完全一致，PNG 往返误差 ≤ 1/255
- `--mock` 全链路（health→编码→解码→编码）：0.4 s

---

## 九、排错对照表

| 现象 | 原因 / 解决 |
|---|---|
| 双击 `0_启动图形界面.bat` 弹出黑框又立刻消失 | 黑框就是网页面板进程，它会自己去开浏览器窗口；浏览器没起来就手动打开黑框里打印的 `http://127.0.0.1:819x/` |
| 面板能开，点「启动服务」后状态仍是「未运行」 | 看「本机日志」的红字：通常是记录的 python 不存在，点「自动找」重选 `python_embeded\python.exe` |
| 面板顶部给的地址不是你想要的网卡 | 多网卡（tailscale / 虚拟网卡）按 192.168 → 10 → 172 排序取第一个；要换就把「监听地址」改成具体 IP |
| 拖进「本机 ComfyUI 的 Python」后启动，报「还没指定 ComfyUI 的 python」或路径看着没错却不通 | Windows 拖拽会自动加双引号（`"D:\...\python.exe"`）。v1.6.1 起面板和 bat 都会自动去掉引号；旧包请手动删掉引号再点一次 |
| 用的是 ComfyUI Desktop（python 在 `ComfyUI\ComfyUI\.venv\Scripts\python.exe`） | 把「ComfyUI 路径」选到 `...\ComfyUI\ComfyUI`（里面有 comfy 和 main.py）即可，python 会自动识别（含 `.venv\Scripts`，也看上一层目录的 `.venv`），优先挑带 torch 的那个 |
| 选了模型目录，列表还是 0 条 | 先看「本机日志」里的扫描行：目录不存在会红字提示；目录里只有 `put_*.safetensors` 占位文件、或者都是 `clip_vision` / `EVA02` 视觉编码器时会被跳过（那些本来不能当文本编码器） |
| 「刷新模型列表」扫出 0 个条目 | Desktop 版 `models\vae`、`models\text_encoders` 常是空目录、模型放别处。把「CLIP 模型目录」「VAE 模型目录」填成真正的模型所在目录，再刷新 |
| 「运行状态」里设备显示 `—` | 服务还没起来；起来后会显示显卡名和显存实时占用 |
| 节点下拉框只有「（连接失败，检查服务端）」 | 看节点提示里的具体原因：服务端没启动 / IP 端口错 / 防火墙 / token 不一致 |
| 刷新失败：`Unexpected endpoint or method. (GET /health)` | **这个端口上不是本服务端**：要么「服务端地址」填到了 ComfyUI(8188) / LM Studio / Ollama 这类同机 HTTP 服务，要么端口号写错。用服务端面板标题上的 `http://<服务端IP>:8199`；面板「运行状态」必须是运行中；浏览器打开 `http://<IP>:8199/health` 应看到 `{"ok":true,...}` |
| 刷新失败：`无法连接服务端 ... 目标计算机积极拒绝` | 那个端口没人监听：exe 没点「启动服务」、IP/端口错、或防火墙没放行入站 TCP 8199 |
| `无法连接 服务端服务` | 两台机不在同一网段；`ping B的IP`，浏览器打开 `http://B的IP:8199/health` |
| 服务端启动就报 `No module named 'codec'` | 用了整合包 `python_embeded`（带 `._pth` 会丢脚本目录）—— 本包已在 `app.py` 里显式补 `sys.path`，别删那段 |
| `无法从 ... 导入 comfy 模块` | `comfyui_dir` 指错，要指到含 `comfy/` 和 `main.py` 的目录 |
| `CLIP 文件不存在` | `config.json` 里 `files` 路径错（相对路径是相对 `<comfyui_dir>/models/`） |
| `VAE is invalid: None` | 那个 checkpoint 里没有 VAE；换成独立 VAE 文件（`models/vae/*.safetensors`、`ae.safetensors`、`sdxl_vae.safetensors`） |
| `VAE 解码失败：latent 有 4 个通道，但该 VAE 需要 16 个` | 跨机模型不同源（SD1.5 的 latent 配了 Flux/Qwen 的 VAE） |
| KSampler 维度不匹配 / 出鬼图 | 同上，A 的 UNet 与 B 的 CLIP/VAE 必须同一代模型 |
| 服务端 VAE 解码 OOM | 勾 `use_tiled`，或调小 `tile_size` |
| 图像与本地解码略有差异 | PNG 8bit 量化，≤1/255，正常 |
| 第一次执行卡很久 | 服务端正在装模型，看 B 控制台；可把加载器 `preload` 打开提前装 |

---

## 十、安全

- 内网方案：`host` 用 `0.0.0.0` 时只在信任的局域网里用，**不要把 8199 映射到公网**
- 多人共用设 `server.token`
- 随时释放模型：`curl -X POST http://B的IP:8199/rpc -d "{\"op\":\"unload\"}"`，或等 `idle_unload_seconds`
