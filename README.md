# photo-upscale-kit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%2B%20WSL2-lightgrey.svg?logo=windows&logoColor=white)](#环境要求)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Real-ESRGAN](https://img.shields.io/badge/Real--ESRGAN-x4-4B8BBE.svg)](https://github.com/xinntao/Real-ESRGAN)
[![Stable Diffusion](https://img.shields.io/badge/Stable%20Diffusion-1.5-8957e5.svg)](https://huggingface.co/runwayml/stable-diffusion-v1-5)
[![ControlNet](https://img.shields.io/badge/ControlNet-Tile-ff6b6b.svg)](https://huggingface.co/lllyasviel/ControlNet-v1-1)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![GitHub stars](https://img.shields.io/github/stars/Alexander390370/photo-upscale-kit?style=social)](https://github.com/Alexander390370/photo-upscale-kit/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/Alexander390370/photo-upscale-kit)](https://github.com/Alexander390370/photo-upscale-kit/commits/main)
[![GitHub repo size](https://img.shields.io/github/repo-size/Alexander390370/photo-upscale-kit)](https://github.com/Alexander390370/photo-upscale-kit)

**自动判断该超分还是重绘** —— 给低分辨率 / 低质量照片做放大与修复，跑在 **Windows + WSL2 + NVIDIA GPU** 上。

> ⚠️ **一句话结论**：边缘密度 < 0.01 的平滑虚焦图**不要重绘**，模型会凭空补出假纹理，越弄越糊。

---

## 目录

- [它解决什么问题](#它解决什么问题)
- [三条路，别混着用](#三条路别混着用)
- [模块详解](#模块详解)
  - [classify.py — 自动判断](#classifypy--自动判断)
  - [upscale_wsl.py — 超分](#upscale_wslpy--超分)
  - [redraw.py — 重绘](#redrawpy--重绘)
  - [auto.py — 一键入口](#autopy--一键入口)
  - [denoise.py — 降噪锐化](#denoisepy--降噪锐化)
  - [compare.py — 客观测量](#comparepy--客观测量)
- [实测结论](#实测结论)
- [快速开始](#快速开始)
- [实现要点（踩过的坑）](#实现要点踩过的坑)
- [已知限制](#已知限制)

---

## 它解决什么问题

常见的"AI 放大"工具都是一个按钮怼上去。实际用下来会发现：

**同一套参数，有的图效果好得惊人，有的图反而更糊。**

原因不在参数，在**原图本身有没有可恢复的结构**：

| 原图情况 | 适合的做法 |
| --- | --- |
| 有清晰主体、硬边、可辨轮廓 | 超分放大 / ControlNet 重绘 |
| 大面积平滑渐变、虚焦、低光 | 只能超分，或干脆别动 |
| 极小但有结构（< 300px） | 先超分再重绘，收益最大 |

这个工具的核心就是**先用指标判断，再决定怎么处理**，而不是盲目套一种方法。

---

## 三条路，别混着用

| 方案 | 做什么 | 放大 | 改内容 | 什么时候用 |
| --- | --- | --- | --- | --- |
| **仅超分**<br>`upscale_only` | Real-ESRGAN 4× | ✅ | 极小 | 只要变大；或原图太平滑，重绘会画蛇添足 |
| **仅重绘**<br>`redraw_only` | SD img2img + ControlNet Tile | ❌ | 中等 | 原图够大（>400k 像素），只需要提升质感 |
| **超分+重绘**<br>`upscale_then_redraw` | 先 4× 超分，再分块重绘 | ✅ | 中等 | 原图很小但有结构，收益最大 |

### 为什么不能混着用

- **对平滑虚焦图重绘** → SD 找不到纹理可依，但 prompt 要求"皮肤纹理"，于是**凭空画**，把柔和光影切成色块
- **对小图只超分** → Real-ESRGAN 只能插值猜，出来的细节是糊的，浪费了小图重建的潜力
- **对已够大的图再 4× 超分** → 文件涨到几十 MB，屏幕上看不出差别，纯属虚胖

---

## 模块详解

### `classify.py` — 自动判断

**它做什么**：测量图片特征，判断该走哪条路。不修改图片。

**怎么用**：

```bash
# 人类可读
python classify.py 图片.jpg 图片2.png

# 机器可读（给 auto.py 用）
python classify.py --json 图片.jpg
```

**输出示例**：

```
文件                        尺寸      边缘密度      拉普拉斯      梯度         建议
thumb_1476.jpg        1365x1024      0.0001        4.7    7.64        仅超分
thumb_2627.jpg          256x455      0.0247      185.7   27.98      超分+重绘
thumb_1526.jpg         1024x576      0.0321      273.4   24.49        仅重绘

  thumb_1476.jpg
    → 仅超分：边缘密度仅 0.0001，画面以平滑渐变为主，ControlNet 锁不住结构，重绘会补出假纹理
```

**测量哪些指标**：

| 指标 | 算法 | 含义 |
| --- | --- | --- |
| `edge_density` | Canny 边缘检测，算边缘像素占比 | **主判据**。画面里有多少明确边界 |
| `lap_var` | 拉普拉斯算子方差 | 局部微对比强度（注意：噪点也会拉高） |
| `grad_mean` | Sobel 梯度幅值均值 | 整体边缘能量 |
| `hf_ratio` | `(原图-模糊)² / 原图²` | 细节在总能量中的占比 |

Canny 的阈值按图中位亮度自适应（`0.66×median` / `1.33×median`），避免全黑图或过曝图判错。

**判断逻辑**（阈值来自实测校准，见 [实测结论](#实测结论)）：

```python
if edge_density < 0.010:
    return "upscale_only"        # 平滑/虚焦，重绘会补假纹理
elif edge_density < 0.015:
    if lap_var < 30:
        return "upscale_only"    # 灰区，细节实在太少
    return "upscale_then_redraw" if 像素 < 400k else "redraw_only"
else:
    return "upscale_then_redraw" if 像素 < 400k else "redraw_only"
```

**为什么用 edge_density 当主判据**：

- 它直接对应"ControlNet 有多少东西可以锁"
- 比 lap_var 稳定（后者被噪点污染严重：虚焦图 lap_var 只有 5，但噪点图能到 200+）
- 阈值好定，跨图片尺寸也适用

---

### `upscale_wsl.py` — 超分

**它做什么**：用 Real-ESRGAN 放大图片，自动适配不同权重结构。

**怎么用**：

```bash
python upscale_wsl.py 输入.jpg 输出.png \
    --model ~/models/RealESRGAN_x4plus_anime_6B.pth \
    --scale 4 \
    --tile 384
```

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `--model` | 权重文件路径 | `~/ComfyUI/models/upscale_models/RealESRGAN_x4plus_anime_6B.pth` |
| `--scale` | 放大倍率。模型原生 4×，要别的倍率会先 4× 再缩回 | `4` |
| `--tile` | 切块大小。`0` = 整图一次过，显存不够时用 `256`/`384` | `0` |
| `--pre-blur` | 超分前先做高斯降噪（sigma） | `0` |

**支持三种权重**（自动识别，不用手填参数）：

| 权重 | 架构 | 大小 | 特点 |
| --- | --- | --- | --- |
| `RealESRGAN_x4plus.pth` | RRDB, 23 块, nf=64 | 64 MB | 保守，纹理自然 |
| `RealESRGAN_x4plus_anime_6B.pth` | RRDB, 6 块, nf=64 | 17 MB | **最锐**，实测效果最好 |
| `realesr-general-x4v3.pth` | SRVGGNetCompact | 4.7 MB | 轻量，带降噪倾向 |

> 💡 虽然 `anime_6B` 名字是给动漫图设计的，但因为它的边缘锐度更高，在**对比强烈、主体清晰**的照片上实测效果反而最好。

**自动推断架构的原理**：

```python
# RRDB 的键形如 body.<块号>.rdb<N>.conv1.weight
block_ids = {int(k.split(".")[1]) for k in state if k.startswith("body.") and 4段以上}
num_block = max(block_ids) + 1
num_feat  = state["conv_first.weight"].shape[0]      # 64
num_grow  = state["body.0.rdb1.conv1.weight"].shape[0]  # 32

# SRVGG 没有 conv_first，键形如 body.<序号>.weight
# 输出是残差，forward 里要加回双三次放大的原图
```

**关键细节**：SRVGG 那个权重的输出是**残差**，不加回原图的话画面接近全黑（输出均值 0.0025，98.8% 像素黑）。`build_srvgg()` 的 `forward()` 里有这一行：

```python
base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
return out + base
```

---

### `redraw.py` — 重绘

**它做什么**：用 Stable Diffusion 的 img2img，把图片分块重绘，补出真实感纹理。可选 ControlNet 锁结构。

**怎么用**：

```bash
python redraw.py 输入.jpg 输出.png \
    --model ~/models/Realistic_Vision_V5.1.safetensors \
    --controlnet ~/models/control_v11f1e_sd15_tile.pth \
    --cn-scale 1.0 \
    --scale 4 \
    --denoise 0.35 \
    --pre-denoise 1.2 \
    --tile 512 --overlap 128
```

**两个阶段**：

```
① 前置降噪（可选） → ② 超分（可选） → ③ 分块 SD 重绘 → ④ 羽化拼接
```

**参数详解**：

| 参数 | 作用 | 建议值 |
| --- | --- | --- |
| `--denoise` | **核心参数**。重绘强度，0=不改，1=完全重画 | `0.25~0.30` 保守 / `0.35` 平衡 / `0.45+` 大幅改 |
| `--cn-scale` | ControlNet 强度，1.0 完全锁结构 | `1.0`，想更自由用 `0.7~0.8` |
| `--pre-denoise` | 超分前的高斯降噪 sigma | 低光噪点图 `1.0~1.5`，干净的图 `0` |
| `--tile` | SD 处理的块大小 | `512`（SD1.5 训练分辨率） |
| `--overlap` | 块之间的重叠像素 | `128`（至少 64，否则接缝明显） |
| `--steps` | 采样步数 | `28` |
| `--guidance` | CFG scale，越大越贴 prompt | `4.0~5.0` |
| `--scale` | 先超分到几倍 | `4`，或加 `--skip-upscale` 跳过 |
| `--prompt` | 正向提示词 | 见下 |
| `--negative` | 负向提示词 | 见下 |

**默认提示词**：

```
正向: highly detailed photograph, sharp focus, natural skin texture, fine pores,
      soft studio lighting, professional photography
负向: blurry, lowres, jpeg artifacts, noise, grainy, watermark, text, cartoon, anime,
      deformed, oversaturated, plastic skin
```

> ⚠️ 提示词里的 `natural skin texture, fine pores` 是**双刃剑**：在有结构的图上能让皮肤更真实，在平滑虚焦图上会让 SD 凭空画出假毛孔。这就是为什么要先分类。

**分块接缝怎么解决的**：

```python
# 1) 步长 = tile - overlap，最后一块必须贴边
step = size - overlap
pos = list(range(0, total - size + 1, step))
if pos[-1] != total - size:
    pos.append(total - size)

# 2) 每个块用羽化权重图混合
#    中间 1，四边按 overlap 线性衰减到 0
mask[:, :, :overlap, :] *= ramp          # 上
mask[:, :, -overlap:, :] *= ramp.flip()  # 下
mask[:, :, :, :overlap] *= ramp          # 左
mask[:, :, :, -overlap:] *= ramp.flip()  # 右

# 3) 累加后加权平均
merged = accum / weight.clamp(min=1e-6)
```

**ControlNet 怎么接**：

```python
pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
    model, controlnet=ControlNetModel.from_single_file(cn_path))

# 每块推理时：
kwargs["control_image"] = crop              # 把原图自己作为条件
kwargs["controlnet_conditioning_scale"] = cn_scale
```

---

### `auto.py` — 一键入口

**它做什么**：调 `classify.py` 判断 → 按结果调 `upscale_wsl.py` 或 `redraw.py`。

**怎么用**：

```bash
# 只判断，不执行（先看看它打算怎么做）
python auto.py 图片.jpg --dry

# 自动判断并执行
python auto.py 图片.jpg

# 批量处理目录
python auto.py ~/图片/ --outdir ~/成品/

# 强制指定方案（跳过判断）
python auto.py 图片.jpg --force redraw_only
```

**输出文件名规则**：

| 方案 | 输出 |
| --- | --- |
| `upscale_only` | `原名_upscaled.png` |
| `redraw_only` | `原名_redrawn.png` |
| `upscale_then_redraw` | `原名_full.png` |

**模型路径配置**：复制 `models.example.json` 为 `models.json`，改成自己机器上的实际路径：

```json
{
  "upscale":    "~/ComfyUI/models/upscale_models/RealESRGAN_x4plus_anime_6B.pth",
  "sd":         "~/stable-diffusion-webui-forge/models/Stable-diffusion/Realistic_Vision_V5.1.safetensors",
  "controlnet": "~/stable-diffusion-webui-forge/models/ControlNet/control_v11f1e_sd15_tile.pth"
}
```

`models.json` 已在 `.gitignore` 里，不会带本机路径进仓库。

---

### `denoise.py` — 降噪锐化

**它做什么**：**不放大**，只做降噪和锐化。针对低光噪点图。

> ⚠️ 这条路实测**效果有限**。降噪的本质是抹掉微小差异，而边缘也属于微小差异——所以降噪必然降低锐度。这个脚本只适合"图够用，只是噪点多"的情况。

**流程**：NLM 降噪 → CLAHE 局部对比 → 带边缘保护的 unsharp mask

**怎么用**：

```bash
python denoise.py 图片.jpg 输出目录 --preset gentle
```

| 预设 | 降噪 h | 锐化 | CLAHE clip | 适用 |
| --- | --- | --- | --- | --- |
| `soft` | 3 | 0.5 | 0.4 | 尽量保细节 |
| `gentle` | 5 | 0.7 | 0.7 | **推荐** |
| `light` | 3 | 0.6 | 1.0 | 轻度 |
| `medium` | 6 | 1.0 | 2.0 | 中度（clip 偏高，可能不自然） |
| `strong` | 10 | 1.4 | 3.0 | 重度（很容易做成假面） |
| `all` | — | — | — | 三档全跑 |

**关键教训**：CLAHE 的 `clipLimit` 千万不能高。设在 2.0 以上会把皮肤做成"假面"，看起来反而更糊。

---

### `compare.py` — 客观测量

**它做什么**：算客观指标，对比不同版本的清晰度。不靠肉眼判断。

**怎么用**：

```bash
python compare.py 基准图.jpg 变体1.png 变体2.png ...
```

**输出**：

```
文件                        拉普拉斯      梯度      高频占比     相对基准      大小
原图.jpg                       4.7      7.64    0.00005     +0.0%      118K  ← 基准
超分4x.png                     5.1      4.56    0.00003     +9.5%    18323K
降噪中度.png                  14.8      8.09    0.00007   +218.2%     1114K
```

**指标怎么读**：

- **拉普拉斯越高** = 局部微对比越强 = 人眼越觉得锐
- **但拉普拉斯也能被噪点拉高**，所以要和梯度一起看
- **梯度下降而拉普拉斯上升** = 锐度是硬拉上去的，不是真细节

> 所有变体都会先缩放到基准图的尺寸再测量，保证可比。

---

## 实测结论

这是整个项目最有价值的部分——三类图片的实际对比。

### ✅ 重绘有效：有可锁的结构

原图小但有明确主体轮廓（手指、姿态、高光、文字水印）：

| 原图 | 输出 | 效果 |
| --- | --- | --- |
| 256×455 / 21 KB | 1024×1820 | 皮肤纹理、毛发层次全部出来 |
| 256×342 / 11 KB | 1024×1368 | 指甲高光成形，背景噪点压掉 |
| 1024×576 / 93 KB | 直接重绘 | 质感明显提升 |

### ❌ 重绘反而变差：画面以平滑渐变为主

1365×1024 但只有 118 KB 的低光虚焦图：

1. 原图几乎没有边缘（`edge_density = 0.0001`）
2. ControlNet 没有东西可以锁
3. prompt 又要求 `natural skin texture`
4. → **模型在平滑区凭空画出纹理**
5. → 原本柔和的光影过渡被画成一块块色块分界

**结论：这种图只做超分，别碰重绘。**

### 📊 客观指标对照

| 图 | 尺寸 | 文件 | 边缘密度 | 拉普拉斯 | 实测 |
| --- | --- | --- | --- | --- | --- |
| 虚焦A | 1365×1024 | 118 KB | 0.00007 | 4.7 | ❌ 重绘差 |
| 虚焦B | 1365×1024 | 127 KB | 0.0007 | 5.7 | ❌ 重绘差 |
| 结构A | 256×455 | 21 KB | 0.0247 | 185.7 | ✅ 重绘好 |
| 结构B | 455×256 | 20 KB | 0.0172 | 232.0 | ✅ 重绘好 |
| 结构C | 256×342 | 12 KB | 0.0239 | 16.9 | ✅ 重绘好 |
| 结构D | 256×342 | 11 KB | 0.0270 | 21.8 | ✅ 重绘好 |
| 大图 | 1024×576 | 93 KB | 0.0321 | 273.4 | ✅ 重绘好 |
| 大图B | 910×512 | 105 KB | 0.0343 | 89.3 | ✅ 重绘好 |

**分界线在 `edge_density ≈ 0.010`。**

低于它，ControlNet 无事可做，重绘就是让模型自由发挥。

### 超分的极限

对虚焦大图做超分，客观指标：

```
                拉普拉斯    梯度均值
原图（1365×1024）   4.7      7.64
超分 4×             5.1      4.56    ← 梯度反而降了 40%
```

**超分把边缘抹平了。** 看起来"细腻"只是尺寸变大后缩略显示的效果。

118 KB 塞 1365×1024 的图，信息在 JPEG 编码时就丢了，超分补不回来。

---

## 快速开始

### 环境要求

- Windows 10/11 + WSL2（Ubuntu）
- NVIDIA GPU，CUDA 11.8+（实测 RTX 4060 Laptop 8 GB 够用）
- 显存不足时用 `--tile` 切块，别硬撑

### 安装

```bash
# WSL 里
conda create -n comfyui python=3.11 -y
conda activate comfyui

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 下载模型

```bash
# 超分模型（17 MB，实测效果最好）
wget -P ~/ComfyUI/models/upscale_models/ \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth

# ControlNet Tile（1.4 GB，支持代理和断点续传）
bash download_controlnet.sh ~/ComfyUI/models/ControlNet http://127.0.0.1:7890

# SD 底模：自己准备一个写实系 SD1.5 模型
#   推荐 Realistic Vision V5.1 / V6.0
```

### 配置路径

```bash
cp models.example.json models.json
# 编辑 models.json，改成实际路径
```

### 开始用

```bash
PY=python
KIT=/path/to/photo-upscale-kit

# 先判断看看
$PY $KIT/auto.py 图片.jpg --dry

# 确认没问题再执行
$PY $KIT/auto.py 图片.jpg

# 批量
$PY $KIT/auto.py ~/图片/ --outdir ~/成品/
```

---

## 实现要点（踩过的坑）

### 1. Windows 中文路径

`cv2.imread` 处理不了 `C:\Users\xxx\桌面\` 这类路径，返回 `None` 且不报错。

```python
def imread_unicode(path: Path):
    buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def imwrite_unicode(path: Path, img, ext: str, quality: int = 96):
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    path.write_bytes(buf.tobytes())
```

### 2. Real-ESRGAN 三种权重结构不同

| 权重 | 结构 | 关键差异 |
| --- | --- | --- |
| `x4plus` | RRDB, 23 块 | 标准 |
| `anime_6B` | RRDB, **6** 块 | 块数少，但 nf 还是 64 |
| `general-x4v3` | **SRVGGNetCompact** | 完全不同，**输出是残差** |

`load_model()` 按权重键的形状自动推断，不写死参数。

### 3. 分块接缝

第一版 `tiles()` 的 `range()` 终点算错，产生 448px 的异常重叠和不规则窄条，拼接后有明显网格。

正确做法见 [redraw.py 分块接缝](#redrawpy--重绘)。

### 4. 噪点被当成纹理

低光图的噪点，SD 会当成皮肤纹理画实，出来一片颗粒。

**解决**：超分前先做高斯降噪（`--pre-denoise 1.2`）。

### 5. 8 GB 显存 OOM

5460×4096 的图一次过需要 5.7 GB 连续显存，会 OOM。

**解决**：`upscale_wsl.py --tile 384`，`redraw.py --tile 512`（本来就是分块的）。

### 6. 下载模型需要代理

HuggingFace 在国内经常 SSL 断连。`download_controlnet.sh` 支持：

- 传代理地址：`bash download_controlnet.sh <目录> http://127.0.0.1:7890`
- 自动回退镜像站：`huggingface.co` → `hf-mirror.com`
- 断点续传：`curl -C -`，中断后重跑会接着下

WSL2 用 mirrored 网络模式时，`127.0.0.1` 可以直通 Windows 的代理端口。

---

## 已知限制

### 本质限制（改不掉）

- **超分不能凭空造信息**。118 KB 塞 1365×1024 的图，信息在 JPEG 编码时就没了
- **重绘的细节是模型补的**，不是原图还原的。皮肤纹理、毛发细节来自训练数据，可能与真实情况不同（脸型微变、纹路不符）
- **降噪必然降低锐度**。边缘信息也属于微小差异，抹噪点的同时会抹掉一部分边缘

### 实现限制（可以改）

- 只测过 SD1.5 + Realistic Vision；SDXL 在 8 GB 显存上跑重绘很勉强，没跑通
- GFPGAN / CodeFormer 人脸修复没测试
- prompt 是硬编码的默认值，没有针对不同题材调优
- 批量处理是串行的，没做并发

---

## 什么时候别用这套

| 场景 | 原因 |
| --- | --- |
| 需要**忠实还原**（证据、档案、医疗影像） | 重绘会改内容，超分也补不出真实细节 |
| 原图信息量已经很高 | 没必要，反而可能引入模型偏差 |
| 只有 CPU | SD 重绘会慢到不可接受（几十分钟一张） |
| 原图是平滑虚焦、大面积渐变 | 只能超分，重绘帮倒忙 |

---

## 参考

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — 超分模型
- [ControlNet-v1-1](https://huggingface.co/lllyasviel/ControlNet-v1-1) — Tile 模型
- [Realistic Vision](https://civitai.com/models/4201) — 写实 SD1.5 底模
- [diffusers](https://github.com/huggingface/diffusers) — 推理框架

详细探索记录见 [FINDINGS.md](FINDINGS.md)，原理解释见 [HOW-IT-WORKS.md](HOW-IT-WORKS.md)。

## License

MIT
