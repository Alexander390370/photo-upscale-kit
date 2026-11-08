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

**自动判断该超分还是重绘**——给低分辨率 / 低质量照片做放大与修复，跑在 **Windows + WSL2 + NVIDIA GPU** 上。

核心思路：**不同的图该走不同的路**，硬套一种方法只会越弄越糟。仓库自带一个分类器，测量图片特征后自动选方案。

> ⚠️ **一句话结论**：边缘密度 < 0.01 的平滑虚焦图**不要重绘**，模型会凭空补出假纹理，越弄越糊。

## 三条路，别混着用

| 方案 | 做法 | 什么时候用 |
| --- | --- | --- |
| **仅超分** | Real-ESRGAN 4× 放大 | 原图结构清晰、只想变大；或原图太平滑，重绘会画蛇添足 |
| **仅重绘** | SD img2img + ControlNet Tile，不放大 | 原图够大，只需要提升质感 |
| **超分+重绘** | 先 4× 超分，再分块重绘 | 原图很小（<400k 像素）但有结构，收益最大 |

## 实测结论（这是本项目最有价值的部分）

三类图片的实测对比：

### ✅ 重绘有效的图：有可锁的结构

原图小但有明确主体轮廓（手指、姿态、高光、文字水印）：

- 256×455 / 21 KB → 1024×1820，皮肤纹理、毛发层次全部出来
- 256×342 / 11 KB → 1024×1368，指甲高光成形，背景噪点压掉
- 1024×576 / 93 KB → 直接重绘，质感明显提升

### ❌ 重绘反而变差的图：画面以平滑渐变为主

1365×1024 但只有 118 KB 的低光虚焦图：

- 原图几乎没有边缘（边缘密度 0.0001）
- SD 看不到纹理，但 prompt 要求 "natural skin texture"，于是**凭空补出假毛孔和硬边**
- 原本柔和的光影过渡被画成一块块的色块分界
- **结论：这种图只做超分，别碰重绘**

### 📊 客观指标对照

| 图 | 尺寸 | 边缘密度 | 拉普拉斯 | 实测 |
| --- | --- | --- | --- | --- |
| 虚焦A | 1365×1024 | 0.00007 | 4.7 | ❌ 重绘差 |
| 虚焦B | 1365×1024 | 0.0007 | 5.7 | ❌ 重绘差 |
| 结构A | 256×455 | 0.0247 | 185.7 | ✅ 重绘好 |
| 结构B | 256×342 | 0.0239 | 16.9 | ✅ 重绘好 |
| 大图 | 1024×576 | 0.0321 | 273.4 | ✅ 重绘好 |

**分界线在边缘密度 ≈ 0.01**。低于它，ControlNet 没有东西可以锁，重绘就是让模型自由发挥。

## 快速开始

### 环境要求

- Windows + WSL2（Ubuntu）
- NVIDIA GPU（实测 RTX 4060 Laptop 8 GB 够用）
- 显存不足时用 `--tile` 切块，别硬撑

### 安装

```bash
# WSL 里
conda create -n comfyui python=3.11 -y
conda activate comfyui

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python-headless pillow numpy diffusers transformers accelerate safetensors
```

### 下载模型

```bash
# 超分模型（17 MB，对带结构的图效果最好）
wget -P ~/ComfyUI/models/upscale_models/ \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth

# ControlNet Tile（1.4 GB）
bash download_controlnet.sh

# SD 底模：自己准备一个写实系 SD1.5 模型放这
#   ~/stable-diffusion-webui-forge/models/Stable-diffusion/
```

`auto.py` 里的路径是写死的，按自己的实际路径改 `MODELS` 字典。

### 用法

```bash
PY=/home/alxai/miniconda3/envs/comfyui/bin/python
KIT=/path/to/photo-upscale-kit

# 只判断，不执行（先看看它打算怎么做）
$PY $KIT/auto.py 图片.jpg --dry

# 自动判断并执行
$PY $KIT/auto.py 图片.jpg

# 批量处理一个目录
$PY $KIT/auto.py ~/图片/ --outdir ~/成品/

# 强制指定方案
$PY $KIT/auto.py 图片.jpg --force redraw_only
```

## 各脚本说明

| 文件 | 作用 |
| --- | --- |
| `classify.py` | 测量图片特征，判断该走哪条路（可单独用，输出人类可读表格或 `--json`） |
| `auto.py` | 一键入口：分类 → 执行 |
| `upscale_wsl.py` | Real-ESRGAN 超分，自动适配三种权重结构（RRDB x4plus / anime 6B / SRVGG general） |
| `redraw.py` | SD img2img 分块重绘，带 ControlNet Tile 和羽化拼接 |
| `denoise.py` | 降噪+锐化（不放大），针对低光噪点图 |
| `compare.py` | 客观测量：拉普拉斯方差、梯度均值、高频占比 |
| `download_controlnet.sh` | 下载 ControlNet Tile |

## 实现要点（踩过的坑）

### 1. Node/Python 的 Windows 中文路径

`cv2.imread` 处理不了 `桌面` 这类中文路径，要用 `np.frombuffer(path.read_bytes())` + `cv2.imdecode` 绕过。

### 2. Real-ESRGAN 三种权重的结构不同

- `x4plus` / `anime_6B`：RRDB，23 / 6 个块，`nf=64`
- `general-x4v3`：SRVGGNetCompact，轻量，**输出的是残差**，必须加回双三次放大的原图，否则画面接近全黑

`load_model()` 按权重键的形状自动推断结构，不用手填参数。

### 3. 分块重绘的接缝

- 步长必须是 `size - overlap`，最后一块贴边对齐，否则会出现窄条和网格
- 每块用羽化权重图混合（中间 1，边缘线性衰减到 0），最后加权平均

### 4. 低光图的噪点会被 SD 当成纹理

必须先做高斯降噪（`--pre-denoise 1.2`），否则 SD 会把噪点画成实心的颗粒。

### 5. 8 GB 显存跑 4× 大图会 OOM

`upscale_wsl.py --tile 384` 开启切块；`redraw.py` 本身按 `--tile 512` 分块。

## 已知限制

- **细节是模型补的，不是原图还原的**。重绘路线的本质是"照着原图结构重新画一张"，皮肤纹理、毛发细节来自训练数据，可能与真实情况不同（脸型微变、纹路不符）
- **超分不能凭空造信息**。118 KB 塞 1365×1024 的图，信息在 JPEG 编码时就没了，后期补不回来
- 只测过 SD1.5 + Realistic Vision；SDXL 显存吃紧，没跑通
- 脚本路径写死了作者本机的模型位置，移植需要改

## 什么时候别用这套

- 需要**忠实还原**（比如证据、档案）——重绘会改内容，超分也补不出真实细节
- 原图本身信息量已经很高——没必要
- 只有 CPU——SD 重绘会慢到不可接受

## License

MIT
