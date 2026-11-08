#!/usr/bin/env python3
"""
重绘放大 / 修复（img2img + 分块）

流程：
  1. 先超分到目标尺寸（Real-ESRGAN），得到结构基础
  2. 分块送进 SD 做 img2img，低重绘强度只精修不改变内容
  3. 块间用羽化混合，避免接缝

与单纯超分的区别：SD 会按训练数据补出真实的皮肤纹理、毛发、光影，
所以看起来"真清楚"。代价是细节是模型编的，不是原图还原。

用法：
  python redraw.py 输入 输出 --model <safetensors> [--scale 4] [--denoise 0.35]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DEFAULT_NEG = (
    "blurry, lowres, jpeg artifacts, pixelated, watermark, text, logo, "
    "deformed, extra limbs, bad anatomy, bad hands, poorly drawn, "
    "oversaturated, oversharpened, plastic skin, cartoon, anime, 3d render"
)


def load_pipeline(model_path: Path, device: str, controlnet_path: Path | None = None):
    from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionControlNetImg2ImgPipeline
    from diffusers import ControlNetModel

    dtype = torch.float16 if device == "cuda" else torch.float32
    common = dict(torch_dtype=dtype, safety_checker=None, requires_safety_checker=False)

    if controlnet_path is not None:
        cn = ControlNetModel.from_single_file(str(controlnet_path), torch_dtype=dtype)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
            str(model_path), controlnet=cn, **common)
    else:
        pipe = StableDiffusionImg2ImgPipeline.from_single_file(str(model_path), **common)

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    except Exception:
        pass
    return pipe


def _axis_positions(total: int, size: int, overlap: int) -> list[int]:
    """一条轴上的起点：步长 = size - overlap，最后一块贴边对齐，不产生窄条。"""
    if total <= size:
        return [0]
    step = max(size - overlap, 1)
    pos = list(range(0, total - size + 1, step))
    if pos[-1] != total - size:
        pos.append(total - size)
    return pos


def tiles(w: int, h: int, size: int, overlap: int):
    """按 size 切块，相邻块重叠 overlap，返回 (x, y, w, h)。"""
    for y in _axis_positions(h, size, overlap):
        for x in _axis_positions(w, size, overlap):
            yield x, y, min(size, w - x), min(size, h - y)


def feather_mask(size: int, overlap: int, device: str, dtype) -> torch.Tensor:
    """中间为 1、边缘按 overlap 线性衰减的权重图。"""
    m = torch.ones((1, 1, size, size), device=device, dtype=dtype)
    if overlap > 0:
        ramp = torch.linspace(0, 1, overlap, device=device, dtype=dtype)
        v = ramp.view(1, 1, overlap, 1)
        h = ramp.view(1, 1, 1, overlap)
        m[:, :, :overlap, :] *= v
        m[:, :, -overlap:, :] *= v.flip(2)
        m[:, :, :, :overlap] *= h
        m[:, :, :, -overlap:] *= h.flip(3)
    return m


def redraw(pipe, img: Image.Image, prompt: str, negative: str, denoise: float,
           tile: int, overlap: int, steps: int, guidance: float, seed: int,
           device: str, use_control: bool = False, cn_scale: float = 1.0) -> Image.Image:
    w, h = img.size
    gen = torch.Generator(device=device).manual_seed(seed)
    dtype = torch.float16 if device == "cuda" else torch.float32

    accum = torch.zeros((1, 3, h, w), device=device, dtype=dtype)
    weight = torch.zeros((1, 1, h, w), device=device, dtype=dtype)
    mask_full = feather_mask(tile, overlap, device, dtype)

    total = len(list(tiles(w, h, tile, overlap)))
    for i, (x, y, tw, th) in enumerate(tiles(w, h, tile, overlap), 1):
        crop = img.crop((x, y, x + tw, y + th))
        # 小于 tile 的块补齐，SD 需要固定尺寸
        if tw < tile or th < tile:
            canvas = Image.new("RGB", (tile, tile))
            canvas.paste(crop, (0, 0))
            crop = canvas
        kwargs = dict(prompt=prompt, negative_prompt=negative, image=crop,
                      strength=denoise, num_inference_steps=steps,
                      guidance_scale=guidance, generator=gen)
        if use_control:
            # Tile 控制：把原图自己作为条件，逼 SD 保持结构
            kwargs["control_image"] = crop
            kwargs["controlnet_conditioning_scale"] = cn_scale
        out = pipe(**kwargs).images[0]
        t = torch.from_numpy(np.array(out)).float().div(255).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
        m = mask_full[:, :, :th, :tw]
        accum[:, :, y:y + th, x:x + tw] += t[:, :, :th, :tw] * m
        weight[:, :, y:y + th, x:x + tw] += m
        print(f"  tile {i}/{total}  ({x},{y})", flush=True)

    weight = weight.clamp(min=1e-6)
    merged = (accum / weight).clamp(0, 1)
    arr = (merged.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="重绘放大（SD img2img 分块）")
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--model", type=Path, required=True, help="SD .safetensors 底模")
    ap.add_argument("--scale", type=float, default=4, help="先超分到几倍（默认 4）")
    ap.add_argument("--denoise", type=float, default=0.35,
                    help="重绘强度：0.2 保守（几乎不改），0.5 明显，0.7 大幅重画")
    ap.add_argument("--tile", type=int, default=512, help="SD 处理的块大小")
    ap.add_argument("--overlap", type=int, default=64, help="块重叠像素")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--prompt", default="a highly detailed photograph, sharp focus, natural skin texture, fine details, photorealistic")
    ap.add_argument("--negative", default=DEFAULT_NEG)
    ap.add_argument("--skip-upscale", action="store_true", help="跳过前置超分，直接对原图重绘")
    ap.add_argument("--pre-denoise", type=float, default=0.0,
                    help="超分前的高斯降噪半径（像素）；低光噪点图建议 0.8-1.5，"
                         "否则 SD 会把噪点当成皮肤纹理画实")
    ap.add_argument("--controlnet", type=Path, default=None,
                    help="ControlNet Tile 模型路径；给了就用它锁住结构")
    ap.add_argument("--cn-scale", type=float, default=1.0,
                    help="ControlNet 强度，1.0 完全约束结构，0.7 允许更多自由")
    args = ap.parse_args(argv)

    if not args.input.is_file():
        print(f"找不到输入：{args.input}", file=sys.stderr)
        return 1
    if not args.model.is_file():
        print(f"找不到模型：{args.model}", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备 : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    img = Image.open(args.input).convert("RGB")
    print(f"输入 : {img.width} x {img.height}")

    # 前置降噪：不先抹掉噪点，SD 会把它当成纹理强化，出来一片颗粒
    if args.pre_denoise > 0:
        from PIL import ImageFilter
        img = img.filter(ImageFilter.GaussianBlur(radius=args.pre_denoise))
        print(f"降噪 : 高斯 sigma={args.pre_denoise}")

    if not args.skip_upscale and args.scale != 1:
        print(f"步骤 1: 超分 {args.scale}x ...")
        import subprocess
        up = args.output.with_name(args.output.stem + "_up.png")
        here = Path(__file__).parent
        # 降噪后的图先落盘，超分脚本按路径读
        src = args.input
        if args.pre_denoise > 0:
            src = args.output.with_name(args.output.stem + "_denoised.png")
            img.save(src)
        subprocess.run([sys.executable, str(here / "upscale_wsl.py"), str(src), str(up),
                        "--scale", str(args.scale), "--tile", "384"], check=True)
        img = Image.open(up).convert("RGB")
        print(f"       → {img.width} x {img.height}")

    print(f"步骤 2: SD 重绘 (denoise={args.denoise}, tile={args.tile}"
          + (f", ControlNet={args.controlnet.name}@{args.cn_scale}" if args.controlnet else ", 无 ControlNet") + ") ...")
    pipe = load_pipeline(args.model, device, args.controlnet)
    out = redraw(pipe, img, args.prompt, args.negative, args.denoise,
                 args.tile, args.overlap, args.steps, args.guidance, args.seed, device,
                 use_control=args.controlnet is not None, cn_scale=args.cn_scale)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.output)
    print(f"输出 : {out.width} x {out.height}  ({args.output})")
    print(f"文件 : {args.output.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
