#!/usr/bin/env python3
"""
WSL 端超分：直接加载 Real-ESRGAN 权重，用 GPU 跑。

与 Windows 版逻辑相同，但跑在 WSL 的 comfyui 环境里（torch 2.14 + cu130）。
额外支持 --pre-blur：先做一次轻微降噪再超分，对低光噪点图效果更好。

用法：
    python upscale_wsl.py 输入 输出 [--scale 4] [--tile 0] [--pre-blur 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

DEFAULT_MODEL = Path("/home/alxai/ComfyUI/models/upscale_models/RealESRGAN_x4plus.pth")


def build_network(scale: int = 4, num_block: int = 23, num_feat: int = 64, num_grow_ch: int = 32):
    class ResidualDenseBlock(nn.Module):
        def __init__(self, nf=64, gc=32):
            super().__init__()
            self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
            self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
            self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
            self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
            self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, nf, gc=32):
            super().__init__()
            self.rdb1 = ResidualDenseBlock(nf, gc)
            self.rdb2 = ResidualDenseBlock(nf, gc)
            self.rdb3 = ResidualDenseBlock(nf, gc)

        def forward(self, x):
            out = self.rdb1(x)
            out = self.rdb2(out)
            out = self.rdb3(out)
            return out * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4):
            super().__init__()
            self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
            self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
            self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
            self.scale = scale

        def forward(self, x):
            feat = self.conv_first(x)
            feat = feat + self.conv_body(self.body(feat))
            feat = self.lrelu(self.conv_up1(
                nn.functional.interpolate(feat, scale_factor=2, mode="nearest")))
            if self.scale >= 4:
                feat = self.lrelu(self.conv_up2(
                    nn.functional.interpolate(feat, scale_factor=2, mode="nearest")))
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    return RRDBNet(nf=num_feat, nb=num_block, gc=num_grow_ch, scale=scale)


def build_srvgg(num_feat: int = 64, num_conv: int = 32, scale: int = 4):
    """SRVGGNetCompact：general-x4v3 用的轻量网络。

    权重键形如：body.0=conv, body.1=prelu, body.2=conv, body.3=prelu, ...
    最后一层 conv 输出 3*scale^2 通道，之后 pixel_shuffle。
    """

    class SRVGGNetCompact(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = scale
            layers = []
            layers.append(nn.Conv2d(3, num_feat, 3, 1, 1))
            layers.append(nn.PReLU(num_parameters=num_feat))
            for _ in range(num_conv):
                layers.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
                layers.append(nn.PReLU(num_parameters=num_feat))
            layers.append(nn.Conv2d(num_feat, 3 * scale * scale, 3, 1, 1))
            self.body = nn.Sequential(*layers)

        def forward(self, x):
            out = self.body(x)
            out = nn.functional.pixel_shuffle(out, self.scale)
            # 这个权重输出的是残差：要加回（双三次放大的）原图，否则画面接近全黑
            base = nn.functional.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
            return out + base

    return SRVGGNetCompact()


def load_model(path: Path, device: str):
    """按权重形状推断网络类型与规模，兼容 x4plus / anime_6B / general-x4v3。"""
    raw = torch.load(str(path), map_location="cpu", weights_only=True)
    state = raw
    if isinstance(raw, dict):
        state = raw.get("params_ema") or raw.get("params") or raw

    # SRVGG：没有 conv_first，且 body.N.weight 是 4 维卷积核
    if not any(k.startswith("conv_first") for k in state) and "body.0.weight" in state:
        conv_idx = sorted({int(k.split(".")[1]) for k in state
                           if k.startswith("body.") and k.split(".")[1].isdigit()
                           and state[k].dim() == 4})
        if conv_idx:
            num_conv = len(conv_idx) - 2
            num_feat = state["body.0.weight"].shape[0]
            net = build_srvgg(num_feat=num_feat, num_conv=num_conv, scale=4)
            net.load_state_dict(state, strict=True)
            net.eval()
            return net.to(device)

    # RRDB：body.<块>.rdb<N>.conv1.weight 等
    block_ids = set()
    for k in state:
        parts = k.split(".")
        if len(parts) > 3 and parts[0] == "body" and parts[1].isdigit():
            block_ids.add(int(parts[1]))
    num_block = max(block_ids) + 1 if block_ids else 23

    first = state.get("conv_first.weight")
    num_feat = first.shape[0] if first is not None else 64
    gc_w = state.get("body.0.rdb1.conv1.weight")
    num_grow_ch = gc_w.shape[0] if gc_w is not None else 32

    net = build_network(scale=4, num_block=num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
    net.load_state_dict(state, strict=True)
    net.eval()
    return net.to(device)


def upscale(img: Image.Image, net, device: str, tile: int = 0, tile_pad: int = 12) -> Image.Image:
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    s = net.scale

    with torch.inference_mode():
        if tile <= 0:
            out = net(tensor)
        else:
            b, c, h, w = tensor.shape
            out = torch.zeros((b, c, h * s, w * s), device=device)
            for y in range(0, h, tile):
                for x in range(0, w, tile):
                    y0, y1 = max(y - tile_pad, 0), min(y + tile + tile_pad, h)
                    x0, x1 = max(x - tile_pad, 0), min(x + tile + tile_pad, w)
                    res = net(tensor[:, :, y0:y1, x0:x1])
                    ey0, ex0 = (y - y0) * s, (x - x0) * s
                    tw, th = min(tile, w - x) * s, min(tile, h - y) * s
                    out[:, :, y * s:y * s + th, x * s:x * s + tw] = res[:, :, ey0:ey0 + th, ex0:ex0 + tw]

    out = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return Image.fromarray((out * 255.0).round().astype(np.uint8))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Real-ESRGAN 超分（WSL/CUDA）")
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--scale", type=float, default=4)
    ap.add_argument("--tile", type=int, default=0)
    ap.add_argument("--pre-blur", type=float, default=0.0,
                    help="超分前先做轻微高斯降噪（sigma），低光噪点图建议 0.5-1.0")
    args = ap.parse_args(argv)

    if not args.input.is_file():
        print(f"找不到输入：{args.input}", file=sys.stderr)
        return 1
    if not args.model.is_file():
        print(f"找不到模型：{args.model}", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备   : {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    img = Image.open(args.input)
    print(f"输入   : {img.width} x {img.height}")

    if args.pre_blur > 0:
        from PIL import ImageFilter
        img = img.filter(ImageFilter.GaussianBlur(radius=args.pre_blur))
        print(f"预处理 : 高斯降噪 sigma={args.pre_blur}")

    net = load_model(args.model, device)
    result = upscale(img, net, device, tile=args.tile)

    if args.scale != 4:
        result = result.resize(
            (round(img.width * args.scale), round(img.height * args.scale)), Image.LANCZOS)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output, quality=96, subsampling=0)
    print(f"输出   : {result.width} x {result.height}  ({args.output})")
    print(f"文件   : {args.output.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
