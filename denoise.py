"""
降噪 + 锐化（不放大，保持原尺寸）

针对低光、高噪点、对焦不足的照片。与超分的区别：
  - 超分：增加像素，靠猜补细节，对糊图反而更糊
  - 本脚本：在原有像素上压制噪点、恢复局部对比，观感更干净

用法：
    python denoise.py 输入图 [输出目录] [--preset 名]

预设：
    light   轻度：保住细节，噪点去一半
    medium  中度（默认）：明显的降噪 + 适度锐化
    strong  重度：噪点压得很狠，细节损失较多
    all     全部跑一遍，出三份对比

依赖：opencv-python, numpy, pillow
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# 预设：(降噪强度 h, 锐化强度 amount, 局部对比 clipLimit, 锐化阈值 threshold)
# clip 越低越自然；低光皮肤照 0.4-0.8 就够，高了会破坏柔和过渡，看起来反而更硬更糊。
PRESETS = {
    "soft":   {"h": 4,  "amount": 0.5, "clip": 0.4, "thr": 8, "label": "柔和"},
    "gentle": {"h": 5,  "amount": 0.7, "clip": 0.7, "thr": 8, "label": "温和"},
    "light":  {"h": 3,  "amount": 0.6, "clip": 1.0, "thr": 6, "label": "轻度"},
    "medium": {"h": 6,  "amount": 1.0, "clip": 2.0, "thr": 3, "label": "中度"},
    "strong": {"h": 10, "amount": 1.4, "clip": 3.0, "thr": 3, "label": "重度"},
}


def unsharp(img: np.ndarray, amount: float, radius: int = 2, threshold: int = 3) -> np.ndarray:
    """Unsharp mask：原图 + (原图 - 模糊) * amount，只锐化超过 threshold 的边缘。"""
    if amount <= 0:
        return img
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    # 用 float 避免 uint8 溢出
    sharp = img.astype(np.float32) + (img.astype(np.float32) - blurred.astype(np.float32)) * amount
    out = np.clip(sharp, 0, 255).astype(np.uint8)
    if threshold > 0:
        # 差值小于阈值的区域保留降噪后的样子，避免把噪点也锐化出来
        diff = cv2.absdiff(img, blurred)
        mask = (diff > threshold).astype(np.float32)
        # 逐通道判断：只有该通道的边缘明显才锐化它，避免色彩溢出
        mask = cv2.GaussianBlur(mask, (0, 0), 1)
        out = (out.astype(np.float32) * mask + img.astype(np.float32) * (1 - mask)).astype(np.uint8)
    return out


def imread_unicode(path: Path) -> np.ndarray | None:
    """Windows 上 cv2.imread 处理不了中文路径，改用字节流解码。"""
    try:
        buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img: np.ndarray, ext: str, quality: int = 96) -> bool:
    """同理，cv2.imwrite 也要绕开中文路径。"""
    params = [cv2.IMWRITE_JPEG_QUALITY, quality, cv2.IMWRITE_JPEG_OPTIMIZE, 1] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext if ext else ".png", img, params)
    if not ok:
        return False
    path.write_bytes(buf.tobytes())
    return True


def process(src: Path, dst: Path, h: int, amount: float, clip: float, thr: int = 8, quality: int = 96) -> None:
    img = imread_unicode(src)
    if img is None:
        raise SystemExit(f"读不了图片：{src}")

    # 1) 彩色非局部均值降噪：对皮肤这类大面积平滑区域尤其有效
    denoised = cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)

    # 2) CLAHE 局部对比：低光图整体发灰，这一步把压在暗部的层次提出来
    #    clip 必须压得很低（0.4-0.8），否则会把皮肤做成假面，观感反而更糊
    if clip > 0:
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l)
        denoised = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # 3) 锐化（带边缘保护，不会把残余噪点放大）
    result = unsharp(denoised, amount, threshold=thr)

    if not imwrite_unicode(dst, result, dst.suffix.lower(), quality):
        raise SystemExit(f"写不了图片：{dst}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="降噪 + 锐化（保持原尺寸）")
    ap.add_argument("input", type=Path, help="输入图片")
    ap.add_argument("outdir", type=Path, nargs="?", help="输出目录；省略则与原图同目录")
    ap.add_argument("--preset", default="gentle", choices=[*PRESETS, "all"], help="预设强度")
    ap.add_argument("--format", default="png", choices=["png", "jpg"], help="输出格式")
    ap.add_argument("--quality", type=int, default=96, help="jpg 质量")
    args = ap.parse_args(argv)

    if not args.input.is_file():
        print(f"找不到输入：{args.input}", file=sys.stderr)
        return 1

    outdir = args.outdir or args.input.parent
    outdir.mkdir(parents=True, exist_ok=True)

    names = list(PRESETS) if args.preset == "all" else [args.preset]
    for name in names:
        cfg = PRESETS[name]
        dst = outdir / f"{args.input.stem}_v2_{name}.{args.format}"
        process(args.input, dst, cfg["h"], cfg["amount"], cfg["clip"], cfg["thr"], args.quality)
        print(f"{cfg['label']:4s} → {dst.name}  ({dst.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
