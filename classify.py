#!/usr/bin/env python3
"""
自动判断该走哪条路：仅超分 / 仅重绘 / 超分+重绘 / 不处理

判断依据（都在原图尺寸上算，不做任何缩放）：
  edge_density  边缘密度    —— Canny 检出的边缘像素占比，衡量"有多少可锁的结构"
  lap_var       拉普拉斯方差 —— 局部微对比强度
  grad_mean     梯度均值    —— 整体边缘能量
  hf_ratio      高频占比    —— 细节在整图能量中的比例

经验阈值（来自实测：1476/1485 虚焦图 vs 2627/1912/1526 有结构图）：
  edge_density < 0.02  → 平滑/虚焦，ControlNet 锁不住结构，重绘会画蛇添足
  edge_density ≥ 0.05  → 结构充足，重绘收益最大
  中间地带看 lap_var

用法：
  python classify.py 图片 [更多图片...]
  python classify.py --json 图片     # 机器可读，供 auto.py 调用
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path: Path) -> np.ndarray | None:
    try:
        buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


@dataclass
class Features:
    width: int
    height: int
    edge_density: float
    lap_var: float
    grad_mean: float
    hf_ratio: float
    kb: float


def measure(img: np.ndarray, size_bytes: int) -> Features:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Canny 的阈值按图中位亮度自适应，避免全黑图/过曝图判错
    median = float(np.median(gray))
    lo = int(max(0, 0.66 * median))
    hi = int(min(255, 1.33 * median))
    edges = cv2.Canny(gray, lo, hi)
    edge_density = float((edges > 0).mean())

    g = gray.astype(np.float32)
    lap_var = float(cv2.Laplacian(g, cv2.CV_32F).var())
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad_mean = float(np.sqrt(gx * gx + gy * gy).mean())

    blurred = cv2.GaussianBlur(g, (0, 0), 1.5)
    hf = g - blurred
    hf_ratio = float((hf * hf).mean() / max((g * g).mean(), 1e-6))

    return Features(w, h, edge_density, lap_var, grad_mean, hf_ratio, size_bytes / 1024)


def classify(f: Features) -> tuple[str, str]:
    """返回 (建议, 理由)。

    阈值来自实测校准（每张都跑过超分+ControlNet 重绘，肉眼确认效果）：
      边缘密度 < 0.010  → 平滑/虚焦，重绘会补出假纹理（1476: 0.00007, 1485: 0.0007）
      边缘密度 ≥ 0.015  → 有可锁结构，重绘可行（2627: 0.025, 1912: 0.024, 1526: 0.032）
      0.010–0.015       → 看拉普拉斯兜底
    """
    px = f.width * f.height

    if f.edge_density < 0.010:
        return "upscale_only", (
            f"边缘密度仅 {f.edge_density:.4f}，画面以平滑渐变为主，"
            "ControlNet 锁不住结构，重绘会补出假纹理")

    if f.edge_density < 0.015:
        # 灰区：细节实在太少就别重绘了
        if f.lap_var < 30:
            return "upscale_only", (
                f"边缘密度 {f.edge_density:.4f}，拉普拉斯仅 {f.lap_var:.0f}，"
                "细节不足以支撑重绘")
        return ("upscale_then_redraw" if px < 400_000 else "redraw_only",
                f"边缘密度 {f.edge_density:.4f}，拉普拉斯 {f.lap_var:.0f}，可以重绘")

    # 结构充足
    if px < 400_000:
        return "upscale_then_redraw", (
            f"边缘密度 {f.edge_density:.4f} 结构充足，原图仅 {f.width}x{f.height}，"
            "先超分再重绘收益最大")
    return "redraw_only", (
        f"边缘密度 {f.edge_density:.4f} 结构充足，原图已有 {f.width}x{f.height}，"
        "直接重绘提质感即可，再放大是虚胖")

    # 注：仍保留不处理的出口给后续扩展（如纯色图）


ADVICE_LABEL = {
    "upscale_only": "仅超分",
    "redraw_only": "仅重绘",
    "upscale_then_redraw": "超分+重绘",
    "skip": "不处理",
}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="判断图片该走超分还是重绘")
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args(argv)

    results = []
    for p in args.images:
        if not p.is_file():
            print(f"跳过（不存在）：{p}", file=sys.stderr)
            continue
        img = imread_unicode(p)
        if img is None:
            print(f"跳过（读不了）：{p}", file=sys.stderr)
            continue
        f = measure(img, p.stat().st_size)
        advice, why = classify(f)
        results.append({"file": str(p), "features": asdict(f), "advice": advice, "label": ADVICE_LABEL[advice], "reason": why})

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    print(f"{'文件':<24} {'尺寸':>11} {'边缘密度':>9} {'拉普拉斯':>9} {'梯度':>7} {'建议':>10}")
    print("-" * 82)
    for r in results:
        f = r["features"]
        print(f"{Path(r['file']).name:<24} {f['width']:>5}x{f['height']:<5} "
              f"{f['edge_density']:>9.3f} {f['lap_var']:>9.1f} {f['grad_mean']:>7.2f} "
              f"{r['label']:>10}")
    print()
    for r in results:
        print(f"  {Path(r['file']).name}")
        print(f"    → {r['label']}：{r['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
