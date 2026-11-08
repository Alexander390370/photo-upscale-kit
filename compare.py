"""
客观测量清晰度：对每个变体，先统一缩放到同一尺寸再算指标。

指标说明：
  lap_var   拉普拉斯方差 —— 越高说明局部微对比越强，人眼越觉得"锐"
  grad_mean 梯度幅值均值 —— 边缘能量
  hf_ratio  高频能量占比 —— 细节在整图能量中的比例
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path: Path) -> np.ndarray | None:
    try:
        buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def metrics(img: np.ndarray, target: tuple[int, int]) -> tuple[float, float, float]:
    """缩放到 target 尺寸后测量，保证可比。"""
    resized = cv2.resize(img, target, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_var = float(lap.var())

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mean = float(np.sqrt(gx * gx + gy * gy).mean())

    # 高频能量：原图减去模糊版，取平方均值
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.5)
    hf = (gray - blurred).astype(np.float32)
    hf_ratio = float((hf * hf).mean() / max((gray * gray).mean(), 1e-6))

    return lap_var, grad_mean, hf_ratio


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("用法: compare.py 基准图 变体1 [变体2 ...]", file=sys.stderr)
        return 2
    base_path = Path(argv[0])
    base = imread_unicode(base_path)
    if base is None:
        print(f"读不了基准图：{base_path}", file=sys.stderr)
        return 1
    target = (base.shape[1], base.shape[0])
    print(f"统一缩放到 {target[0]} x {target[1]} 后比较\n")

    rows: list[tuple[str, float, float, float, int]] = []
    b = metrics(base, target)
    rows.append((base_path.name, *b, base_path.stat().st_size))

    for raw in argv[1:]:
        p = Path(raw)
        if not p.is_file():
            print(f"跳过（不存在）：{p.name}", file=sys.stderr)
            continue
        img = imread_unicode(p)
        if img is None:
            print(f"跳过（读不了）：{p.name}", file=sys.stderr)
            continue
        m = metrics(img, target)
        rows.append((p.name, *m, p.stat().st_size))

    # 相对基准的百分比
    base_lap, base_grad, base_hf = base[0] if False else rows[0][1:4]
    print(f"{'文件':<34} {'拉普拉斯':>10} {'梯度':>9} {'高频占比':>10} {'相对基准':>9} {'大小':>9}")
    print("-" * 88)
    for name, lap, grad, hf, size in rows:
        rel = (lap / base_lap - 1) * 100 if base_lap else 0
        flag = "  ← 基准" if name == rows[0][0] else ""
        print(f"{name:<34} {lap:>10.1f} {grad:>9.2f} {hf:>10.5f} {rel:>+8.1f}% {size/1024:>8.0f}K{flag}")
    print("\n拉普拉斯越高 = 局部微对比越强 = 人眼越觉得锐。负数说明比原图更糊。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
