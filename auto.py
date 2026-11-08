#!/usr/bin/env python3
"""
一键处理：自动判断该走哪条路，然后执行。

用法：
    python auto.py 图片或目录 [--outdir 输出目录] [--force 方案]

--force 可选：upscale_only / redraw_only / upscale_then_redraw
不给就按 classify.py 的判断自动选。

在 WSL 里跑（需要 CUDA + diffusers + 模型）：
    wsl -d Ubuntu -- /home/alxai/miniconda3/envs/comfyui/bin/python \
        /mnt/c/.../tools/upscale/auto.py /mnt/c/Users/34246/Desktop/桌面/图.jpg
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
PY = sys.executable

# 模型路径：优先读同目录的 models.json，其次是下面的默认值。
# 换机器时只改 models.json，不用动代码。
_DEFAULTS = {
    "upscale": "~/ComfyUI/models/upscale_models/RealESRGAN_x4plus_anime_6B.pth",
    "sd": "~/stable-diffusion-webui-forge/models/Stable-diffusion/Realistic_Vision_V5.1.safetensors",
    "controlnet": "~/stable-diffusion-webui-forge/models/ControlNet/control_v11f1e_sd15_tile.pth",
}


def load_models() -> dict[str, Path]:
    cfg = HERE / "models.json"
    raw = dict(_DEFAULTS)
    if cfg.is_file():
        raw.update({k: v for k, v in json.loads(cfg.read_text("utf-8")).items()
                    if v and not k.startswith("_")})
    paths = {k: Path(v).expanduser() for k, v in raw.items()}
    missing = [f"{k}: {v}" for k, v in paths.items() if not v.is_file()]
    if missing:
        raise SystemExit("模型文件缺失（改 models.json 指向实际位置）：\n  " + "\n  ".join(missing))
    return paths


MODELS = load_models()

PROMPT = ("highly detailed photograph, sharp focus, natural skin texture, fine pores, "
          "soft studio lighting, professional photography")
NEGATIVE = ("blurry, lowres, jpeg artifacts, noise, grainy, watermark, text, cartoon, anime, "
            "deformed, oversaturated, plastic skin")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def run(cmd: list[str]) -> int:
    print(f"  $ {' '.join(str(c) for c in cmd[1:3])} ...", flush=True)
    r = subprocess.run(cmd)
    return r.returncode


def classify(path: Path) -> dict:
    out = subprocess.run([PY, str(HERE / "classify.py"), "--json", str(path)],
                         capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        raise RuntimeError(f"classify 失败: {out.stderr[:400]}")
    data = json.loads(out.stdout)
    if not data:
        raise RuntimeError("classify 没有返回结果")
    return data[0]


def do_upscale(src: Path, out: Path, scale: float = 4) -> int:
    return run([PY, str(HERE / "upscale_wsl.py"), str(src), str(out),
                "--model", str(MODELS["upscale"]), "--scale", str(scale), "--tile", "384"])


def do_redraw(src: Path, out: Path, denoise: float = 0.35, scale: float = 4,
              pre_denoise: float = 1.2) -> int:
    return run([PY, str(HERE / "redraw.py"), str(src), str(out),
                "--model", str(MODELS["sd"]), "--controlnet", str(MODELS["controlnet"]),
                "--cn-scale", "1.0", "--scale", str(scale), "--denoise", str(denoise),
                "--pre-denoise", str(pre_denoise), "--tile", "512", "--overlap", "128",
                "--steps", "28", "--guidance", "4.5",
                "--prompt", PROMPT, "--negative", NEGATIVE])


def process(path: Path, outdir: Path, force: str | None, dry: bool) -> bool:
    print(f"\n{'=' * 60}")
    print(f"文件: {path.name}  ({path.stat().st_size / 1024:.0f} KB)")

    info = classify(path)
    f = info["features"]
    print(f"测量: {f['width']}x{f['height']}  边缘={f['edge_density']:.4f}  "
          f"拉普拉斯={f['lap_var']:.0f}")

    advice = force or info["advice"]
    label = {"upscale_only": "仅超分", "redraw_only": "仅重绘",
             "upscale_then_redraw": "超分+重绘", "skip": "跳过"}.get(advice, advice)
    print(f"方案: {label}")
    print(f"理由: {info['reason']}")

    if dry or advice == "skip":
        return True

    outdir.mkdir(parents=True, exist_ok=True)
    stem = path.stem

    if advice == "upscale_only":
        out = outdir / f"{stem}_upscaled.png"
        ok = do_upscale(path, out) == 0
    elif advice == "redraw_only":
        out = outdir / f"{stem}_redrawn.png"
        ok = do_redraw(path, out) == 0
    else:  # upscale_then_redraw
        out = outdir / f"{stem}_full.png"
        ok = do_redraw(path, out) == 0

    if ok:
        print(f"✅ 完成: {out}  ({out.stat().st_size / 1024:.0f} KB)")
    else:
        print(f"❌ 失败: {path.name}")
    return ok


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="自动选择超分/重绘并执行")
    ap.add_argument("inputs", nargs="+", type=Path, help="图片文件或目录")
    ap.add_argument("--outdir", type=Path, default=None, help="输出目录（默认与原图同目录）")
    ap.add_argument("--force", default=None,
                    choices=["upscale_only", "redraw_only", "upscale_then_redraw", "skip"])
    ap.add_argument("--dry", action="store_true", help="只判断不执行")
    args = ap.parse_args(argv)

    files: list[Path] = []
    for p in args.inputs:
        if p.is_dir():
            files += sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS)
        elif p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
        else:
            print(f"跳过: {p}", file=sys.stderr)

    if not files:
        print("没有可处理的图片", file=sys.stderr)
        return 1

    print(f"待处理 {len(files)} 张")
    ok = fail = 0
    for p in files:
        # 跳过自己产出的中间文件
        if any(tag in p.stem for tag in ("_upscaled", "_redrawn", "_full", "_up", "_denoised", "_对比")):
            continue
        try:
            outdir = args.outdir or p.parent
            if process(p, outdir, args.force, args.dry):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"❌ {p.name}: {e}", file=sys.stderr)
            fail += 1

    print(f"\n{'=' * 60}")
    print(f"完成 {ok} 张，失败 {fail} 张")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
