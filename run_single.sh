#!/bin/bash
# 单张跑：超分 + ControlNet 重绘，参数写在命令行上
#
# 用法:
#   bash run_single.sh 输入.jpg 输出名 [denoise] [scale]
#
# 例:
#   bash run_single.sh ~/图片/a.jpg result              # 默认 denoise=0.35, scale=4
#   bash run_single.sh ~/图片/a.jpg result 0.25 2       # 更保守，只放大 2 倍

set -e

PY="${PY:-python3}"
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ $# -lt 2 ]; then
  echo "用法: bash run_single.sh 输入.jpg 输出名 [denoise=0.35] [scale=4]" >&2
  exit 2
fi

INPUT="$1"
NAME="$2"
DENOISE="${3:-0.35}"
SCALE="${4:-4}"
OUTDIR="$(dirname "$INPUT")"

echo "===== $(basename "$INPUT")  denoise=$DENOISE  scale=$SCALE ====="
"$PY" "$HERE/redraw.py" "$INPUT" "$OUTDIR/${NAME}.png" \
  --model "$(python3 -c "import json,sys;print(json.load(open('$HERE/models.json'))['sd'])" 2>/dev/null || echo "$SD_MODEL")" \
  --controlnet "$(python3 -c "import json,sys;print(json.load(open('$HERE/models.json'))['controlnet'])" 2>/dev/null || echo "$CONTROLNET")" \
  --cn-scale 1.0 \
  --scale "$SCALE" --denoise "$DENOISE" --pre-denoise 1.2 \
  --tile 512 --overlap 128 --steps 28 --guidance 4.5 \
  --prompt "highly detailed photograph, sharp focus, natural skin texture, fine pores, soft studio lighting, professional photography" \
  --negative "blurry, lowres, jpeg artifacts, noise, grainy, watermark, text, cartoon, anime, deformed, oversaturated, plastic skin"
