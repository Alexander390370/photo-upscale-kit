#!/bin/bash
# 下载 ControlNet Tile (SD1.5)，支持断点续传和代理
#
# 用法:
#   bash download_controlnet.sh [目标目录] [代理地址]
#
# 例:
#   bash download_controlnet.sh                              # 默认下到 ~/ComfyUI/models/ControlNet
#   bash download_controlnet.sh ~/models/ControlNet
#   bash download_controlnet.sh ~/models/ControlNet http://127.0.0.1:7890

set +e

DEST="${1:-$HOME/ComfyUI/models/ControlNet}"
PROXY="${2:-${https_proxy:-}}"

mkdir -p "$DEST"
cd "$DEST" || exit 1

FILE=control_v11f1e_sd15_tile.pth
URL_PATH="lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.pth"

if [ -f "$FILE" ] && [ "$(stat -c%s "$FILE")" -gt 1400000000 ]; then
  echo "已存在且完整: $(ls -lh "$FILE" | awk '{print $5}')"
  exit 0
fi

if [ -n "$PROXY" ]; then
  export http_proxy="$PROXY" https_proxy="$PROXY" all_proxy="$PROXY"
  echo "使用代理: $PROXY"
fi

try_download() {
  local host="$1"
  echo "--- 尝试 $host ---"
  curl -L --fail -C - --retry 5 --retry-delay 3 --retry-all-errors \
    --connect-timeout 20 --max-time 3600 \
    -o "$FILE.part" "https://$host/$URL_PATH" \
    -w "\nHTTP:%{http_code} 已下载:%{size_download}\n" 2>&1 | tail -3
  local sz=$(stat -c%s "$FILE.part" 2>/dev/null || echo 0)
  echo "当前大小: $sz"
  [ "$sz" -gt 1400000000 ]
}

if try_download "huggingface.co" || try_download "hf-mirror.com"; then
  mv "$FILE.part" "$FILE"
  echo "✅ 完成: $(ls -lh "$FILE" | awk '{print $5}')"
  exit 0
fi

echo "❌ 下载未完成，保留 $FILE.part 供下次续传"
echo "   如果网络受限，试试先挂代理：bash $0 \"$DEST\" http://127.0.0.1:7890"
exit 1
