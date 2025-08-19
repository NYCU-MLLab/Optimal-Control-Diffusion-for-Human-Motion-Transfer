#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

if [ $# -lt 1 ]; then
  echo "Usage: $0 <sequence_dir>  # e.g. $0 00001"
  exit 1
fi

### 1. 設定路徑 ###
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_NAME="$1"
# 如果多打一個尾巴底線，嘗試修正
if [ ! -d "$BASE_DIR/$INPUT_NAME" ] && [ -d "$BASE_DIR/${INPUT_NAME%_}" ]; then
  echo "Notice: using '${INPUT_NAME%_}' instead of '$INPUT_NAME'."
  SEQ_NAME="${INPUT_NAME%_}"
else
  SEQ_NAME="$INPUT_NAME"
fi

SEQ_DIR="$BASE_DIR/$SEQ_NAME"
SRC_HANDS="$SEQ_DIR/hands"
SRC_POSES="$SEQ_DIR/poses"
SRC_FACES="$SEQ_DIR/faces"
TEMP_DIR="$SEQ_DIR/temp"
OUTPUT_DIR="$BASE_DIR/$SEQ_NAME"
ANIM_DIR="$OUTPUT_DIR/animated_images"
RESULTS_DIR="$SEQ_DIR/results"
BATCH_SIZE=20

# 檢查必備資料夾
for D in "$SRC_HANDS" "$SRC_POSES" "$SRC_FACES"; do
  [ -d "$D" ] || { echo "Error: missing folder '$D'"; exit 1; }
done

# 準備輸出
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"/{hands,poses,faces}
mkdir -p "$ANIM_DIR" "$RESULTS_DIR"

# 抓取並排序所有 frame 名稱
mapfile -t FRAMES < <(ls "$SRC_HANDS"/frame_*.png 2>/dev/null | xargs -n1 basename | sort -V)
TOTAL=${#FRAMES[@]}
[ "$TOTAL" -gt 0 ] || { echo "Error: no frames in '$SRC_HANDS'"; exit 1; }

BATCHES=$(( (TOTAL + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "→ $TOTAL frames, $BATCHES batches of $BATCH_SIZE."

GLOBAL_START=0
for (( b=0; b<BATCHES; b++ )); do
  START=$(( b * BATCH_SIZE ))
  END=$(( START + BATCH_SIZE - 1 ))
  (( END >= TOTAL )) && END=$(( TOTAL - 1 ))
  echo "=== Batch $((b+1))/$BATCHES: frames $START..$END ==="

  # 2. 清空 temp 並搬 copy 這批的檔案
  rm -f "$TEMP_DIR"/{hands,poses,faces}/*
  for (( i=START; i<=END; i++ )); do
    F="${FRAMES[i]}"
    cp "$SRC_HANDS/$F" "$TEMP_DIR/hands/"
    cp "$SRC_POSES/$F" "$TEMP_DIR/poses/"
    cp "$SRC_FACES/$F" "$TEMP_DIR/faces/"
  done

  # 3. 執行 inference
  bash "$BASE_DIR/command_op_infer.sh"

  # 4. 把 animated_images 裡的 frame_*.png 改名，對應回原始 FRAMES，並移到 results
  for (( i=START; i<=END; i++ )); do
    idx_in_batch=$(( i - START ))
    input_name="${FRAMES[i]}"
    src_png="$ANIM_DIR/frame_${idx_in_batch}.png"
    if [ -f "$src_png" ]; then
      mv "$src_png" "$RESULTS_DIR/$input_name"
    else
      echo "Warning: expected '$src_png' not found—skipping."
    fi
  done

  # 5. 更新 global pointer（只是紀錄總共搬了多少張，用不到就不管它）
  GLOBAL_START=$(( GLOBAL_START + (END - START + 1) ))
done

echo ">>> Done! All $TOTAL frames are in '$RESULTS_DIR'."
