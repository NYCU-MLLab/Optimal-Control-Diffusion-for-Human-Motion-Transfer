#!/bin/bash
set -e

# 取得腳本所在的絕對路徑，並切換到該目錄（專案根目錄）
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

# 定義目錄與檔案路徑
VIDEOS_DIR="$SCRIPT_DIR/animation_data/videos/validation"
TARGET="validation"

REC_DIR="$SCRIPT_DIR/animation_data/$TARGET"
TRAINING_SCRIPT="$SCRIPT_DIR/DWPose/training_skeleton_extraction.py"
HAND_MASK_SCRIPT="$SCRIPT_DIR/hand_mask_extraction.py"
FACE_MASK_SCRIPT="$SCRIPT_DIR/face_mask_extraction.py"
HAMER_SCRIPT="$SCRIPT_DIR/hamer_process.py"

TOTAL=333

# 定義進度條函式
progress_bar() {
    local progress=$1
    local total=$2
    local percent=$(( progress * 100 / total ))
    local bar_len=50
    local filled=$(( progress * bar_len / total ))
    local empty=$(( bar_len - filled ))
    printf "\r["
    for (( i=0; i<filled; i++ )); do printf "#"; done
    for (( i=0; i<empty; i++ )); do printf "-"; done
    printf "] %d%%" "$percent"
}

# Step 1: 從 animation_data/videos 中讀取影片並切成影格
counter=1
mkdir -p "$REC_DIR"

echo "Running ffmpeg to extract frames from videos..."
for video in $(ls "$VIDEOS_DIR"/*.mp4 | sort); do
    new_num=$(printf "%05d" "$counter")
    mkdir -p "$REC_DIR/$new_num/images"
    # 使用 ffmpeg 將影片切成影格，隱藏其他輸出訊息
    ffmpeg -loglevel quiet -i "$video" -q:v 1 -start_number 0 "$REC_DIR/$new_num/images/frame_%d.png"
    progress_bar "$counter" "$TOTAL"
    counter=$((counter+1))
done
echo "\nStep 1 complete."

# Step 2: 執行骨架姿勢提取（隱藏所有輸出）
echo "Running training_skeleton_extraction.py..."
#####
python "$TRAINING_SCRIPT" --root_path="animation_data" --name="$TARGET" --start=1 --end=333 > /dev/null 2>&1
#####
echo "Step 2 complete."

# Step 3: 針對 animation_data/rec 下每個資料夾內的 images 執行人臉遮罩提取，並更新進度條
echo "Running face mask extraction for all folders..."
folder_counter=0
for folder in "$REC_DIR"/*; do
    if [ -d "$folder" ]; then
        python "$FACE_MASK_SCRIPT" --image_folder="$folder/images" > /dev/null 2>&1
        folder_counter=$((folder_counter+1))
        progress_bar "$folder_counter" "$TOTAL"
    fi
done
echo "\nStep 3 complete."



# Step 4: 針對 animation_data/rec_new 下每個資料夾，執行 hamer_process.py
# 將每個資料夾內的 images 處理成 hands，輸出路徑為 <folder>/hands
echo "Running hamer_process.py for all folders..."
python "$HAMER_SCRIPT" --root_path "animation_data/$TARGET" > /dev/null 2>&1
echo "Step 4 complete."


# Step 5: 對 animation_data/rec_new 下每個資料夾執行 hand mask extraction
echo "Running hand mask extraction for all folders..."
python "$HAND_MASK_SCRIPT" "animation_data/$TARGET" > /dev/null 2>&1
echo "Step 5 complete."


echo "All processing complete."
