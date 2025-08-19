#!/bin/bash

# HandAnimator Optimal Control Single Step - Inference Script
# 使用 train_optimal_control_single_step.py 訓練的模型進行推理

# 設定 GPU
export CUDA_VISIBLE_DEVICES=0

# 基本路徑設定
PRETRAINED_MODEL_PATH="checkpoints/stable-video-diffusion-img2vid-xt"
OUTPUT_DIR="inference/optimal_control_results/00001"

# Optimal Control 訓練的檢查點路徑
OC_CHECKPOINT_DIR="outputs_optimal_control_single_step/ck-45000"
POSENET_PATH="${OC_CHECKPOINT_DIR}/pose_net.pth"
HANDNET_PATH="${OC_CHECKPOINT_DIR}/hand_net.pth"
FACE_ENCODER_PATH="${OC_CHECKPOINT_DIR}/face_encoder.pth"
UNET_PATH="${OC_CHECKPOINT_DIR}/unet.pth"

# 推理輸入路徑
VALIDATION_HAND_IMAGES="inference/00001/hands"
VALIDATION_CONTROL_FOLDER="inference/00001/poses"
VALIDATION_IMAGE="inference/00001/reference.png"

# 推理參數
WIDTH=576
HEIGHT=1024
GUIDANCE_SCALE=3.0
NUM_INFERENCE_STEPS=1
TILE_SIZE=16
OVERLAP=4
NOISE_AUG_STRENGTH=0.02
FRAMES_OVERLAP=4
DECODE_CHUNK_SIZE=4

echo "🚀 Starting HandAnimator Optimal Control Inference..."
echo "📁 Using Optimal Control trained models from: $OC_CHECKPOINT_DIR"
echo "📊 Inference parameters:"
echo "   - Guidance scale: $GUIDANCE_SCALE"
echo "   - Inference steps: $NUM_INFERENCE_STEPS"
echo "   - Resolution: ${WIDTH}x${HEIGHT}"
echo "   - Output directory: $OUTPUT_DIR"

# 檢查必要文件
echo "🔍 Checking required files..."

if [ ! -d "$PRETRAINED_MODEL_PATH" ]; then
    echo "❌ Error: Pretrained model not found at $PRETRAINED_MODEL_PATH"
    exit 1
fi

if [ ! -d "$OC_CHECKPOINT_DIR" ]; then
    echo "❌ Error: Optimal Control checkpoint directory not found at $OC_CHECKPOINT_DIR"
    echo "💡 Please run train_optimal_control_single_step.py first to generate the checkpoints"
    exit 1
fi

# 檢查各個模型檢查點
for model_path in "$POSENET_PATH" "$HANDNET_PATH" "$FACE_ENCODER_PATH" "$UNET_PATH"; do
    if [ ! -f "$model_path" ]; then
        echo "❌ Error: Model checkpoint not found at $model_path"
        exit 1
    fi
done

# 檢查推理輸入
if [ ! -f "$VALIDATION_IMAGE" ]; then
    echo "❌ Error: Reference image not found at $VALIDATION_IMAGE"
    exit 1
fi

if [ ! -d "$VALIDATION_HAND_IMAGES" ]; then
    echo "❌ Error: Hand images directory not found at $VALIDATION_HAND_IMAGES"
    exit 1
fi

if [ ! -d "$VALIDATION_CONTROL_FOLDER" ]; then
    echo "❌ Error: Pose control directory not found at $VALIDATION_CONTROL_FOLDER"
    exit 1
fi

# 創建輸出目錄
mkdir -p "$OUTPUT_DIR"

echo "✅ All files verified. Starting inference..."

# 執行推理（使用原始的 inference_basic.py）
CUDA_VISIBLE_DEVICES=0 python inference_basic.py \
    --pretrained_model_name_or_path="$PRETRAINED_MODEL_PATH" \
    --output_dir="$OUTPUT_DIR" \
    --validation_hand_images="$VALIDATION_HAND_IMAGES" \
    --validation_control_folder="$VALIDATION_CONTROL_FOLDER" \
    --validation_image="$VALIDATION_IMAGE" \
    --width=$WIDTH \
    --height=$HEIGHT \
    --guidance_scale=$GUIDANCE_SCALE \
    --num_inference_steps=$NUM_INFERENCE_STEPS \
    --posenet_model_name_or_path="$POSENET_PATH" \
    --handnet_model_name_or_path="$HANDNET_PATH" \
    --face_encoder_model_name_or_path="$FACE_ENCODER_PATH" \
    --unet_model_name_or_path="$UNET_PATH" \
    --tile_size=$TILE_SIZE \
    --overlap=$OVERLAP \
    --noise_aug_strength=$NOISE_AUG_STRENGTH \
    --frames_overlap=$FRAMES_OVERLAP \
    --decode_chunk_size=$DECODE_CHUNK_SIZE \
    --gradient_checkpointing

# 檢查推理結果
if [ $? -eq 0 ]; then
    echo "✅ Optimal Control inference completed successfully!"
    echo "📁 Results saved to: $OUTPUT_DIR"
    echo "🎯 This inference used models trained with Optimal Control (drift minimization + state regularization)"
    echo "📊 Expected improvements:"
    echo "   - Better control precision (reduced drift)"
    echo "   - More stable hand movements"
    echo "   - Improved temporal consistency"
else
    echo "❌ Inference failed with exit code: $?"
    echo "🔍 Please check the error messages above"
    echo "💡 Common issues:"
    echo "   - Missing checkpoint files"
    echo "   - Incorrect input paths"
    echo "   - GPU memory issues"
fi