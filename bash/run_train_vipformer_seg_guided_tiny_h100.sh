#!/bin/bash
# ============================================================
# VIPFormer Seg-Guided Tiny (RGB) 训练脚本 (H100)
# Tiny 模型: 4 层 Transformer, 384 hidden dim
# RGB: 4通道输入 (RGB + Target Mask)
# ============================================================
# 用法：
#   bash run_train_vipformer_seg_guided_tiny_h100.sh        # 后台运行
#   bash run_train_vipformer_seg_guided_tiny_h100.sh -f     # 前台运行

BACKGROUND_MODE=true
if [[ "$1" == "--foreground" ]] || [[ "$1" == "-f" ]]; then
    BACKGROUND_MODE=false
fi

set -e

cd /home/hs_25/projs/PA-VWP
source /home/hs_25/miniconda3/etc/profile.d/conda.sh
conda activate isaac

echo "=============================================="
echo "VIPFormer Seg-Guided Tiny (RGB) - H100"
echo "=============================================="
echo "开始时间: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "----------------------------------------------"

# ========== 超参数 ==========
MODEL_VARIANT="tiny"
BATCH_SIZE=32
EPOCHS=150
LR=1e-4
NUM_WORKERS=16
SAVE_DIR="/DATA/disk0/hs_25/pa/checkpoints"
EXP_NAME="vipformer_seg_guided_tiny_h100"
CHECKPOINT_DIR="$SAVE_DIR/$EXP_NAME"

# ========== 恢复训练检测 ==========
RESUME_ARG=""
if [ -d "$CHECKPOINT_DIR" ]; then
    BEST_CKPT=$(ls -t "$CHECKPOINT_DIR"/checkpoint_*best*.pth 2>/dev/null | head -1 || true)
    LATEST_CKPT=$(ls -t "$CHECKPOINT_DIR"/checkpoint_epoch_*.pth 2>/dev/null | head -1 || true)
    if [ -n "$BEST_CKPT" ]; then
        RESUME_ARG="--resume $BEST_CKPT"
        echo "自动检测到最佳 checkpoint: $BEST_CKPT"
    elif [ -n "$LATEST_CKPT" ]; then
        RESUME_ARG="--resume $LATEST_CKPT"
        echo "自动检测到最新 checkpoint: $LATEST_CKPT"
    fi
fi

# ========== 运行训练 ==========
echo "实验名: ${EXP_NAME}"
echo "模型: VIPFormer-SegGuided-${MODEL_VARIANT}"
echo "架构: 4层 Transformer, 384 hidden dim, 8 SEG queries"
echo "输入: RGB 4通道 (RGB + Target Mask)"
echo "Batch Size: ${BATCH_SIZE}"
echo "Learning Rate: ${LR}"
echo "保存目录: ${SAVE_DIR}"
echo "----------------------------------------------"

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_${EXP_NAME}_${TIMESTAMP}.log"
PID_FILE="logs/train_${EXP_NAME}_${TIMESTAMP}.pid"

TRAIN_CMD="python train/train_vipformer_seg_guided.py \
    --config default \
    --model ${MODEL_VARIANT} \
    --exp_name ${EXP_NAME} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --num_workers ${NUM_WORKERS} \
    --save_dir ${SAVE_DIR} \
    ${RESUME_ARG}"

echo "日志: ${LOG_FILE}"

if [ "$BACKGROUND_MODE" = true ]; then
    eval "$TRAIN_CMD" > "$LOG_FILE" 2>&1 &
    TRAIN_PID=$!
    echo "$TRAIN_PID" > "$PID_FILE"
    echo "后台运行中"
    echo "  PID: $TRAIN_PID"
    echo "  查看日志: tail -f $LOG_FILE"
else
    eval "$TRAIN_CMD" 2>&1 | tee "$LOG_FILE"
fi

echo "----------------------------------------------"
echo "结束时间: $(date)"
echo "=============================================="
