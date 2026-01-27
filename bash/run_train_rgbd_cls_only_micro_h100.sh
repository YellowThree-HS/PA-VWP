#!/bin/bash
# ============================================================
# TransUNet RGBD Cls-Only Micro 训练脚本 (H100)
# Micro 模型: 2 层 Transformer, 192 hidden dim (超轻量级)
# RGBD: RGB + Depth 融合 (5通道输入)
# Cls-Only: 仅分类头，无分割头
# ============================================================
# 用法：
#   bash run_train_rgbd_cls_only_micro_h100.sh        # 后台运行
#   bash run_train_rgbd_cls_only_micro_h100.sh -f     # 前台运行

BACKGROUND_MODE=true
if [[ "$1" == "--foreground" ]] || [[ "$1" == "-f" ]]; then
    BACKGROUND_MODE=false
fi

set -e

cd /home/hs_25/projs/PA-VWP
source /home/hs_25/miniconda3/etc/profile.d/conda.sh
conda activate isaac

echo "=============================================="
echo "TransUNet RGBD Cls-Only Micro - H100"
echo "=============================================="
echo "开始时间: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "----------------------------------------------"

# ========== 超参数 ==========
MODEL_VARIANT="micro"
BATCH_SIZE=64              # micro + cls_only，显存占用最小，可以用更大 batch
EPOCHS=150
LR=1e-4
NUM_WORKERS=16
# 注意：train_rgbd_cls_only.py 内部默认保存到 /DATA/disk0/hs_25/pa/checkpoints
EXP_NAME="transunet_rgbd_cls_only_micro_h100"

# ========== 运行训练 ==========
echo "实验名: ${EXP_NAME}"
echo "模型: ${MODEL_VARIANT} (2层 Transformer, 192 hidden dim)"
echo "输入: RGBD 5通道融合"
echo "特点: 仅分类头 (无分割头)"
echo "Batch Size: ${BATCH_SIZE}"
echo "Learning Rate: ${LR}"
echo "----------------------------------------------"

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_${EXP_NAME}_${TIMESTAMP}.log"
PID_FILE="logs/train_${EXP_NAME}_${TIMESTAMP}.pid"

TRAIN_CMD="python train/train_rgbd_cls_only.py \
    --config base \
    --model ${MODEL_VARIANT} \
    --exp_name ${EXP_NAME} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --num_workers ${NUM_WORKERS}"

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
