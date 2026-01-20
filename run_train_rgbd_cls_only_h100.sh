#!/bin/bash
#SBATCH --job-name=transunet_rgbd_cls_only
#SBATCH --output=/DATA/disk0/hs_25/pa/logs/train_transunet_rgbd_cls_only_base_h100_%Y%m%d_%H%M%S.log
#SBATCH --error=/DATA/disk0/hs_25/pa/logs/train_transunet_rgbd_cls_only_base_h100_%Y%m%d_%H%M%S.log
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# ============================================================
# TransUNet ResNet-50 RGBD 仅分类训练脚本 (H100)
# 用于多任务学习消融实验
# ============================================================

# 用法：
#   sbatch run_train_rgbd_cls_only_h100.sh        # 推荐：SLURM 提交（天然后台）
#   bash  run_train_rgbd_cls_only_h100.sh        # 默认：后台运行（本机/交互节点）
#   bash  run_train_rgbd_cls_only_h100.sh -f     # 前台运行

BACKGROUND_MODE=true
if [[ "$1" == "--foreground" ]] || [[ "$1" == "-f" ]]; then
    BACKGROUND_MODE=false
fi

set -e

cd /home/hs_25/projs/PA-VWP
source /home/hs_25/miniconda3/etc/profile.d/conda.sh
conda activate isaac

echo "=============================================="
echo "消融实验: ResNet-50 RGBD 仅分类 (H100)"
echo "=============================================="
echo "开始时间: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "----------------------------------------------"

# ========== 超参数 ==========
MODEL_VARIANT="base"
BATCH_SIZE=32              # RGBD 5通道输入稍微大一点
EPOCHS=150
LR=5e-5
NUM_WORKERS=8
EXP_NAME="transunet_rgbd_cls_only_base_h100_$(date +%Y%m%d_%H%M%S)"
TRAIN_CMD="train/train_rgbd_cls_only.py"

# ========== 运行训练 ==========
echo "实验名: ${EXP_NAME}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Learning Rate: ${LR}"
echo "----------------------------------------------"

mkdir -p logs
LOG_FILE="logs/train_resnet_rgbd_cls_only_h100_${EXP_NAME}.log"
PID_FILE="logs/train_resnet_rgbd_cls_only_h100_${EXP_NAME}.pid"

TRAIN_PY_CMD="python ${TRAIN_CMD} \
    --config base \
    --exp_name ${EXP_NAME} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --model ${MODEL_VARIANT} \
    --num_workers ${NUM_WORKERS}"

echo "日志: ${LOG_FILE}"

if [ "$BACKGROUND_MODE" = true ]; then
    eval "$TRAIN_PY_CMD" > "$LOG_FILE" 2>&1 &
    TRAIN_PID=$!
    echo "$TRAIN_PID" > "$PID_FILE"
    echo "后台运行中"
    echo "  PID: $TRAIN_PID"
    echo "  查看日志: tail -f $LOG_FILE"
else
    eval "$TRAIN_PY_CMD" 2>&1 | tee "$LOG_FILE"
fi

echo "----------------------------------------------"
echo "结束时间: $(date)"
echo "=============================================="
