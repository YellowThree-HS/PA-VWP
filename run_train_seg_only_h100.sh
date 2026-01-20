#!/bin/bash
#SBATCH --job-name=transunet_seg_only
#SBATCH --output=/DATA/disk0/hs_25/pa/logs/train_transunet_seg_only_base_h100_%Y%m%d_%H%M%S.log
#SBATCH --error=/DATA/disk0/hs_25/pa/logs/train_transunet_seg_only_base_h100_%Y%m%d_%H%M%S.log
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# ============================================================
# TransUNet ResNet-50 仅分割训练脚本 (H100)
# 用于多任务学习消融实验
# ============================================================

# 用法：
#   sbatch run_train_seg_only_h100.sh                    # 推荐：SLURM 提交（天然后台）
#   bash  run_train_seg_only_h100.sh                     # 默认：后台运行（本机/交互节点）
#   bash  run_train_seg_only_h100.sh -f                 # 前台运行
#   bash  run_train_seg_only_h100.sh --resume EXP_NAME   # 从最佳检查点恢复训练

BACKGROUND_MODE=true
RESUME_EXP_NAME=""
RESUME_CHECKPOINT=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --foreground|-f)
            BACKGROUND_MODE=false
            shift
            ;;
        --resume)
            RESUME_EXP_NAME="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

set -e

cd /home/hs_25/projs/PA-VWP
source /home/hs_25/miniconda3/etc/profile.d/conda.sh
conda activate isaac

echo "=============================================="
echo "消融实验: ResNet-50 仅分割 (H100)"
echo "=============================================="
echo "开始时间: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "----------------------------------------------"

# ========== 超参数 ==========
MODEL_VARIANT="base"
BATCH_SIZE=36
EPOCHS=150
LR=5e-5
NUM_WORKERS=8
CHECKPOINTS_DIR="/DATA/disk0/hs_25/pa/checkpoints"
TRAIN_CMD="train/train_seg_only.py"

# ========== 检查点恢复逻辑 ==========
if [ -n "$RESUME_EXP_NAME" ]; then
    EXP_NAME="$RESUME_EXP_NAME"
    EXP_CHECKPOINT_DIR="${CHECKPOINTS_DIR}/${EXP_NAME}"
    
    if [ ! -d "$EXP_CHECKPOINT_DIR" ]; then
        echo "错误: 检查点目录不存在: $EXP_CHECKPOINT_DIR"
        exit 1
    fi
    
    # 查找最佳检查点（最新的 *_best.pth 文件）
    BEST_CHECKPOINT=$(ls -t "${EXP_CHECKPOINT_DIR}"/checkpoint_epoch_*_best.pth 2>/dev/null | head -1)
    
    if [ -z "$BEST_CHECKPOINT" ]; then
        echo "错误: 未找到最佳检查点文件 (*_best.pth) 在目录: $EXP_CHECKPOINT_DIR"
        exit 1
    fi
    
    RESUME_CHECKPOINT="$BEST_CHECKPOINT"
    echo "从检查点恢复训练:"
    echo "  实验名: ${EXP_NAME}"
    echo "  检查点: ${RESUME_CHECKPOINT}"
    echo "----------------------------------------------"
else
    EXP_NAME="transunet_seg_only_base_h100_$(date +%Y%m%d_%H%M%S)"
fi

# ========== 运行训练 ==========
echo "实验名: ${EXP_NAME}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Learning Rate: ${LR}"
if [ -n "$RESUME_CHECKPOINT" ]; then
    echo "恢复检查点: ${RESUME_CHECKPOINT}"
fi
echo "----------------------------------------------"

mkdir -p logs
LOG_FILE="logs/train_resnet_seg_only_h100_${EXP_NAME}.log"
PID_FILE="logs/train_resnet_seg_only_h100_${EXP_NAME}.pid"

TRAIN_PY_CMD="python ${TRAIN_CMD} \
    --config base \
    --exp_name ${EXP_NAME} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --model ${MODEL_VARIANT} \
    --num_workers ${NUM_WORKERS}"

if [ -n "$RESUME_CHECKPOINT" ]; then
    TRAIN_PY_CMD="${TRAIN_PY_CMD} --resume ${RESUME_CHECKPOINT}"
fi

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
