#!/bin/bash
# ==========================================
# H100 训练脚本 - ConvNeXt RGBD 仅分类 (消融实验)
# ==========================================
# 用法：
#   bash run_train_convnext_rgbd_cls_only_h100.sh
#   bash run_train_convnext_rgbd_cls_only_h100.sh -f  # 前台运行
# ==========================================

BACKGROUND_MODE=true
if [[ "$1" == "--foreground" ]] || [[ "$1" == "-f" ]]; then
    BACKGROUND_MODE=false
fi

set -e

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}=========================================="
echo "消融实验: ConvNeXt RGBD 仅分类"
echo -e "==========================================${NC}"
echo ""
echo "实验目的: 验证分割任务对分类的辅助作用 (RGBD)"
echo "输入: 5 通道 (RGB + Depth + Mask)"
echo ""

# 检查 GPU
echo -e "${YELLOW}[1/4] 检查 GPU...${NC}"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "  GPU: $GPU_NAME"

IS_H100=false
if [[ "$GPU_NAME" == *"H100"* ]]; then
    IS_H100=true
    echo -e "  ${GREEN}✓ H100 检测到${NC}"
fi

# 设置参数
echo -e "\n${YELLOW}[2/4] 设置参数...${NC}"
cd /home/hs_25/projs/PA-VWP

if [ "$IS_H100" = true ]; then
    BATCH_SIZE=18  # RGBD 多一个通道，略减小
else
    BATCH_SIZE=6
fi

CPU_CORES=$(nproc)
if [ "$CPU_CORES" -ge 32 ]; then
    NUM_WORKERS=16
else
    NUM_WORKERS=8
fi

echo "  Batch Size: $BATCH_SIZE"
echo "  Num Workers: $NUM_WORKERS"

# 环境变量
echo -e "\n${YELLOW}[3/4] 设置环境变量...${NC}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$((CPU_CORES / 4))
export MKL_NUM_THREADS=$((CPU_CORES / 4))

if [ "$IS_H100" = true ]; then
    export NVIDIA_TF32_OVERRIDE=1
    export TORCH_CUDNN_V8_API_ENABLED=1
fi

# 训练
echo -e "\n${CYAN}=========================================="
echo "[4/4] 开始训练"
echo -e "==========================================${NC}"
echo ""
echo "训练参数:"
echo "  --config base"
echo "  --batch_size $BATCH_SIZE"
echo "  --epochs 150"
echo -e "  ${CYAN}任务: 仅分类 (无分割头, RGBD 输入)${NC}"
echo ""

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_convnext_rgbd_cls_only_h100_${TIMESTAMP}.log"
PID_FILE="logs/train_convnext_rgbd_cls_only_h100_${TIMESTAMP}.pid"

TRAIN_CMD="python train/train_convnext_rgbd_cls_only.py \
    --config base \
    --model base \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --epochs 150 \
    --lr 5e-5 \
    --exp_name transunet_convnext_rgbd_cls_only_base_h100"

echo "日志: $LOG_FILE"
echo -e "${GREEN}训练开始...${NC}"

if [ "$BACKGROUND_MODE" = true ]; then
    eval "$TRAIN_CMD" > "$LOG_FILE" 2>&1 &
    TRAIN_PID=$!
    echo $TRAIN_PID > "$PID_FILE"
    echo ""
    echo -e "${GREEN}后台运行中${NC}"
    echo "  PID: $TRAIN_PID"
    echo "  查看日志: tail -f $LOG_FILE"
else
    eval "$TRAIN_CMD" 2>&1 | tee "$LOG_FILE"
fi

echo ""
echo -e "${CYAN}完成!${NC}"
