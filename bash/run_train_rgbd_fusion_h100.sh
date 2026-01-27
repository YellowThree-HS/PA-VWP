#!/bin/bash
# ==========================================
# H100 训练脚本 - TransUNet RGBD Fusion
# 分割感知双路融合模型 (带深度图)
# ==========================================
# 使用方法：
#   方式1 (后台，默认): bash run_train_rgbd_fusion_h100.sh
#   方式2 (前台):       bash run_train_rgbd_fusion_h100.sh --foreground 或 -f
#   方式3 (SLURM):      sbatch run_train_rgbd_fusion_h100.sh
#   方式4 (交互式):     srun -p student --gres=gpu:1 --cpus-per-task=32 --mem=64G --pty bash run_train_rgbd_fusion_h100.sh --foreground
# ==========================================

# 检查是否后台模式（默认后台运行）
BACKGROUND_MODE=true
if [[ "$1" == "--foreground" ]] || [[ "$1" == "-f" ]]; then
    BACKGROUND_MODE=false
fi

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}=========================================="
echo "H100 训练脚本 - TransUNet RGBD Fusion"
echo "分割感知双路融合模型 (带深度图)"
echo -e "==========================================${NC}"

# ==========================================
# 检查环境
# ==========================================
echo -e "\n${YELLOW}[1/5] 检查 GPU 环境...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}错误: 未检测到 GPU，请确保在 GPU 节点上运行${NC}"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
echo "  GPU: $GPU_NAME"
echo "  显存: $GPU_MEM"

# 检查是否是 H100
if [[ "$GPU_NAME" == *"H100"* ]]; then
    echo -e "  ${GREEN}✓ 检测到 H100，启用优化配置${NC}"
    IS_H100=true
else
    echo -e "  ${RED}✗ 未检测到 H100 GPU，当前 GPU: $GPU_NAME${NC}"
    echo -e "  ${RED}  此脚本仅支持在 H100 节点运行，请使用 srun 或 sbatch 提交到 GPU 节点${NC}"
    echo -e "  ${YELLOW}  示例: srun -p student --gres=gpu:1 --cpus-per-task=32 --mem=64G --pty bash run_train_rgbd_fusion_h100.sh -f${NC}"
    exit 1
fi

# ==========================================
# 激活 Conda 环境
# ==========================================
echo -e "\n${YELLOW}[2/6] 激活 Conda 环境...${NC}"
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
else
    echo -e "${RED}错误: 未找到 conda，请确保已安装 conda${NC}"
    exit 1
fi

conda activate isaac
if [ $? -ne 0 ]; then
    echo -e "${RED}错误: 无法激活 conda 环境 isaac${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓ 已激活 conda 环境: isaac${NC}"

# ==========================================
# 检查 CPU 和内存
# ==========================================
echo -e "\n${YELLOW}[3/6] 检查 CPU 和内存...${NC}"
CPU_CORES=$(nproc)
MEM_TOTAL=$(free -g | awk '/^Mem:/{print $2}')
echo "  CPU 核心数: $CPU_CORES"
echo "  可用内存: ${MEM_TOTAL}G"

# ==========================================
# 设置优化参数
# ==========================================
echo -e "\n${YELLOW}[4/6] 设置优化参数...${NC}"

# 进入项目目录
cd /home/hs_25/projs/PA-VWP

# ===== 根据硬件自动调整参数 =====

# Batch Size 计算
# RGBD Fusion 模型输入通道多，batch size 稍减
if [ "$IS_H100" = true ]; then
    BATCH_SIZE=24  # RGBD + Fusion，减少到 24
else
    GPU_MEM_GB=$(echo $GPU_MEM | grep -oP '\d+' | head -1)
    GPU_MEM_GB=$((GPU_MEM_GB / 1024))
    if [ "$GPU_MEM_GB" -lt 1 ]; then
        GPU_MEM_GB=$(echo $GPU_MEM | grep -oP '\d+' | head -1)
    fi
    
    if [ "$GPU_MEM_GB" -ge 80 ]; then
        BATCH_SIZE=12
    elif [ "$GPU_MEM_GB" -ge 40 ]; then
        BATCH_SIZE=5
    elif [ "$GPU_MEM_GB" -ge 24 ]; then
        BATCH_SIZE=2
    else
        BATCH_SIZE=1
    fi
fi

# Num Workers 计算
if [ "$CPU_CORES" -ge 32 ]; then
    NUM_WORKERS=16
elif [ "$CPU_CORES" -ge 16 ]; then
    NUM_WORKERS=6
elif [ "$CPU_CORES" -ge 8 ]; then
    NUM_WORKERS=4
else
    NUM_WORKERS=2
fi

echo "  Batch Size: $BATCH_SIZE"
echo "  Num Workers: $NUM_WORKERS"

# ==========================================
# 环境变量优化
# ==========================================
echo -e "\n${YELLOW}[5/6] 设置环境变量...${NC}"

# CUDA 优化
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# PyTorch 优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 多线程优化
export OMP_NUM_THREADS=$((CPU_CORES / 4))
export MKL_NUM_THREADS=$((CPU_CORES / 4))

# H100 特有优化
if [ "$IS_H100" = true ]; then
    export NVIDIA_TF32_OVERRIDE=1    # 启用 TF32
    export TORCH_CUDNN_V8_API_ENABLED=1
    echo "  ✓ 已启用 H100 特有优化 (TF32, cuDNN v8)"
fi

echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS: $MKL_NUM_THREADS"

# ==========================================
# 恢复训练配置
# ==========================================
SAVE_DIR="/DATA/disk0/hs_25/pa/checkpoints"
CHECKPOINT_DIR="$SAVE_DIR/transunet_rgbd_fusion_base_h100"

if [ -n "${RESUME_PATH:-}" ]; then
    echo "  使用环境变量指定的恢复路径: $RESUME_PATH"
else
    RESUME_PATH=""
    if [ -d "$CHECKPOINT_DIR" ]; then
        BEST_CKPT=$(ls -t "$CHECKPOINT_DIR"/checkpoint_*best*.pth 2>/dev/null | head -1 || true)
        LATEST_CKPT=$(ls -t "$CHECKPOINT_DIR"/checkpoint_epoch_*.pth 2>/dev/null | head -1 || true)
        if [ -n "$BEST_CKPT" ]; then
            RESUME_PATH="$BEST_CKPT"
            echo "  自动检测到最佳 checkpoint: $RESUME_PATH"
        elif [ -n "$LATEST_CKPT" ]; then
            RESUME_PATH="$LATEST_CKPT"
            echo "  自动检测到最新 checkpoint: $RESUME_PATH"
        else
            echo "  未在 $CHECKPOINT_DIR 中找到 checkpoint，将从头训练"
        fi
    else
        echo "  checkpoint 目录不存在: $CHECKPOINT_DIR，将从头训练"
    fi
fi

# ==========================================
# 开始训练
# ==========================================
echo -e "\n${CYAN}=========================================="
echo "[6/6] 开始训练 TransUNet-RGBD-Fusion-Base"
echo -e "==========================================${NC}"
echo ""
echo "训练参数:"
echo "  --config base"
echo "  --model base"
echo "  --batch_size $BATCH_SIZE"
echo "  --num_workers $NUM_WORKERS"
echo "  --epochs 150"
echo "  --lr 5e-5"
echo "  输入通道: 5 (RGB + Depth + Target Mask)"
echo "  融合策略: Mask-Guided Attention + Multi-Scale Decoder Features"
if [ -n "$RESUME_PATH" ]; then
    echo "  --resume $RESUME_PATH"
fi
echo ""

# 创建日志目录
mkdir -p logs

# 生成日志文件名
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_transunet_rgbd_fusion_base_h100_${TIMESTAMP}.log"
PID_FILE="logs/train_transunet_rgbd_fusion_base_h100_${TIMESTAMP}.pid"

echo "日志文件: $LOG_FILE"
echo ""
echo -e "${GREEN}训练开始...${NC}"
echo ""

# 构建训练命令
TRAIN_CMD="python train/train_rgbd_fusion.py \
    --config base \
    --model base \
    --save_dir $SAVE_DIR \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --epochs 150 \
    --lr 5e-5 \
    --exp_name transunet_rgbd_fusion_base_h100"

# 如果设置了恢复路径，添加 --resume 参数
if [ -n "$RESUME_PATH" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume $RESUME_PATH"
fi

# 运行训练
if [ "$BACKGROUND_MODE" = true ]; then
    echo "  → 后台模式：输出重定向到 $LOG_FILE"
    eval "$TRAIN_CMD" > "$LOG_FILE" 2>&1 &
    
    TRAIN_PID=$!
    echo $TRAIN_PID > "$PID_FILE"
    echo ""
    echo -e "${GREEN}训练已在后台启动！${NC}"
    echo "  进程 PID: $TRAIN_PID"
    echo "  日志文件: $LOG_FILE"
    echo "  PID 文件: $PID_FILE"
    echo ""
    echo "查看日志: tail -f $LOG_FILE"
    echo "查看进程: ps -p $TRAIN_PID"
else
    eval "$TRAIN_CMD" 2>&1 | tee "$LOG_FILE"
fi

echo ""
echo -e "${CYAN}=========================================="
echo "训练完成!"
echo -e "==========================================${NC}"
echo "日志已保存到: $LOG_FILE"
