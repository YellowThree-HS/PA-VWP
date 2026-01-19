#!/bin/bash
# ==========================================
# H100 交互式训练脚本 - ConvNeXt 版本 (RGB)
# ==========================================
# 使用方法：
#   方式1 (后台，默认): bash run_train_convnext_h100.sh
#   方式2 (前台):       bash run_train_convnext_h100.sh --foreground 或 -f
#   方式3 (SLURM):      sbatch run_train_convnext_h100.sh  # 需要添加 SLURM 头
#   方式4 (交互式):     srun -p student --gres=gpu:1 --cpus-per-task=32 --mem=64G --pty bash run_train_convnext_h100.sh --foreground
# ==========================================
# 
# 与 ResNet 版本的区别：
#   - 使用 ConvNeXt 替代 ResNet-50 作为 backbone
#   - ConvNeXt 是"现代化"的卷积网络，借鉴了 Transformer 设计
#   - 通常比 ResNet-50 性能更强
#   - 实验名称为 transunet_convnext_base_h100
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
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${MAGENTA}=========================================="
echo "H100 训练脚本 - TransUNet ConvNeXt (RGB)"
echo -e "==========================================${NC}"
echo ""
echo "ConvNeXt 特点:"
echo "  - 现代化卷积网络设计"
echo "  - 借鉴 Transformer: 大 Kernel size, Layer Norm"
echo "  - 通常比 ResNet-50 性能更强"
echo ""

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
    echo -e "  ${YELLOW}⚠ 非 H100 GPU，使用通用配置${NC}"
    IS_H100=false
fi

# ==========================================
# 检查 CPU 和内存
# ==========================================
echo -e "\n${YELLOW}[2/5] 检查 CPU 和内存...${NC}"
CPU_CORES=$(nproc)
MEM_TOTAL=$(free -g | awk '/^Mem:/{print $2}')
echo "  CPU 核心数: $CPU_CORES"
echo "  可用内存: ${MEM_TOTAL}G"

# ==========================================
# 设置优化参数
# ==========================================
echo -e "\n${YELLOW}[3/5] 设置优化参数...${NC}"

# 进入项目目录
cd /home/hs_25/projs/PA-VWP

# ===== 根据硬件自动调整参数 =====

# Batch Size 计算
# ConvNeXt 比 ResNet-50 占用更多显存 (更深的层，更大的中间特征)
# H100 80GB 需要适当减小 batch size
if [ "$IS_H100" = true ]; then
    BATCH_SIZE=16  # ConvNeXt 需要更小的 batch size
else
    # 提取显存 GB 数（去除单位）
    GPU_MEM_GB=$(echo $GPU_MEM | grep -oP '\d+' | head -1)
    GPU_MEM_GB=$((GPU_MEM_GB / 1024))  # 如果是 MB 转 GB
    if [ "$GPU_MEM_GB" -lt 1 ]; then
        GPU_MEM_GB=$(echo $GPU_MEM | grep -oP '\d+' | head -1)
    fi
    
    if [ "$GPU_MEM_GB" -ge 80 ]; then
        BATCH_SIZE=16
    elif [ "$GPU_MEM_GB" -ge 40 ]; then
        BATCH_SIZE=8
    elif [ "$GPU_MEM_GB" -ge 24 ]; then
        BATCH_SIZE=4
    else
        BATCH_SIZE=2
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
echo -e "\n${YELLOW}[4/5] 设置环境变量...${NC}"

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
# 恢复训练配置 (设置为空则从头训练)
# 注意：模型保存在 /DATA/disk0/hs_25/pa/checkpoints/ 目录下
# 例如: RESUME_PATH="/DATA/disk0/hs_25/pa/checkpoints/transunet_convnext_base_h100/checkpoint_epoch_7.pth"
RESUME_PATH=""

# ==========================================
# 开始训练
# ==========================================
echo -e "\n${MAGENTA}=========================================="
echo "[5/5] 开始训练 TransUNet-ConvNeXt-Base"
echo -e "==========================================${NC}"
echo ""
echo "训练参数:"
echo "  --config base"
echo "  --model base"
echo "  --batch_size $BATCH_SIZE"
echo "  --num_workers $NUM_WORKERS"
echo "  --epochs 150"
echo "  --lr 5e-5"
echo -e "  ${MAGENTA}Backbone: ConvNeXt-Base (现代化卷积网络)${NC}"
echo -e "  ${MAGENTA}输入通道: 4 (RGB + Target Mask)${NC}"
echo -e "  ${MAGENTA}模型保存路径: /DATA/disk0/hs_25/pa/checkpoints/transunet_convnext_base_h100/${NC}"
if [ -n "$RESUME_PATH" ]; then
    echo "  --resume $RESUME_PATH"
fi
echo ""

# 创建日志目录
mkdir -p logs

# 生成日志文件名
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_transunet_convnext_base_h100_${TIMESTAMP}.log"
PID_FILE="logs/train_transunet_convnext_base_h100_${TIMESTAMP}.pid"

echo "日志文件: $LOG_FILE"
echo ""
echo -e "${GREEN}训练开始...${NC}"
echo ""

# 构建训练命令 (使用 train_convnext.py)
TRAIN_CMD="python train/train_convnext.py \
    --config base \
    --model base \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --epochs 150 \
    --lr 5e-5 \
    --exp_name transunet_convnext_base_h100"

# 如果设置了恢复路径，添加 --resume 参数
if [ -n "$RESUME_PATH" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume $RESUME_PATH"
fi

# 运行训练
if [ "$BACKGROUND_MODE" = true ]; then
    # 后台模式：输出只写到文件，不打印到终端
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
    # 前台模式：同时输出到终端和文件
    eval "$TRAIN_CMD" 2>&1 | tee "$LOG_FILE"
fi

echo ""
echo -e "${MAGENTA}=========================================="
echo "训练完成!"
echo -e "==========================================${NC}"
echo "日志已保存到: $LOG_FILE"
