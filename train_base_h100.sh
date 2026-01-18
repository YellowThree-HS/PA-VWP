#!/bin/bash
#SBATCH --job-name=transunet_base      # 作业名称
#SBATCH --partition=student             # 使用 student 队列
#SBATCH --gres=gpu:1                    # 申请 1 张 GPU
#SBATCH --cpus-per-task=32              # 申请 32 个 CPU 核心
#SBATCH --mem=64G                       # 申请 64GB 内存
#SBATCH --time=48:00:00                 # 最大运行时间 48 小时
#SBATCH --output=logs/slurm_%j.out      # 标准输出日志
#SBATCH --error=logs/slurm_%j.err       # 错误输出日志

# ==========================================
# TransUNet-Base 训练脚本 (H100 优化版)
# ==========================================

# 创建日志目录
mkdir -p logs

echo "=========================================="
echo "训练环境信息"
echo "=========================================="
echo "作业 ID: $SLURM_JOB_ID"
echo "节点: $SLURM_NODELIST"
echo "GPU 数量: $SLURM_GPUS_ON_NODE"
echo "CPU 核心数: $SLURM_CPUS_PER_TASK"
echo "内存: $SLURM_MEM_PER_NODE"
echo "开始时间: $(date)"
echo ""

# 进入项目目录
cd /home/hs_25/projs/PA-VWP

# ==========================================
# 环境变量优化 (针对 H100)
# ==========================================

# CUDA 优化
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# PyTorch 优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # 减少显存碎片
export TORCH_CUDNN_V8_API_ENABLED=1                       # 启用 cuDNN v8 API

# 数据加载优化
export OMP_NUM_THREADS=8                  # OpenMP 线程数
export MKL_NUM_THREADS=8                  # MKL 线程数

# 禁用调试信息以提升性能
export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_DEBUG=OFF

# 启用 TF32 以加速计算（H100 特有）
export NVIDIA_TF32_OVERRIDE=1

echo "=========================================="
echo "GPU 信息"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ==========================================
# 训练配置 (针对 H100 + base 模型优化)
# ==========================================

# 基础配置
CONFIG="base"
MODEL="base"

# 批次大小优化说明：
# - H100 80GB 显存，TransUNet-base (180M 参数)
# - 输入尺寸 480x640，4 通道
# - 建议 batch_size: 16-32 (单卡)
# - 根据实际显存使用调整
BATCH_SIZE=16

# CPU 核心利用优化：
# - 32 核心，建议 num_workers = 8-12
# - 每个 worker 可能使用 2-3 个线程
# - 留一些核心给主进程和系统
NUM_WORKERS=10

# 学习率（可选调整）
LR=5e-5

# 训练轮数
EPOCHS=150

# ==========================================
# 开始训练
# ==========================================
echo ""
echo "=========================================="
echo "开始训练 TransUNet-Base"
echo "=========================================="
echo "配置: $CONFIG"
echo "模型: $MODEL"
echo "Batch Size: $BATCH_SIZE"
echo "Num Workers: $NUM_WORKERS"
echo "Learning Rate: $LR"
echo "Epochs: $EPOCHS"
echo ""

python train/train.py \
    --config $CONFIG \
    --model $MODEL \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --lr $LR \
    --epochs $EPOCHS \
    --exp_name "transunet_base_h100"

# 记录结束信息
echo ""
echo "=========================================="
echo "训练完成"
echo "=========================================="
echo "结束时间: $(date)"
echo "作业 ID: $SLURM_JOB_ID"
