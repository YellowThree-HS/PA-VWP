#!/bin/bash
# ============================================================
# 在test数据集上测试三个最佳模型
# ============================================================
# 模型:
# 1. transunet_fusion_tiny_h100
# 2. transunet_seg_from_encoded_tiny_h100
# 3. transunet_cls_only_tiny_h100

set -e

cd /home/hs_25/projs/PA-VWP
source /home/hs_25/miniconda3/etc/profile.d/conda.sh
conda activate isaac

echo "=============================================="
echo "测试三个最佳模型"
echo "=============================================="
echo "开始时间: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "----------------------------------------------"

# 测试数据集路径
TEST_DIR="/DATA/disk0/hs_25/pa/all_dataset/test"
BATCH_SIZE=32
DEVICE="cuda"

# Checkpoint路径
CHECKPOINT_DIR="/DATA/disk0/hs_25/pa/checkpoints"

# 模型配置
declare -A MODELS=(
    ["fusion"]="/DATA/disk0/hs_25/pa/checkpoints/transunet_fusion_tiny_h100/checkpoint_epoch_16_best.pth"
    ["seg_from_encoded"]="/DATA/disk0/hs_25/pa/checkpoints/transunet_seg_from_encoded_tiny_h100/checkpoint_epoch_21_best.pth"
    ["cls_only"]="/DATA/disk0/hs_25/pa/checkpoints/transunet_cls_only_tiny_h100/checkpoint_epoch_10_best.pth"
)

# 输出目录
OUTPUT_BASE_DIR="test_results"

# 测试每个模型
for model_name in "${!MODELS[@]}"; do
    checkpoint="${MODELS[$model_name]}"
    
    echo ""
    echo "=============================================="
    echo "测试模型: $model_name"
    echo "Checkpoint: $checkpoint"
    echo "=============================================="
    
    # 检查checkpoint是否存在
    if [ ! -f "$checkpoint" ]; then
        echo "错误: Checkpoint不存在: $checkpoint"
        continue
    fi
    
    # 运行测试
    python test_models.py \
        --checkpoint "$checkpoint" \
        --test_dir "$TEST_DIR" \
        --batch_size "$BATCH_SIZE" \
        --device "$DEVICE" \
        --output_dir "$OUTPUT_BASE_DIR/${model_name}_tiny_h100"
    
    echo "✓ 完成: $model_name"
done

echo ""
echo "=============================================="
echo "所有模型测试完成"
echo "结束时间: $(date)"
echo "=============================================="

# 汇总结果
echo ""
echo "=============================================="
echo "结果汇总"
echo "=============================================="
for model_name in "${!MODELS[@]}"; do
    metrics_file="$OUTPUT_BASE_DIR/${model_name}_tiny_h100/metrics.json"
    if [ -f "$metrics_file" ]; then
        echo ""
        echo "模型: $model_name"
        echo "指标文件: $metrics_file"
        cat "$metrics_file" | python -m json.tool
    fi
done
