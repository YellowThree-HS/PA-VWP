#!/bin/bash
# ============================================================
# 测试 seg_from_encoded 模型并保存预测失败的样本
# ============================================================

set -e

cd /home/hs_25/projs/PA-VWP
source /home/hs_25/miniconda3/etc/profile.d/conda.sh
conda activate isaac

echo "=============================================="
echo "测试 seg_from_encoded 模型并保存失败样本"
echo "=============================================="
echo "开始时间: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "----------------------------------------------"

# 测试数据集路径
TEST_DIR="/DATA/disk0/hs_25/pa/all_dataset/test"
BATCH_SIZE=32
DEVICE="cuda"
CHECKPOINT="/DATA/disk0/hs_25/pa/checkpoints/transunet_seg_from_encoded_tiny_h100/checkpoint_epoch_21_best.pth"
OUTPUT_DIR="test_results/seg_from_encoded_tiny_h100_failures"

# 运行测试
python test_models.py \
    --checkpoint "$CHECKPOINT" \
    --test_dir "$TEST_DIR" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR" \
    --save_failures \
    --num_failures 10

echo ""
echo "=============================================="
echo "测试完成"
echo "失败样本保存在: $OUTPUT_DIR/failures/"
echo "失败信息保存在: $OUTPUT_DIR/failures_info.json"
echo "结束时间: $(date)"
echo "=============================================="
