#!/bin/bash
# ============================================================
# 测试micro模型的最佳checkpoint
# ============================================================

set -e

cd /home/hs_25/projs/PA-VWP
source /home/hs_25/miniconda3/etc/profile.d/conda.sh
conda activate isaac

echo "=============================================="
echo "测试Micro模型"
echo "=============================================="
echo "开始时间: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "----------------------------------------------"

# 测试数据集路径
TEST_DIR="/DATA/disk0/hs_25/pa/all_dataset/test"
BATCH_SIZE=32
DEVICE="cuda"

# Micro模型checkpoint路径
CLS_ONLY_CHECKPOINT="/DATA/disk0/hs_25/pa/checkpoints/transunet_cls_only_micro_h100/checkpoint_epoch_5_best.pth"
SEG_FROM_ENCODED_CHECKPOINT="/DATA/disk0/hs_25/pa/checkpoints/transunet_seg_from_encoded_micro_h100/checkpoint_epoch_9_best.pth"

echo ""
echo "=============================================="
echo "1. 测试 Cls-Only Micro 模型"
echo "=============================================="
echo "Checkpoint: $CLS_ONLY_CHECKPOINT"
echo "最佳F1: 0.8610 (Epoch 5)"
echo "----------------------------------------------"

python test_models.py \
    --checkpoint "$CLS_ONLY_CHECKPOINT" \
    --test_dir "$TEST_DIR" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --output_dir test_results/cls_only_micro_h100

echo ""
echo "✓ Cls-Only Micro 测试完成"
echo ""

echo "=============================================="
echo "2. 测试 Seg-From-Encoded Micro 模型"
echo "=============================================="
echo "Checkpoint: $SEG_FROM_ENCODED_CHECKPOINT"
echo "最佳F1: 0.8627 (Epoch 9)"
echo "----------------------------------------------"

python test_models.py \
    --checkpoint "$SEG_FROM_ENCODED_CHECKPOINT" \
    --test_dir "$TEST_DIR" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --output_dir test_results/seg_from_encoded_micro_h100

echo ""
echo "✓ Seg-From-Encoded Micro 测试完成"
echo ""

echo "=============================================="
echo "测试完成"
echo "结束时间: $(date)"
echo "=============================================="
echo ""
echo "结果保存在:"
echo "  - test_results/cls_only_micro_h100/"
echo "  - test_results/seg_from_encoded_micro_h100/"
