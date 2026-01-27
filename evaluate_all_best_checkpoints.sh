#!/bin/bash
# 批量评估RGBD tiny segfromencoded模型的所有best checkpoint

CHECKPOINT_DIR="/DATA/disk0/hs_25/pa/checkpoints/transunet_rgbd_seg_from_encoded_tiny_h100"
TEST_DIR="/DATA/disk0/hs_25/pa/all_dataset/test"
OUTPUT_BASE_DIR="test_results/rgbd_tiny_segfromencoded_all_epochs"
BATCH_SIZE=32
DEVICE="cuda"

echo "============================================================"
echo "批量评估 RGBD Tiny SegFromEncoded 所有 Best Checkpoints"
echo "============================================================"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_BASE_DIR"

# 获取所有best checkpoint文件，按epoch排序
checkpoints=(
    "$CHECKPOINT_DIR/checkpoint_epoch_1_best.pth"
    "$CHECKPOINT_DIR/checkpoint_epoch_2_best.pth"
    "$CHECKPOINT_DIR/checkpoint_epoch_3_best.pth"
    "$CHECKPOINT_DIR/checkpoint_epoch_5_best.pth"
    "$CHECKPOINT_DIR/checkpoint_epoch_6_best.pth"
    "$CHECKPOINT_DIR/checkpoint_epoch_9_best.pth"
    "$CHECKPOINT_DIR/checkpoint_epoch_10_best.pth"
    "$CHECKPOINT_DIR/checkpoint_epoch_12_best.pth"
    "$CHECKPOINT_DIR/checkpoint_epoch_15_best.pth"
    "$CHECKPOINT_DIR/checkpoint_epoch_20_best.pth"
)

total=${#checkpoints[@]}
current=0

# 存储所有结果的摘要
summary_file="$OUTPUT_BASE_DIR/all_epochs_summary.json"
echo "[" > "$summary_file"

for checkpoint in "${checkpoints[@]}"; do
    if [ ! -f "$checkpoint" ]; then
        echo "警告: 文件不存在 $checkpoint，跳过"
        continue
    fi
    
    # 提取epoch号
    epoch=$(echo "$checkpoint" | grep -oP 'epoch_\K\d+' | head -1)
    current=$((current + 1))
    
    echo ""
    echo "============================================================"
    echo "[$current/$total] 评估 Epoch $epoch"
    echo "============================================================"
    echo "Checkpoint: $checkpoint"
    echo "输出目录: $OUTPUT_BASE_DIR/epoch_${epoch}"
    echo ""
    
    # 运行评估
    python test_models.py \
        --checkpoint "$checkpoint" \
        --test_dir "$TEST_DIR" \
        --batch_size "$BATCH_SIZE" \
        --device "$DEVICE" \
        --output_dir "$OUTPUT_BASE_DIR/epoch_${epoch}" \
        --save_failures \
        --num_failures 10
    
    # 检查评估是否成功
    if [ $? -eq 0 ]; then
        # 读取metrics.json并添加到摘要
        metrics_file="$OUTPUT_BASE_DIR/epoch_${epoch}/metrics.json"
        if [ -f "$metrics_file" ]; then
            # 添加epoch信息到metrics
            if [ $current -gt 1 ]; then
                echo "," >> "$summary_file"
            fi
            echo "  {" >> "$summary_file"
            echo "    \"epoch\": $epoch," >> "$summary_file"
            echo "    \"checkpoint\": \"$checkpoint\"," >> "$summary_file"
            python3 -c "
import json
with open('$metrics_file', 'r') as f:
    metrics = json.load(f)
    print('    ' + json.dumps(metrics, indent=2).replace(chr(10), chr(10) + '    ')[2:-2])
" >> "$summary_file"
            echo "  }" >> "$summary_file"
            
            echo ""
            echo "✅ Epoch $epoch 评估完成"
        else
            echo "⚠️  警告: 未找到 metrics.json 文件"
        fi
    else
        echo "❌ Epoch $epoch 评估失败"
    fi
done

echo "]" >> "$summary_file"

echo ""
echo "============================================================"
echo "所有评估完成！"
echo "============================================================"
echo "结果摘要已保存到: $summary_file"
echo ""
echo "生成对比报告..."

# 生成对比报告
python3 << 'PYEOF'
import json
from pathlib import Path

summary_file = Path("test_results/rgbd_tiny_segfromencoded_all_epochs/all_epochs_summary.json")
if summary_file.exists():
    with open(summary_file, 'r') as f:
        results = json.load(f)
    
    print("\n" + "=" * 70)
    print("所有 Epoch 性能对比")
    print("=" * 70)
    print(f"{'Epoch':<8} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1':<10} {'IoU':<10} {'Dice':<10}")
    print("-" * 70)
    
    for r in sorted(results, key=lambda x: x['epoch']):
        epoch = r['epoch']
        metrics = {k: v for k, v in r.items() if k not in ['epoch', 'checkpoint']}
        
        acc = metrics.get('cls_accuracy', 0)
        prec = metrics.get('cls_precision', 0)
        rec = metrics.get('cls_recall', 0)
        f1 = metrics.get('cls_f1', 0)
        iou = metrics.get('seg_iou', 0)
        dice = metrics.get('seg_dice', 0)
        
        print(f"{epoch:<8} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f} {iou:<10.4f} {dice:<10.4f}")
    
    # 找出最佳epoch
    best_f1 = max(results, key=lambda x: x.get('cls_f1', 0))
    best_acc = max(results, key=lambda x: x.get('cls_accuracy', 0))
    best_iou = max(results, key=lambda x: x.get('seg_iou', 0))
    
    print("\n" + "=" * 70)
    print("最佳性能:")
    print(f"  最佳F1: Epoch {best_f1['epoch']} (F1={best_f1.get('cls_f1', 0):.4f})")
    print(f"  最佳准确率: Epoch {best_acc['epoch']} (Acc={best_acc.get('cls_accuracy', 0):.4f})")
    print(f"  最佳IoU: Epoch {best_iou['epoch']} (IoU={best_iou.get('seg_iou', 0):.4f})")
    print("=" * 70)
PYEOF

echo ""
echo "完成！"
