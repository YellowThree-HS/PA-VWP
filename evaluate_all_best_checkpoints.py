#!/usr/bin/env python
"""
批量评估RGBD tiny segfromencoded模型的所有best checkpoint
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict

CHECKPOINT_DIR = Path("/DATA/disk0/hs_25/pa/checkpoints/transunet_rgbd_seg_from_encoded_tiny_h100")
TEST_DIR = "/DATA/disk0/hs_25/pa/all_dataset/test"
OUTPUT_BASE_DIR = Path("test_results/rgbd_tiny_segfromencoded_all_epochs")
BATCH_SIZE = 32
DEVICE = "cuda"


def find_all_best_checkpoints() -> List[Path]:
    """查找所有best checkpoint文件"""
    checkpoints = sorted(
        CHECKPOINT_DIR.glob("*best.pth"),
        key=lambda x: int(x.stem.split("_")[2]) if x.stem.split("_")[2].isdigit() else 0
    )
    return checkpoints


def extract_epoch(checkpoint_path: Path) -> int:
    """从checkpoint路径提取epoch号"""
    name = checkpoint_path.stem
    # checkpoint_epoch_20_best -> 20
    parts = name.split("_")
    for i, part in enumerate(parts):
        if part == "epoch" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return 0


def evaluate_checkpoint(checkpoint_path: Path, epoch: int, total: int, current: int) -> Dict:
    """评估单个checkpoint"""
    output_dir = OUTPUT_BASE_DIR / f"epoch_{epoch}"
    
    print("\n" + "=" * 70)
    print(f"[{current}/{total}] 评估 Epoch {epoch}")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 运行评估
    cmd = [
        sys.executable, "test_models.py",
        "--checkpoint", str(checkpoint_path),
        "--test_dir", TEST_DIR,
        "--batch_size", str(BATCH_SIZE),
        "--device", DEVICE,
        "--output_dir", str(output_dir),
        "--save_failures",
        "--num_failures", "10"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        # 读取metrics
        metrics_file = output_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            print(f"\n✅ Epoch {epoch} 评估完成")
            print(f"   准确率: {metrics.get('cls_accuracy', 0):.4f}")
            print(f"   F1: {metrics.get('cls_f1', 0):.4f}")
            if 'seg_iou' in metrics:
                print(f"   IoU: {metrics.get('seg_iou', 0):.4f}")
            
            return {
                'epoch': epoch,
                'checkpoint': str(checkpoint_path),
                **metrics
            }
        else:
            print(f"⚠️  警告: 未找到 metrics.json 文件")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Epoch {epoch} 评估失败")
        print(f"错误输出: {e.stderr}")
        return None


def generate_summary(all_results: List[Dict]):
    """生成对比报告"""
    if not all_results:
        print("没有评估结果")
        return
    
    # 保存JSON摘要
    summary_file = OUTPUT_BASE_DIR / "all_epochs_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("所有 Epoch 性能对比")
    print("=" * 70)
    print(f"{'Epoch':<8} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1':<10} {'IoU':<10} {'Dice':<10}")
    print("-" * 70)
    
    for r in sorted(all_results, key=lambda x: x['epoch']):
        epoch = r['epoch']
        acc = r.get('cls_accuracy', 0)
        prec = r.get('cls_precision', 0)
        rec = r.get('cls_recall', 0)
        f1 = r.get('cls_f1', 0)
        iou = r.get('seg_iou', 0)
        dice = r.get('seg_dice', 0)
        
        print(f"{epoch:<8} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f} {iou:<10.4f} {dice:<10.4f}")
    
    # 找出最佳epoch
    best_f1 = max(all_results, key=lambda x: x.get('cls_f1', 0))
    best_acc = max(all_results, key=lambda x: x.get('cls_accuracy', 0))
    best_iou = max(all_results, key=lambda x: x.get('seg_iou', 0))
    
    print("\n" + "=" * 70)
    print("最佳性能:")
    print(f"  最佳F1: Epoch {best_f1['epoch']} (F1={best_f1.get('cls_f1', 0):.4f})")
    print(f"  最佳准确率: Epoch {best_acc['epoch']} (Acc={best_acc.get('cls_accuracy', 0):.4f})")
    if 'seg_iou' in best_iou:
        print(f"  最佳IoU: Epoch {best_iou['epoch']} (IoU={best_iou.get('seg_iou', 0):.4f})")
    print("=" * 70)
    print(f"\n结果摘要已保存到: {summary_file}")


def main():
    print("=" * 70)
    print("批量评估 RGBD Tiny SegFromEncoded 所有 Best Checkpoints")
    print("=" * 70)
    
    # 创建输出目录
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 查找所有checkpoint
    checkpoints = find_all_best_checkpoints()
    
    if not checkpoints:
        print(f"错误: 在 {CHECKPOINT_DIR} 中未找到任何best checkpoint文件")
        return
    
    print(f"\n找到 {len(checkpoints)} 个best checkpoint文件")
    
    # 评估所有checkpoint
    all_results = []
    for i, checkpoint in enumerate(checkpoints, 1):
        epoch = extract_epoch(checkpoint)
        result = evaluate_checkpoint(checkpoint, epoch, len(checkpoints), i)
        if result:
            all_results.append(result)
    
    # 生成摘要报告
    if all_results:
        generate_summary(all_results)
    else:
        print("\n没有成功的评估结果")


if __name__ == '__main__':
    main()
