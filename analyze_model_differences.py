#!/usr/bin/env python
"""
分析cls_only和Seg-From-Encoded模型的预测差异
找出：cls_only预测正确，但Seg-From-Encoded预测错误的样本
"""

import json
from pathlib import Path
from typing import List, Dict


def load_predictions(predictions_file: Path) -> Dict[str, Dict]:
    """加载预测结果，以rgb_path+removal_id为key"""
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    # 创建索引字典
    indexed = {}
    for pred in predictions:
        key = f"{pred['rgb_path']}::{pred['removal_id']}"
        indexed[key] = pred
    
    return indexed


def find_predictions_file(base_dir: Path, model_name: str) -> Path:
    """自动查找预测结果文件"""
    # 可能的路径（包括RGB和RGBD版本）
    possible_paths = [
        base_dir / f'{model_name}/predictions.json',
        base_dir / f'{model_name}_test/predictions.json',
        base_dir / f'{model_name}_tiny_h100/predictions.json',
        base_dir / f'rgbd_tiny_{model_name}/predictions.json',  # RGBD版本
        base_dir / f'rgb_tiny_{model_name}/predictions.json',   # RGB版本
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='分析模型预测差异')
    parser.add_argument('--cls_only_file', type=str, default=None,
                       help='cls_only预测结果文件路径（可选，自动检测）')
    parser.add_argument('--seg_file', type=str, default=None,
                       help='seg_from_encoded预测结果文件路径（可选，自动检测）')
    parser.add_argument('--output_file', type=str, default='model_differences.json',
                       help='输出文件路径')
    parser.add_argument('--base_dir', type=str, default='test_results',
                       help='测试结果基础目录')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    output_file = Path(args.output_file)
    
    print("=" * 60)
    print("分析模型预测差异")
    print("=" * 60)
    
    # 查找或使用指定的预测结果文件
    if args.cls_only_file:
        cls_only_pred_file = Path(args.cls_only_file)
    else:
        print(f"\n自动查找 cls_only 预测结果文件...")
        cls_only_pred_file = find_predictions_file(base_dir, 'cls_only')
        if cls_only_pred_file is None:
            # 尝试查找所有可能的文件
            all_files = list(base_dir.glob('**/predictions.json'))
            cls_only_files = [f for f in all_files if 'cls_only' in str(f)]
            if cls_only_files:
                cls_only_pred_file = cls_only_files[0]
                print(f"  找到: {cls_only_pred_file}")
            else:
                print(f"错误: 在 {base_dir} 中找不到 cls_only 预测结果文件")
                print("请先运行 cls_only 模型的测试，或使用 --cls_only_file 指定路径")
                return
        else:
            print(f"  找到: {cls_only_pred_file}")
    
    if args.seg_file:
        seg_from_encoded_pred_file = Path(args.seg_file)
    else:
        print(f"\n自动查找 seg_from_encoded 预测结果文件...")
        seg_from_encoded_pred_file = find_predictions_file(base_dir, 'seg_from_encoded')
        if seg_from_encoded_pred_file is None:
            # 尝试查找所有可能的文件
            all_files = list(base_dir.glob('**/predictions.json'))
            seg_files = [f for f in all_files if 'seg_from_encoded' in str(f)]
            if seg_files:
                seg_from_encoded_pred_file = seg_files[0]
                print(f"  找到: {seg_from_encoded_pred_file}")
            else:
                print(f"错误: 在 {base_dir} 中找不到 seg_from_encoded 预测结果文件")
                print("请先运行 seg_from_encoded 模型的测试，或使用 --seg_file 指定路径")
                return
        else:
            print(f"  找到: {seg_from_encoded_pred_file}")
    
    # 检查文件是否存在
    if not cls_only_pred_file.exists():
        print(f"错误: 找不到文件 {cls_only_pred_file}")
        return
    
    if not seg_from_encoded_pred_file.exists():
        print(f"错误: 找不到文件 {seg_from_encoded_pred_file}")
        return
    
    print(f"\n使用文件:")
    print(f"  cls_only: {cls_only_pred_file}")
    print(f"  seg_from_encoded: {seg_from_encoded_pred_file}")
    
    # 加载预测结果
    print(f"\n加载 cls_only 预测结果: {cls_only_pred_file}")
    cls_only_preds = load_predictions(cls_only_pred_file)
    print(f"  加载了 {len(cls_only_preds)} 个样本")
    
    print(f"\n加载 Seg-From-Encoded 预测结果: {seg_from_encoded_pred_file}")
    seg_preds = load_predictions(seg_from_encoded_pred_file)
    print(f"  加载了 {len(seg_preds)} 个样本")
    
    # 找出交集（两个模型都预测的样本）
    common_keys = set(cls_only_preds.keys()) & set(seg_preds.keys())
    print(f"\n共同样本数: {len(common_keys)}")
    
    # 找出cls_only预测正确但Seg-From-Encoded预测错误的样本
    differences = []
    cls_only_correct_count = 0
    seg_wrong_count = 0
    
    for key in common_keys:
        cls_pred = cls_only_preds[key]
        seg_pred = seg_preds[key]
        
        # cls_only预测是否正确
        cls_correct = cls_pred['cls_pred'] == cls_pred['cls_target']
        
        # Seg-From-Encoded预测是否错误（分类错误或分割IoU低）
        seg_wrong = False
        seg_error_type = []
        
        if seg_pred['cls_pred'] != seg_pred['cls_target']:
            seg_wrong = True
            seg_error_type.append('cls_error')
        
        if seg_pred.get('seg_iou') is not None and seg_pred['seg_iou'] < 0.3:
            seg_wrong = True
            seg_error_type.append(f'low_iou_{seg_pred["seg_iou"]:.3f}')
        
        # 统计
        if cls_correct:
            cls_only_correct_count += 1
        
        if seg_wrong:
            seg_wrong_count += 1
        
        # 找出差异样本：cls_only对，但Seg-From-Encoded错
        if cls_correct and seg_wrong:
            differences.append({
                'rgb_path': cls_pred['rgb_path'],
                'removal_id': cls_pred['removal_id'],
                'cls_only': {
                    'cls_pred': cls_pred['cls_pred'],
                    'cls_target': cls_pred['cls_target'],
                    'cls_prob': cls_pred['cls_prob'],
                    'correct': True,
                },
                'seg_from_encoded': {
                    'cls_pred': seg_pred['cls_pred'],
                    'cls_target': seg_pred['cls_target'],
                    'cls_prob': seg_pred['cls_prob'],
                    'seg_iou': seg_pred.get('seg_iou'),
                    'correct': False,
                    'error_type': seg_error_type,
                },
            })
    
    # 打印统计信息
    print(f"\n" + "=" * 60)
    print("统计信息:")
    print(f"  cls_only 预测正确: {cls_only_correct_count} ({cls_only_correct_count/len(common_keys)*100:.1f}%)")
    print(f"  Seg-From-Encoded 预测错误: {seg_wrong_count} ({seg_wrong_count/len(common_keys)*100:.1f}%)")
    print(f"  差异样本数 (cls_only对但Seg-From-Encoded错): {len(differences)}")
    print("=" * 60)
    
    # 按错误类型分类
    if differences:
        cls_errors = sum(1 for d in differences if 'cls_error' in d['seg_from_encoded']['error_type'])
        iou_errors = sum(1 for d in differences if any('low_iou' in e for e in d['seg_from_encoded']['error_type']))
        both_errors = sum(1 for d in differences if len(d['seg_from_encoded']['error_type']) > 1)
        
        print(f"\n差异样本错误类型:")
        print(f"  仅分类错误: {cls_errors}")
        print(f"  仅分割IoU低: {iou_errors}")
        print(f"  分类+分割都错: {both_errors}")
    
    # 保存结果
    output_data = {
        'summary': {
            'total_common_samples': len(common_keys),
            'cls_only_correct': cls_only_correct_count,
            'seg_from_encoded_wrong': seg_wrong_count,
            'difference_samples': len(differences),
        },
        'differences': differences,
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    print(f"  差异样本数: {len(differences)}")
    
    # 打印前10个差异样本的路径
    if differences:
        print(f"\n前10个差异样本:")
        for i, diff in enumerate(differences[:10], 1):
            print(f"  {i}. {diff['rgb_path']} (removal_id: {diff['removal_id']})")
            print(f"     cls_only: {diff['cls_only']['cls_pred']} (prob: {diff['cls_only']['cls_prob']:.3f})")
            print(f"     seg_from_encoded: {diff['seg_from_encoded']['cls_pred']} (prob: {diff['seg_from_encoded']['cls_prob']:.3f})")
            if diff['seg_from_encoded'].get('seg_iou') is not None:
                print(f"     seg_iou: {diff['seg_from_encoded']['seg_iou']:.3f}")
            print(f"     错误类型: {', '.join(diff['seg_from_encoded']['error_type'])}")


if __name__ == '__main__':
    main()
