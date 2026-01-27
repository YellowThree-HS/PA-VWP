#!/usr/bin/env python
"""
统计test和val数据集的样本数量
"""

import json
from pathlib import Path
from collections import defaultdict


def count_samples_in_dir(data_dir: Path) -> dict:
    """统计目录下的样本数量"""
    if not data_dir.exists():
        return {
            'total_samples': 0,
            'round_dirs': 0,
            'stable_samples': 0,
            'unstable_samples': 0,
            'error_samples': 0,
        }
    
    total_samples = 0
    stable_samples = 0
    unstable_samples = 0
    error_samples = 0
    round_dirs = 0
    
    # 遍历所有round目录
    for round_dir in sorted(data_dir.iterdir()):
        if not round_dir.is_dir() or not round_dir.name.startswith('round_'):
            continue
        
        round_dirs += 1
        removals_dir = round_dir / 'removals'
        
        if not removals_dir.exists():
            continue
        
        # 遍历removals下的所有编号目录
        for removal_dir in sorted(removals_dir.iterdir()):
            if not removal_dir.is_dir():
                continue
            
            result_json = removal_dir / 'result.json'
            if not result_json.exists():
                error_samples += 1
                continue
            
            try:
                with open(result_json, 'r') as f:
                    result = json.load(f)
                
                total_samples += 1
                
                # 统计稳定性
                is_stable = result.get('is_stable', False)
                if is_stable:
                    stable_samples += 1
                else:
                    unstable_samples += 1
            except Exception as e:
                error_samples += 1
                print(f"警告: 无法读取 {result_json}: {e}")
    
    return {
        'total_samples': total_samples,
        'round_dirs': round_dirs,
        'stable_samples': stable_samples,
        'unstable_samples': unstable_samples,
        'error_samples': error_samples,
    }


def main():
    # 数据集路径
    base_dir = Path('/DATA/disk0/hs_25/pa/all_dataset')
    test_dir = base_dir / 'test'
    val_dir = base_dir / 'val'
    train_dir = base_dir / 'train'
    
    print("=" * 60)
    print("数据集统计")
    print("=" * 60)
    
    # 统计test集
    print(f"\n📊 测试集 (test):")
    print(f"   路径: {test_dir}")
    test_stats = count_samples_in_dir(test_dir)
    print(f"   Round目录数: {test_stats['round_dirs']}")
    print(f"   总样本数: {test_stats['total_samples']}")
    print(f"   ├─ 稳定样本: {test_stats['stable_samples']} ({test_stats['stable_samples']/max(test_stats['total_samples'],1)*100:.1f}%)")
    print(f"   ├─ 不稳定样本: {test_stats['unstable_samples']} ({test_stats['unstable_samples']/max(test_stats['total_samples'],1)*100:.1f}%)")
    if test_stats['error_samples'] > 0:
        print(f"   └─ 错误样本: {test_stats['error_samples']}")
    
    # 统计val集
    print(f"\n📊 验证集 (val):")
    print(f"   路径: {val_dir}")
    val_stats = count_samples_in_dir(val_dir)
    print(f"   Round目录数: {val_stats['round_dirs']}")
    print(f"   总样本数: {val_stats['total_samples']}")
    print(f"   ├─ 稳定样本: {val_stats['stable_samples']} ({val_stats['stable_samples']/max(val_stats['total_samples'],1)*100:.1f}%)")
    print(f"   ├─ 不稳定样本: {val_stats['unstable_samples']} ({val_stats['unstable_samples']/max(val_stats['total_samples'],1)*100:.1f}%)")
    if val_stats['error_samples'] > 0:
        print(f"   └─ 错误样本: {val_stats['error_samples']}")
    
    # 统计train集（可选）
    if train_dir.exists():
        print(f"\n📊 训练集 (train):")
        print(f"   路径: {train_dir}")
        train_stats = count_samples_in_dir(train_dir)
        print(f"   Round目录数: {train_stats['round_dirs']}")
        print(f"   总样本数: {train_stats['total_samples']}")
        print(f"   ├─ 稳定样本: {train_stats['stable_samples']} ({train_stats['stable_samples']/max(train_stats['total_samples'],1)*100:.1f}%)")
        print(f"   ├─ 不稳定样本: {train_stats['unstable_samples']} ({train_stats['unstable_samples']/max(train_stats['total_samples'],1)*100:.1f}%)")
        if train_stats['error_samples'] > 0:
            print(f"   └─ 错误样本: {train_stats['error_samples']}")
    
    # 汇总
    total_all = test_stats['total_samples'] + val_stats['total_samples']
    if train_dir.exists():
        total_all += train_stats['total_samples']
    
    print(f"\n" + "=" * 60)
    print("汇总:")
    print(f"   测试集: {test_stats['total_samples']} 个样本")
    print(f"   验证集: {val_stats['total_samples']} 个样本")
    if train_dir.exists():
        print(f"   训练集: {train_stats['total_samples']} 个样本")
        print(f"   总计: {total_all} 个样本")
    print("=" * 60)


if __name__ == '__main__':
    main()
