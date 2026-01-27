#!/usr/bin/env python3
"""
创建 train_v2 数据集
- 复制整个训练集到 train_v2
- 随机选择一半的测试集添加到 train_v2
- 不修改原始数据集

用法:
    python create_train_v2.py
    python create_train_v2.py --dry-run  # 仅预览，不实际操作
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from tqdm import tqdm


def create_train_v2(
    data_root: str = '/DATA/disk0/hs_25/pa/all_dataset',
    seed: int = 42,
    dry_run: bool = False,
):
    """创建 train_v2 数据集"""
    
    data_root = Path(data_root)
    train_dir = data_root / 'train'
    test_dir = data_root / 'test'
    train_v2_dir = data_root / 'train_v2'
    
    # 检查源目录是否存在
    if not train_dir.exists():
        raise FileNotFoundError(f"训练集目录不存在: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"测试集目录不存在: {test_dir}")
    
    # 获取所有样本文件夹
    train_samples = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    test_samples = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    
    print("=" * 60)
    print("创建 train_v2 数据集")
    print("=" * 60)
    print(f"训练集样本数: {len(train_samples)}")
    print(f"测试集样本数: {len(test_samples)}")
    
    # 随机选择一半测试集
    random.seed(seed)
    half_test_count = len(test_samples) // 2
    selected_test_samples = random.sample(test_samples, half_test_count)
    
    print(f"选择测试集样本数: {len(selected_test_samples)} (一半)")
    print(f"train_v2 总样本数: {len(train_samples) + len(selected_test_samples)}")
    print(f"目标目录: {train_v2_dir}")
    print("-" * 60)
    
    if dry_run:
        print("[DRY RUN] 仅预览，不实际操作")
        print("\n将复制的训练集样本:")
        for i, sample in enumerate(train_samples[:5]):
            print(f"  - {sample.name}")
        if len(train_samples) > 5:
            print(f"  ... 共 {len(train_samples)} 个")
        
        print("\n将添加的测试集样本:")
        for i, sample in enumerate(selected_test_samples[:10]):
            print(f"  - {sample.name}")
        if len(selected_test_samples) > 10:
            print(f"  ... 共 {len(selected_test_samples)} 个")
        
        return
    
    # 检查目标目录是否已存在
    if train_v2_dir.exists():
        response = input(f"目标目录已存在: {train_v2_dir}\n是否删除并重新创建? [y/N]: ")
        if response.lower() != 'y':
            print("取消操作")
            return
        print(f"删除已存在目录: {train_v2_dir}")
        shutil.rmtree(train_v2_dir)
    
    # 创建目标目录
    train_v2_dir.mkdir(parents=True, exist_ok=True)
    print(f"创建目录: {train_v2_dir}")
    
    # 复制训练集样本
    print("\n[1/2] 复制训练集样本...")
    for sample in tqdm(train_samples, desc="复制训练集"):
        src = sample
        dst = train_v2_dir / sample.name
        shutil.copytree(src, dst)
    
    # 复制选中的测试集样本
    print("\n[2/2] 复制测试集样本...")
    for sample in tqdm(selected_test_samples, desc="复制测试集"):
        src = sample
        dst = train_v2_dir / sample.name
        
        # 检查是否有同名冲突
        if dst.exists():
            # 如果有冲突，加上后缀
            dst = train_v2_dir / f"{sample.name}_from_test"
            print(f"  警告: 重名，重命名为 {dst.name}")
        
        shutil.copytree(src, dst)
    
    # 统计结果
    final_samples = list(train_v2_dir.iterdir())
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"train_v2 总样本数: {len(final_samples)}")
    print(f"  - 来自训练集: {len(train_samples)}")
    print(f"  - 来自测试集: {len(selected_test_samples)}")
    print(f"目录: {train_v2_dir}")
    
    # 保存记录
    record_file = train_v2_dir.parent / 'train_v2_info.txt'
    with open(record_file, 'w') as f:
        f.write(f"train_v2 数据集信息\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"随机种子: {seed}\n")
        f.write(f"训练集样本数: {len(train_samples)}\n")
        f.write(f"测试集样本数 (总): {len(test_samples)}\n")
        f.write(f"选择的测试集样本数: {len(selected_test_samples)}\n")
        f.write(f"train_v2 总样本数: {len(final_samples)}\n")
        f.write(f"\n选择的测试集样本:\n")
        for sample in sorted(selected_test_samples, key=lambda x: x.name):
            f.write(f"  - {sample.name}\n")
    
    print(f"记录已保存: {record_file}")


def main():
    parser = argparse.ArgumentParser(description='创建 train_v2 数据集')
    parser.add_argument('--data-root', type=str, default='/DATA/disk0/hs_25/pa/all_dataset',
                        help='数据集根目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--dry-run', action='store_true', help='仅预览，不实际操作')
    
    args = parser.parse_args()
    
    create_train_v2(
        data_root=args.data_root,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()
