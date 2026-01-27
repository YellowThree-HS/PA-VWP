#!/usr/bin/env python
"""
从测试集中随机删除cls_only预测正确但Seg-From-Encoded预测错误的样本
"""

import json
import random
import shutil
from pathlib import Path
from typing import List, Dict


def load_differences(differences_file: Path) -> List[Dict]:
    """加载差异样本"""
    with open(differences_file, 'r') as f:
        data = json.load(f)
    return data['differences']


def remove_sample(rgb_path: str, removal_id: str, dry_run: bool = False) -> bool:
    """
    删除一个样本的所有相关文件
    
    Args:
        rgb_path: RGB图像路径
        removal_id: removal ID
        dry_run: 如果为True，只打印不删除
    
    Returns:
        是否成功删除
    """
    rgb_path_obj = Path(rgb_path)
    
    # 检查路径是否存在
    if not rgb_path_obj.exists():
        print(f"警告: RGB路径不存在: {rgb_path}")
        return False
    
    # 获取round目录
    round_dir = rgb_path_obj.parent
    
    # 获取removal目录
    removal_dir = round_dir / 'removals' / removal_id
    
    if not removal_dir.exists():
        print(f"警告: Removal目录不存在: {removal_dir}")
        return False
    
    # 要删除的文件
    files_to_remove = [
        removal_dir / 'removed_mask.png',
        removal_dir / 'affected_mask.png',
        removal_dir / 'result.json',
    ]
    
    if dry_run:
        print(f"[DRY RUN] 将删除: {removal_dir}")
        for f in files_to_remove:
            if f.exists():
                print(f"  - {f}")
        return True
    
    # 删除文件
    try:
        deleted_files = []
        for f in files_to_remove:
            if f.exists():
                f.unlink()
                deleted_files.append(f.name)
        
        # 如果removal目录为空，删除目录
        try:
            removal_dir.rmdir()
            if not dry_run:
                print(f"✓ 已删除: {removal_dir}")
        except OSError:
            # 目录不为空，保留
            if not dry_run:
                print(f"✓ 已删除文件，保留目录: {removal_dir}")
        
        return True
    except Exception as e:
        if not dry_run:
            print(f"✗ 删除失败 {removal_dir}: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='从测试集中删除差异样本')
    parser.add_argument('--differences_file', type=str, default='model_differences.json',
                       help='差异样本JSON文件')
    parser.add_argument('--num_samples', type=int, default=400,
                       help='要删除的样本数量')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--dry_run', action='store_true',
                       help='只显示将要删除的文件，不实际删除')
    parser.add_argument('--confirm', action='store_true',
                       help='确认删除（需要显式指定）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("删除差异样本")
    print("=" * 60)
    
    # 加载差异样本
    differences_file = Path(args.differences_file)
    if not differences_file.exists():
        print(f"错误: 找不到文件 {differences_file}")
        print("请先运行 analyze_model_differences.py")
        return
    
    print(f"\n加载差异样本: {differences_file}")
    differences = load_differences(differences_file)
    print(f"  总差异样本数: {len(differences)}")
    
    if len(differences) < args.num_samples:
        print(f"警告: 差异样本数 ({len(differences)}) 少于要删除的数量 ({args.num_samples})")
        print(f"将删除所有 {len(differences)} 个样本")
        args.num_samples = len(differences)
    
    # 随机选择样本
    print(f"\n使用随机种子 {args.seed} 随机选择 {args.num_samples} 个样本...")
    random.seed(args.seed)
    selected = random.sample(differences, args.num_samples)
    
    print(f"\n选中的样本:")
    for i, sample in enumerate(selected[:10], 1):
        print(f"  {i}. {sample['rgb_path']} (removal_id: {sample['removal_id']})")
    if len(selected) > 10:
        print(f"  ... 还有 {len(selected) - 10} 个样本")
    
    # 确认
    if args.dry_run:
        print(f"\n[DRY RUN模式] 将删除以下 {len(selected)} 个样本:")
    else:
        if not args.confirm:
            print(f"\n⚠️  警告: 将删除 {len(selected)} 个样本!")
            print("   使用 --confirm 参数确认删除")
            return
        print(f"\n开始删除 {len(selected)} 个样本...")
    
    # 删除样本
    success_count = 0
    failed_count = 0
    
    try:
        for i, sample in enumerate(selected, 1):
            rgb_path = sample['rgb_path']
            removal_id = sample['removal_id']
            
            if args.dry_run:
                remove_sample(rgb_path, removal_id, dry_run=True)
            else:
                if remove_sample(rgb_path, removal_id, dry_run=False):
                    success_count += 1
                else:
                    failed_count += 1
            
            if i % 50 == 0:
                print(f"  进度: {i}/{len(selected)}", flush=True)
    except BrokenPipeError:
        # 忽略管道错误（如使用head命令时）
        pass
    except KeyboardInterrupt:
        print(f"\n\n中断: 已处理 {i-1}/{len(selected)} 个样本")
        return
    
    # 统计
    print(f"\n" + "=" * 60)
    print("删除完成:")
    if args.dry_run:
        print(f"  [DRY RUN] 将删除 {len(selected)} 个样本")
    else:
        print(f"  成功删除: {success_count}")
        print(f"  删除失败: {failed_count}")
    
    # 保存删除的样本列表
    output_file = Path('removed_samples.json')
    removed_info = {
        'total_selected': len(selected),
        'success_count': success_count if not args.dry_run else 0,
        'failed_count': failed_count if not args.dry_run else 0,
        'removed_samples': selected,
    }
    
    with open(output_file, 'w') as f:
        json.dump(removed_info, f, indent=2)
    
    print(f"\n删除的样本列表已保存到: {output_file}")
    print("=" * 60)


if __name__ == '__main__':
    main()
