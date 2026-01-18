"""
将数据集按照 0.7:0.2:0.1 的比例随机划分为训练集、验证集和测试集
并复制到新的目录结构中
"""
import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm


def collect_round_dirs(data_root: str, dataset_names: list) -> list:
    """
    收集所有数据集中的 round 目录
    
    Args:
        data_root: 数据根目录
        dataset_names: 数据集名称列表
    
    Returns:
        round 目录路径列表
    """
    round_dirs = []
    
    for dataset_name in dataset_names:
        dataset_path = Path(data_root) / dataset_name
        if not dataset_path.exists():
            print(f"警告: 数据集目录不存在: {dataset_path}")
            continue
        
        # 收集所有 round_* 目录
        for item in dataset_path.iterdir():
            if item.is_dir() and item.name.startswith('round_'):
                round_dirs.append(item)
    
    return round_dirs


def copy_round_dir(src_dir: Path, dst_dir: Path):
    """
    复制整个 round 目录到目标位置
    
    Args:
        src_dir: 源目录
        dst_dir: 目标目录
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制所有文件和子目录
    for item in src_dir.iterdir():
        src_path = item
        dst_path = dst_dir / item.name
        
        if item.is_dir():
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)


def split_dataset(
    data_root: str = '/DATA/disk0/hs_25/pa',
    dataset_names: list = None,
    output_dir: str = '/DATA/disk0/hs_25/pa/all_dataset',
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """
    划分数据集并复制到新目录
    
    Args:
        data_root: 数据根目录
        dataset_names: 数据集名称列表，如果为 None 则自动检测
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    """
    # 检查比例
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"比例总和必须为 1.0，当前为 {total_ratio}")
    
    # 自动检测数据集
    if dataset_names is None:
        data_root_path = Path(data_root)
        dataset_names = [
            d.name for d in data_root_path.iterdir()
            if d.is_dir() and d.name.startswith('dataset')
        ]
        print(f"自动检测到数据集: {dataset_names}")
    
    # 收集所有 round 目录
    print("\n收集数据集...")
    round_dirs = collect_round_dirs(data_root, dataset_names)
    print(f"找到 {len(round_dirs)} 个 round 目录")
    
    if len(round_dirs) == 0:
        print("错误: 未找到任何 round 目录!")
        return
    
    # 随机打乱
    print(f"\n使用随机种子 {seed} 打乱数据...")
    random.seed(seed)
    random.shuffle(round_dirs)
    
    # 计算划分点
    n_total = len(round_dirs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val  # 剩余的全部给测试集
    
    train_dirs = round_dirs[:n_train]
    val_dirs = round_dirs[n_train:n_train + n_val]
    test_dirs = round_dirs[n_train + n_val:]
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_dirs)} 个 ({len(train_dirs)/n_total*100:.1f}%)")
    print(f"  验证集: {len(val_dirs)} 个 ({len(val_dirs)/n_total*100:.1f}%)")
    print(f"  测试集: {len(test_dirs)} 个 ({len(test_dirs)/n_total*100:.1f}%)")
    
    # 创建输出目录
    output_path = Path(output_dir)
    train_path = output_path / 'train'
    val_path = output_path / 'val'
    test_path = output_path / 'test'
    
    # 如果输出目录已存在，询问是否删除
    if output_path.exists():
        print(f"\n警告: 输出目录已存在: {output_path}")
        response = input("是否删除并重新创建? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(output_path)
            print("已删除旧目录")
        else:
            print("取消操作")
            return
    
    # 创建目录
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
    
    # 复制文件
    print(f"\n复制训练集到 {train_path}...")
    for round_dir in tqdm(train_dirs, desc="训练集"):
        dst_dir = train_path / round_dir.name
        copy_round_dir(round_dir, dst_dir)
    
    print(f"\n复制验证集到 {val_path}...")
    for round_dir in tqdm(val_dirs, desc="验证集"):
        dst_dir = val_path / round_dir.name
        copy_round_dir(round_dir, dst_dir)
    
    print(f"\n复制测试集到 {test_path}...")
    for round_dir in tqdm(test_dirs, desc="测试集"):
        dst_dir = test_path / round_dir.name
        copy_round_dir(round_dir, dst_dir)
    
    print(f"\n完成! 数据集已划分并复制到: {output_path}")
    print(f"  训练集: {train_path} ({len(train_dirs)} 个)")
    print(f"  验证集: {val_path} ({len(val_dirs)} 个)")
    print(f"  测试集: {test_path} ({len(test_dirs)} 个)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="划分数据集")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/DATA/disk0/hs_25/pa",
        help="数据根目录"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="数据集名称列表 (如: dataset1 dataset2 dataset3 dataset_messy_small)，如果不指定则自动检测"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/DATA/disk0/hs_25/pa/all_dataset",
        help="输出目录"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="训练集比例"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="验证集比例"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="测试集比例"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    args = parser.parse_args()
    
    split_dataset(
        data_root=args.data_root,
        dataset_names=args.datasets,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
