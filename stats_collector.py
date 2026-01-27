"""
统计数据集的round数量和数据对数量
"""
import os
import json
import sys
from collections import defaultdict
from pathlib import Path

VIEWS = ["top", "north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"]

def get_all_rounds(dataset_dir: str):
    """获取数据集中的所有round基础名称（去重）"""
    rounds_set = set()
    if not os.path.exists(dataset_dir):
        return []
    for name in sorted(os.listdir(dataset_dir)):
        if name.startswith("round_") and "_" in name:
            parts = name.rsplit("_", 1)
            if len(parts) == 2 and parts[1] in VIEWS:
                base_name = parts[0]  # 提取基础round名称（去掉视角后缀）
                rounds_set.add(base_name)
    return sorted(list(rounds_set))

def count_pairs_in_round(round_name: str, dataset_dir: str):
    """统计一个round中的数据对数量"""
    total_pairs = 0
    for view in VIEWS:
        view_dir = f"{round_name}_{view}"
        view_path = os.path.join(dataset_dir, view_dir)
        removals_dir = os.path.join(view_path, "removals")
        
        if os.path.exists(removals_dir):
            # 遍历removals目录下的所有编号目录
            for item in os.listdir(removals_dir):
                removal_path = os.path.join(removals_dir, item)
                if os.path.isdir(removal_path):
                    result_file = os.path.join(removal_path, "result.json")
                    # 如果存在result.json，认为是一个有效的数据对
                    if os.path.exists(result_file):
                        total_pairs += 1
    return total_pairs

def count_unique_boxes(round_name: str, dataset_dir: str):
    """统计一个round中唯一可见/稳定/不稳定的箱子"""
    unique_visible = set()
    unique_stable = set()
    unique_unstable = set()

    # 读取所有视角的 visible_boxes.json
    for view in VIEWS:
        view_dir = f"{round_name}_{view}"
        view_path = os.path.join(dataset_dir, view_dir)
        visible_file = os.path.join(view_path, "visible_boxes.json")

        if os.path.exists(visible_file):
            with open(visible_file, "r") as f:
                boxes = json.load(f)
                for box in boxes:
                    unique_visible.add(box["prim_path"])

    # 读取所有视角的测试结果
    for view in VIEWS:
        view_dir = f"{round_name}_{view}"
        view_path = os.path.join(dataset_dir, view_dir)
        removals_dir = os.path.join(view_path, "removals")

        if os.path.exists(removals_dir):
            for item in os.listdir(removals_dir):
                removal_path = os.path.join(removals_dir, item)
                if os.path.isdir(removal_path):
                    result_file = os.path.join(removal_path, "result.json")
                    if os.path.exists(result_file):
                        with open(result_file, "r") as f:
                            result = json.load(f)
                            box_path = result.get("removed_box", {}).get("prim_path", "")
                            if box_path:
                                if result.get("is_stable"):
                                    unique_stable.add(box_path)
                                else:
                                    unique_unstable.add(box_path)

    return len(unique_visible), len(unique_stable), len(unique_unstable)

def analyze_dataset(dataset_dir: str, dataset_name: str = None):
    """分析单个数据集"""
    if dataset_name is None:
        dataset_name = os.path.basename(os.path.abspath(dataset_dir))
    
    rounds = get_all_rounds(dataset_dir)
    if not rounds:
        return {
            "name": dataset_name,
            "rounds": 0,
            "pairs": 0,
            "unique_visible": 0,
            "unique_stable": 0,
            "unique_unstable": 0
        }
    
    total_pairs = 0
    total_unique_visible = 0
    total_unique_stable = 0
    total_unique_unstable = 0
    
    for round_name in rounds:
        pairs = count_pairs_in_round(round_name, dataset_dir)
        total_pairs += pairs
        
        visible, stable, unstable = count_unique_boxes(round_name, dataset_dir)
        total_unique_visible += visible
        total_unique_stable += stable
        total_unique_unstable += unstable
    
    return {
        "name": dataset_name,
        "rounds": len(rounds),
        "pairs": total_pairs,
        "unique_visible": total_unique_visible,
        "unique_stable": total_unique_stable,
        "unique_unstable": total_unique_unstable
    }

def main():
    # 从命令行参数获取数据集目录，如果没有则使用默认目录
    if len(sys.argv) > 1:
        # 如果提供了参数，使用参数作为数据集根目录
        data_root = sys.argv[1]
        # 查找所有数据集目录
        dataset_dirs = []
        if os.path.isdir(data_root):
            # 首先检查是否有train/val/test目录结构
            train_dir = os.path.join(data_root, "train")
            val_dir = os.path.join(data_root, "val")
            test_dir = os.path.join(data_root, "test")
            
            if os.path.isdir(train_dir) or os.path.isdir(val_dir) or os.path.isdir(test_dir):
                # 这是已划分的数据集结构
                if os.path.isdir(train_dir):
                    dataset_dirs.append((train_dir, "train"))
                if os.path.isdir(val_dir):
                    dataset_dirs.append((val_dir, "val"))
                if os.path.isdir(test_dir):
                    dataset_dirs.append((test_dir, "test"))
            else:
                # 查找所有以dataset开头的目录
                for item in sorted(os.listdir(data_root)):
                    item_path = os.path.join(data_root, item)
                    if os.path.isdir(item_path) and item.startswith("dataset"):
                        dataset_dirs.append((item_path, item))
    else:
        # 默认使用当前目录下的dataset目录
        dataset_dirs = [("dataset", "dataset")]
    
    if not dataset_dirs:
        print("未找到数据集目录")
        return
    
    print("=" * 70)
    print("数据集统计")
    print("=" * 70)
    print()
    
    all_results = []
    total_rounds = 0
    total_pairs = 0
    total_unique_visible = 0
    total_unique_stable = 0
    total_unique_unstable = 0
    
    # 统计每个数据集
    for dataset_dir, dataset_name in dataset_dirs:
        print(f"正在分析: {dataset_name} ({dataset_dir})")
        result = analyze_dataset(dataset_dir, dataset_name)
        all_results.append(result)
        
        total_rounds += result["rounds"]
        total_pairs += result["pairs"]
        total_unique_visible += result["unique_visible"]
        total_unique_stable += result["unique_stable"]
        total_unique_unstable += result["unique_unstable"]
        
        print(f"  Round数: {result['rounds']}")
        print(f"  数据对数: {result['pairs']}")
        print()
    
    # 打印详细统计表
    print("=" * 70)
    print("详细统计")
    print("=" * 70)
    print(f"{'数据集':<25} {'Round数':<12} {'数据对数':<12}")
    print("-" * 70)
    for result in all_results:
        print(f"{result['name']:<25} {result['rounds']:<12} {result['pairs']:<12}")
    print("-" * 70)
    print(f"{'总计':<25} {total_rounds:<12} {total_pairs:<12}")
    print("=" * 70)
    
    # 打印最终汇总
    print()
    print("=" * 70)
    print("最终汇总")
    print("=" * 70)
    print(f"总数据集数:      {len(all_results)}")
    print(f"总Round数:       {total_rounds}")
    print(f"总数据对数:      {total_pairs}")
    print(f"唯一可见箱子:    {total_unique_visible}")
    print(f"唯一稳定箱子:    {total_unique_stable}")
    print(f"唯一不稳定箱子:  {total_unique_unstable}")
    print("=" * 70)

if __name__ == "__main__":
    main()
