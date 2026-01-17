"""
统计数据集的可见箱子、稳定/不稳定箱子数量
"""
import os
import json
from collections import defaultdict

VIEWS = ["top", "north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"]

def get_all_rounds(dataset_dir: str = "dataset"):
    rounds = []
    if not os.path.exists(dataset_dir):
        return rounds
    for name in sorted(os.listdir(dataset_dir)):
        if name.startswith("round_") and "_" in name:
            parts = name.rsplit("_", 1)
            if len(parts) == 2 and parts[1] in VIEWS:
                base_name = name.rsplit("_", 1)[0]
                if base_name not in rounds:
                    rounds.append(base_name)
    return sorted(rounds)

def count_unique_boxes(round_name: str):
    """统计一个round中唯一可见/稳定/不稳定的箱子"""
    unique_visible = set()
    unique_stable = set()
    unique_unstable = set()

    # 读取所有视角的 visible_boxes.json
    for view in VIEWS:
        view_dir = f"{round_name}_{view}"
        view_path = os.path.join("dataset", view_dir)
        visible_file = os.path.join(view_path, "visible_boxes.json")

        if os.path.exists(visible_file):
            with open(visible_file, "r") as f:
                boxes = json.load(f)
                for box in boxes:
                    unique_visible.add(box["prim_path"])

    # 读取所有视角的测试结果
    for view in VIEWS:
        view_dir = f"{round_name}_{view}"
        view_path = os.path.join("dataset", view_dir)
        removals_dir = os.path.join(view_path, "removals")

        if os.path.exists(removals_dir):
            for i in range(100):
                result_file = os.path.join(removals_dir, str(i), "result.json")
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

def main():
    rounds = get_all_rounds()
    if not rounds:
        print("未找到round数据")
        return

    print(f"找到 {len(rounds)} 个round\n")

    total_unique_visible = 0
    total_unique_stable = 0
    total_unique_unstable = 0
    total_rounds = len(rounds)

    for round_name in rounds:
        visible, stable, unstable = count_unique_boxes(round_name)
        total_unique_visible += visible
        total_unique_stable += stable
        total_unique_unstable += unstable

    # 打印最终汇总
    print("=" * 50)
    print("最终统计结果")
    print("=" * 50)
    print(f"总Round数:     {total_rounds}")
    print(f"唯一可见箱子:  {total_unique_visible}")
    print(f"唯一稳定箱子:  {total_unique_stable}")
    print(f"唯一不稳定箱子: {total_unique_unstable}")
    print("=" * 50)

if __name__ == "__main__":
    main()
