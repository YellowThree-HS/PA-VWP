"""稳定性检测模块"""

import numpy as np


class StabilityChecker:
    """箱子稳定性检测器"""

    def __init__(self, threshold: float = 0.02):
        """
        Args:
            threshold: 位移阈值（米），默认2cm
        """
        self.threshold = threshold
        self.positions_before = {}

    def snapshot_positions(self, box_gen):
        """保存当前所有箱子位置快照"""
        self.positions_before = box_gen.get_all_box_positions()

    def check(self, box_gen, removed_path: str) -> dict:
        """检测移除箱子后的稳定性

        Args:
            box_gen: BoxGenerator实例
            removed_path: 被移除箱子的路径

        Returns:
            稳定性检测结果字典
        """
        positions_after = box_gen.get_all_box_positions(
            exclude_paths=[removed_path]
        )

        moved_boxes = []
        for path, pos_before in self.positions_before.items():
            if path == removed_path or path not in positions_after:
                continue

            pos_after = positions_after[path]
            displacement = np.linalg.norm(pos_after - pos_before)

            if displacement > self.threshold:
                moved_boxes.append({
                    "path": path,
                    "name": path.split("/")[-1],
                    "displacement": float(displacement),
                    "pos_before": pos_before.tolist(),
                    "pos_after": pos_after.tolist()
                })

        is_stable = len(moved_boxes) == 0
        return {
            "is_stable": is_stable,
            "stability_label": 1 if is_stable else 0,
            "moved_boxes": moved_boxes,
            "total_boxes": len(positions_after)
        }

    def print_report(self, result: dict, removed_path: str):
        """打印稳定性检测报告"""
        print("\n" + "=" * 50)
        print("稳定性检测报告")
        print("=" * 50)

        removed_name = removed_path.split("/")[-1]
        print(f"被移除的箱子: {removed_name}")
        print(f"剩余箱子数量: {result['total_boxes']}")
        print(f"位移阈值: {self.threshold * 100:.1f} cm")
        print("-" * 50)

        if result["is_stable"]:
            print("结果: ✓ 稳定")
        else:
            print(f"结果: ✗ 不稳定")
            print(f"发生位移的箱子: {len(result['moved_boxes'])}")
            for box in result["moved_boxes"]:
                print(f"  - {box['name']}: {box['displacement']*100:.2f} cm")

        print("=" * 50 + "\n")
