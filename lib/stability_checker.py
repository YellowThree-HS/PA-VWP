"""稳定性检测模块"""

import numpy as np


class StabilityChecker:
    """箱子稳定性检测器"""

    def __init__(self, threshold: float = 0.02, velocity_threshold: float = 0.005):
        """
        Args:
            threshold: 位移阈值（米），默认2cm
            velocity_threshold: 速度收敛阈值（米/步），默认5mm/step
        """
        self.threshold = threshold
        self.velocity_threshold = velocity_threshold
        self.positions_before = {}
        # 动态收敛检测状态
        self._last_positions = {}
        self._stable_frames = 0

    def snapshot_positions(self, box_gen):
        """保存当前所有箱子位置快照"""
        self.positions_before = box_gen.get_all_box_positions()

    def reset_convergence(self):
        """重置收敛检测状态"""
        self._last_positions = {}
        self._stable_frames = 0

    def check_converged(self, box_gen, min_stable_frames: int = 10,
                        max_frames: int = 120, excluded_path: str = None) -> bool:
        """
        检测场景是否已收敛（所有箱子速度趋近于0）

        Args:
            box_gen: BoxGenerator实例
            min_stable_frames: 连续稳定帧数阈值
            max_frames: 最大等待帧数（超时保护）
            excluded_path: 要排除的箱子路径

        Returns:
            bool: 是否已收敛
        """
        exclude_paths = [excluded_path] if excluded_path else []
        current_positions = box_gen.get_all_box_positions(exclude_paths=exclude_paths)

        # 首次调用，初始化
        if not self._last_positions:
            self._last_positions = current_positions
            self._stable_frames = 0
            return False

        # 计算最大速度（位移/帧）
        max_velocity = 0.0
        for path, pos in current_positions.items():
            if path in self._last_positions:
                velocity = np.linalg.norm(pos - self._last_positions[path])
                max_velocity = max(max_velocity, velocity)

        # 更新位置记录
        self._last_positions = current_positions

        # 判断是否稳定
        if max_velocity < self.velocity_threshold:
            self._stable_frames += 1
        else:
            self._stable_frames = 0

        return self._stable_frames >= min_stable_frames

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
