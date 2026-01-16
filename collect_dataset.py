"""
BoxWorld 数据集采集脚本
无头模式批量采集训练数据
"""

import argparse

# 解析参数（必须在SimulationApp之前）
parser = argparse.ArgumentParser(description="BoxWorld Dataset Collector")
parser.add_argument("--rounds", type=int, default=10, help="Number of rounds")
parser.add_argument("--output", type=str, default="dataset", help="Output dir")
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--gui", action="store_true", help="Run with GUI")
args = parser.parse_args()

from isaacsim import SimulationApp

headless_mode = args.headless and not args.gui
simulation_app = SimulationApp({"headless": headless_mode})

import numpy as np
import cv2
import os
import json
from isaacsim.core.api import World

import sys
sys.path.append("src")
sys.path.append(".")
from box_generator import BoxGenerator
from lib import CameraManager, StabilityChecker, ImageUtils


class DatasetCollector:
    """数据集采集器（无头模式）"""

    def __init__(self, output_dir: str = "dataset"):
        self.world = None
        self.box_gen = None
        self.camera = CameraManager()
        self.stability_checker = StabilityChecker(threshold=0.02)

        self.output_dir = output_dir
        self.current_round_dir = None

        # 状态机
        self.state = "INIT"
        self.step_count = 0

        # 箱子参数
        self.min_boxes = 30
        self.max_boxes = 100
        self.current_box_count = 0
        self.box_spawn_steps = []
        self.box_paths = []

        # 时序控制
        self.hold_duration = 120
        self.hold_timer = 0
        self.post_removal_wait = 60
        self.post_removal_timer = 0

        # 采集相关
        self.round_index = 0
        self.visible_boxes_to_test = []
        self.current_test_index = 0
        self.removed_box_path = None

        # 场景状态
        self.initial_scene_state = None
        self.round_results = []

        # 图像数据
        self.initial_rgb = None
        self.initial_depth = None
        self.initial_mask = None
        self.initial_id_to_labels = None

        self._detect_existing_rounds()

    def _detect_existing_rounds(self):
        """检测已有轮次"""
        if not os.path.exists(self.output_dir):
            return
        max_round = 0
        for name in os.listdir(self.output_dir):
            if name.startswith("round_"):
                try:
                    round_num = int(name.split("_")[1])
                    max_round = max(max_round, round_num)
                except (IndexError, ValueError):
                    pass
        if max_round > 0:
            self.round_index = max_round
            print(f"Continue from round {max_round + 1}")

    def setup_scene(self):
        """设置场景"""
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        self.box_gen = BoxGenerator()
        self.camera.create_top_camera(with_depth=True)
        print("Scene setup complete")

    def start_new_round(self):
        """开始新一轮"""
        self.round_index += 1
        self.current_box_count = np.random.randint(self.min_boxes, self.max_boxes + 1)

        # 创建目录
        self.current_round_dir = os.path.join(
            self.output_dir, f"round_{self.round_index:04d}_{self.current_box_count}"
        )
        os.makedirs(os.path.join(self.current_round_dir, "initial"), exist_ok=True)
        os.makedirs(os.path.join(self.current_round_dir, "removals"), exist_ok=True)

        print(f"\nRound {self.round_index}: {self.current_box_count} boxes")

        # 计算分批生成
        batch_size = self.current_box_count // 3
        remainder = self.current_box_count % 3
        batches = [batch_size + (1 if i < remainder else 0) for i in range(3)]
        self.box_spawn_steps = [(20, batches[0]), (80, batches[1]), (140, batches[2])]

        # 重置状态
        self.box_paths = []
        self.visible_boxes_to_test = []
        self.current_test_index = 0
        self.round_results = []
        self.initial_scene_state = None
        self.hold_timer = 0
        self.step_count = 0
        self.state = "SPAWNING"

    def spawn_boxes(self, count: int):
        """生成箱子"""
        paths = self.box_gen.create_random_boxes(
            count=count,
            center=[0.45, 0.0],
            spread=0.15,
            drop_height=0.30,
            size_range=((0.06, 0.12), (0.06, 0.12), (0.06, 0.12)),
            mass_range=(0.3, 1.5)
        )
        self.box_paths.extend(paths)
        print(f"Spawned {count}, total: {len(self.box_paths)}")

    def clear_scene(self):
        """清除场景"""
        for path in list(self.box_paths):
            self.box_gen.delete_box(path)
        self.box_paths = []
        self.box_gen._boxes = {}
        self.box_gen._box_count = 0

    def capture_initial_state(self):
        """捕获初始状态"""
        initial_dir = os.path.join(self.current_round_dir, "initial")

        self.initial_rgb = self.camera.get_rgb()
        if self.initial_rgb is not None:
            ImageUtils.save_rgb(self.initial_rgb, os.path.join(initial_dir, "rgb.png"))

        self.initial_depth = self.camera.get_depth()
        if self.initial_depth is not None:
            ImageUtils.save_depth(self.initial_depth, os.path.join(initial_dir, "depth.png"))
            np.save(os.path.join(initial_dir, "depth.npy"), self.initial_depth)

        self.initial_mask, self.initial_id_to_labels = self.camera.get_segmentation()
        if self.initial_mask is not None:
            np.save(os.path.join(initial_dir, "mask.npy"), self.initial_mask)

        # 保存可见箱子信息
        visible_info = [{"name": b["name"], "prim_path": b["prim_path"],
                         "uid": int(b["uid"])} for b in self.visible_boxes_to_test]
        with open(os.path.join(initial_dir, "visible_boxes.json"), "w") as f:
            json.dump(visible_info, f, indent=2)

    def save_removal_result(self, test_index: int, box: dict, result: dict):
        """保存移除结果"""
        box_dir = os.path.join(self.current_round_dir, "removals", str(test_index))
        os.makedirs(box_dir, exist_ok=True)

        # 标注图
        if self.initial_rgb is not None and self.initial_mask is not None:
            annotated = ImageUtils.create_annotated_image(
                self.initial_rgb, self.initial_mask,
                self.initial_id_to_labels, box["prim_path"], result["moved_boxes"]
            )
            cv2.imwrite(os.path.join(box_dir, "annotated.png"), annotated)

            # 消失箱子掩码
            removed_id = ImageUtils.get_mask_id(box["prim_path"], self.initial_id_to_labels)
            if removed_id is not None:
                mask_img = (self.initial_mask == removed_id).astype(np.uint8) * 255
                cv2.imwrite(os.path.join(box_dir, "mask.png"), mask_img)

        # 保存结果JSON
        result_data = {
            "removed_box": {"name": box["name"], "prim_path": box["prim_path"], "uid": int(box["uid"])},
            "is_stable": result["is_stable"],
            "stability_label": result["stability_label"],
            "affected_boxes": result["moved_boxes"]
        }
        with open(os.path.join(box_dir, "result.json"), "w") as f:
            json.dump(result_data, f, indent=2)

        status = "稳定" if result["is_stable"] else "不稳定"
        print(f"  [{test_index}] {box['name']}: {status}")

    def save_round_summary(self):
        """保存轮次总结"""
        stable = [r["removed_box"] for r in self.round_results if r["is_stable"]]
        unstable = [r["removed_box"] for r in self.round_results if not r["is_stable"]]

        # 总结图
        if self.initial_rgb is not None and self.initial_mask is not None:
            summary = ImageUtils.create_summary_image(
                self.initial_rgb, self.initial_mask,
                self.initial_id_to_labels, stable, unstable
            )
            cv2.imwrite(os.path.join(self.current_round_dir, "summary.png"), summary)

        # 总结JSON
        summary_data = {
            "total_boxes": self.current_box_count,
            "tested": len(self.round_results),
            "stable_count": len(stable),
            "unstable_count": len(unstable)
        }
        with open(os.path.join(self.current_round_dir, "summary.json"), "w") as f:
            json.dump(summary_data, f, indent=2)
        print(f"Round summary: {len(stable)} stable, {len(unstable)} unstable")

    def run(self, num_rounds: int = 10):
        """运行采集"""
        self.setup_scene()
        self.world.reset()
        print(f"Starting collection for {num_rounds} rounds...")

        self.start_new_round()
        spawn_batch_index = 0

        while simulation_app.is_running() and self.round_index <= num_rounds:
            self.world.step(render=True)
            self.step_count += 1

            if self.state == "SPAWNING":
                if spawn_batch_index < len(self.box_spawn_steps):
                    step, count = self.box_spawn_steps[spawn_batch_index]
                    if self.step_count == step:
                        self.spawn_boxes(count)
                        spawn_batch_index += 1
                        if spawn_batch_index >= len(self.box_spawn_steps):
                            self.state = "STABILIZING"
                            self.hold_timer = 0

            elif self.state == "STABILIZING":
                self.hold_timer += 1
                if self.hold_timer >= self.hold_duration:
                    self.initial_scene_state = self.box_gen.save_scene_state()
                    self.visible_boxes_to_test = self.camera.get_visible_boxes(self.box_gen)
                    if not self.visible_boxes_to_test:
                        self.state = "ROUND_COMPLETE"
                    else:
                        self.capture_initial_state()
                        self.current_test_index = 0
                        self.state = "TESTING"

            elif self.state == "TESTING":
                if self.current_test_index >= len(self.visible_boxes_to_test):
                    self.state = "ROUND_COMPLETE"
                else:
                    box = self.visible_boxes_to_test[self.current_test_index]
                    self.removed_box_path = box["prim_path"]
                    self.stability_checker.snapshot_positions(self.box_gen)
                    self.box_gen.delete_box(self.removed_box_path)
                    self.post_removal_timer = 0
                    self.state = "WAITING_STABILITY"

            elif self.state == "WAITING_STABILITY":
                self.post_removal_timer += 1
                if self.post_removal_timer >= self.post_removal_wait:
                    result = self.stability_checker.check(self.box_gen, self.removed_box_path)
                    box = self.visible_boxes_to_test[self.current_test_index]
                    self.save_removal_result(self.current_test_index, box, result)
                    self.round_results.append({"removed_box": box, "is_stable": result["is_stable"]})

                    self.current_test_index += 1
                    if self.current_test_index < len(self.visible_boxes_to_test):
                        self.box_gen.restore_scene_state(self.initial_scene_state)
                        self.hold_timer = 0
                        self.state = "RESTORING"
                    else:
                        self.state = "ROUND_COMPLETE"

            elif self.state == "RESTORING":
                self.hold_timer += 1
                if self.hold_timer >= 60:
                    self.state = "TESTING"

            elif self.state == "ROUND_COMPLETE":
                self.save_round_summary()
                if self.round_index < num_rounds:
                    self.clear_scene()
                    self.start_new_round()
                    spawn_batch_index = 0
                else:
                    print("All rounds complete!")
                    break

        simulation_app.close()


if __name__ == "__main__":
    collector = DatasetCollector(output_dir=args.output)
    collector.run(num_rounds=args.rounds)
