"""
BoxWorld MVP - 可视化仿真脚本
纸箱堆叠稳定性测试（带GUI）
"""

from isaacsim import SimulationApp

# 启动仿真应用（可视化模式）
simulation_app = SimulationApp({"headless": False})

import numpy as np
import cv2
import os
from isaacsim.core.api import World

# 导入自定义模块
import sys
sys.path.append("src")
sys.path.append(".")
from box_generator import BoxGenerator
from lib import SceneBuilder, CameraManager, StabilityChecker, ImageUtils


class BoxPickSimulation:
    """纸箱堆叠稳定性测试仿真（可视化模式）"""

    def __init__(self):
        self.world = None
        self.box_gen = None
        self.scene_builder = SceneBuilder()
        self.camera = CameraManager()
        self.stability_checker = StabilityChecker(threshold=0.02)

        # 状态机
        self.state = "INIT"
        self.step_count = 0
        self.box_spawn_steps = [20, 80, 140]  # 分三批生成
        self.hold_duration = 5 * 60  # 保持5秒
        self.hold_timer = 0
        self.post_removal_wait = 30  # 移除后等待0.5秒

        # 箱子相关
        self.box_paths = []
        self.removed_box_path = None
        self.removal_done = False
        self.post_removal_timer = 0

        # 循环测试相关
        self.initial_scene_state = None
        self.test_iteration = 0
        self.visible_boxes_to_test = []
        self.total_boxes_to_test = 0
        self.test_results = []

        # 图像数据
        self.initial_rgb_image = None
        self.initial_mask_data = None
        self.initial_id_to_labels = None

    def setup_scene(self):
        """设置仿真场景"""
        self.world = World(stage_units_in_meters=1.0)
        self.scene_builder.create_textured_floor(size=5.0)
        self.box_gen = BoxGenerator(texture_dir="assets/cardboard_textures_processed")
        self.camera.create_top_camera()
        print("Scene setup complete!")

    def spawn_box(self, batch: int = 1):
        """分批生成箱子"""
        batch_counts = {1: 34, 2: 33, 3: 33}
        count = batch_counts.get(batch, 34)
        paths = self.box_gen.create_random_boxes(
            count=count,
            center=[0.45, 0.0],
            spread=0.15,
            drop_height=0.30,
            size_range=((0.06, 0.12), (0.06, 0.12), (0.06, 0.12)),
            mass_range=(0.3, 1.5)
        )
        self.box_paths.extend(paths)
        print(f"Batch {batch}: Spawned {count} boxes, total: {len(self.box_paths)}")

    def capture_initial_state(self):
        """捕获初始状态"""
        os.makedirs("result", exist_ok=True)
        self.initial_rgb_image = self.camera.get_rgb()
        if self.initial_rgb_image is not None:
            ImageUtils.save_rgb(self.initial_rgb_image, "result/initial_rgb.png")
        self.initial_mask_data, self.initial_id_to_labels = self.camera.get_segmentation()

    def save_annotated_result(self, iteration: int, removed_path: str, affected_boxes: list):
        """保存标注结果图"""
        if self.initial_rgb_image is None or self.initial_mask_data is None:
            return
        result = ImageUtils.create_annotated_image(
            self.initial_rgb_image, self.initial_mask_data,
            self.initial_id_to_labels, removed_path, affected_boxes
        )
        removed_name = removed_path.split("/")[-1]
        cv2.imwrite(f"result/iter{iteration:02d}_{removed_name}.png", result)

    def print_final_summary(self):
        """打印测试总结"""
        print("\n" + "=" * 60)
        print("最终测试总结")
        print("=" * 60)
        stable_count = sum(1 for r in self.test_results if r["is_stable"])
        print(f"总测试: {len(self.test_results)}, 稳定: {stable_count}, 不稳定: {len(self.test_results) - stable_count}")
        print("=" * 60 + "\n")

    def run(self):
        """运行仿真"""
        self.setup_scene()
        self.world.reset()
        print("Starting simulation...")

        while simulation_app.is_running():
            self.world.step(render=True)
            self.step_count += 1
            self._update_state_machine()

        simulation_app.close()

    def _update_state_machine(self):
        """状态机更新"""
        if self.state == "INIT":
            self._handle_init_state()
        elif self.state == "HOLD":
            self._handle_hold_state()
        elif self.state == "REMOVE_BOX":
            self._handle_remove_state()
        elif self.state == "RESTORE_WAIT":
            self._handle_restore_state()
        elif self.state == "DONE":
            self._handle_done_state()

    def _handle_init_state(self):
        """处理初始化状态 - 分批生成箱子"""
        if self.step_count == self.box_spawn_steps[0]:
            self.spawn_box(batch=1)
        elif self.step_count == self.box_spawn_steps[1]:
            self.spawn_box(batch=2)
        elif self.step_count == self.box_spawn_steps[2]:
            self.spawn_box(batch=3)
            self.state = "HOLD"
            self.hold_timer = 0
            print("All boxes dropped, waiting...")

    def _handle_hold_state(self):
        """处理等待状态"""
        self.hold_timer += 1
        if self.hold_timer == 120:
            if self.initial_scene_state is None:
                self.initial_scene_state = self.box_gen.save_scene_state()
                self.capture_initial_state()
                self.visible_boxes_to_test = self.camera.get_visible_boxes(self.box_gen)
                self.total_boxes_to_test = len(self.visible_boxes_to_test)
                print(f"开始测试 {self.total_boxes_to_test} 个可见箱子")

        if self.hold_timer == 150:
            if not self.visible_boxes_to_test:
                self.print_final_summary()
                self.state = "DONE"
            else:
                self._start_next_test()

    def _start_next_test(self):
        """开始下一个测试"""
        self.test_iteration += 1
        box = self.visible_boxes_to_test.pop(0)
        self.removed_box_path = box["prim_path"]
        self.stability_checker.snapshot_positions(self.box_gen)
        self.state = "REMOVE_BOX"
        self.removal_done = False
        print(f"测试 {self.test_iteration}/{self.total_boxes_to_test}: {box['name']}")

    def _handle_remove_state(self):
        """处理移除箱子状态"""
        if not self.removal_done:
            self.box_gen.delete_box(self.removed_box_path)
            self.removal_done = True
            self.post_removal_timer = 0
        else:
            self.post_removal_timer += 1
            if self.post_removal_timer >= self.post_removal_wait:
                self._finish_test()

    def _finish_test(self):
        """完成当前测试"""
        result = self.stability_checker.check(self.box_gen, self.removed_box_path)
        self.stability_checker.print_report(result, self.removed_box_path)
        result["removed_box"] = self.removed_box_path.split("/")[-1]
        self.test_results.append(result)
        self.save_annotated_result(self.test_iteration, self.removed_box_path, result["moved_boxes"])

        if self.visible_boxes_to_test:
            self.box_gen.restore_scene_state(self.initial_scene_state)
            self.hold_timer = 0
            self.state = "RESTORE_WAIT"
        else:
            self.print_final_summary()
            self.state = "DONE"

    def _handle_restore_state(self):
        """处理场景恢复状态"""
        self.hold_timer += 1
        if self.hold_timer >= 60:
            self.hold_timer = 0
            self.state = "HOLD"

    def _handle_done_state(self):
        """处理完成状态"""
        self.hold_timer += 1
        if self.hold_timer >= self.hold_duration:
            print("Simulation complete")


if __name__ == "__main__":
    sim = BoxPickSimulation()
    sim.run()
