"""
BoxWorld 数据集采集脚本
用于训练稳定性预测模型

流程：
1. 生成30-100个随机大小箱子
2. 稳定后保存初始RGB、depth、可见箱子掩码和ID
3. 逐个消失箱子，保存标注图（红绿掩码）和稳定性标签
4. 大循环结束后保存总结图（黄蓝掩码）
5. 重新生成一组箱子，循环采集
"""

import argparse

# 解析参数（必须在SimulationApp之前）
parser = argparse.ArgumentParser(description="BoxWorld Dataset Collector")
parser.add_argument("--rounds", type=int, default=10, help="Number of rounds")
parser.add_argument("--output", type=str, default="dataset", help="Output dir")
parser.add_argument("--headless", action="store_true", default=True, help="Run without visualization")
parser.add_argument("--gui", action="store_true", help="Run with visualization (override headless)")
args = parser.parse_args()

from isaacsim import SimulationApp

# 启动仿真应用
headless_mode = args.headless and not args.gui
simulation_app = SimulationApp({"headless": headless_mode})

import numpy as np
import cv2
import os
import json
from datetime import datetime
from isaacsim.core.api import World
from pxr import UsdGeom, Gf
import omni.usd
import omni.replicator.core as rep

import sys
sys.path.append("src")
from box_generator import BoxGenerator


class DatasetCollector:
    """数据集采集器"""

    def __init__(self, output_dir: str = "dataset"):
        self.world = None
        self.box_gen = None

        # 输出目录
        self.output_dir = output_dir
        self.current_round_dir = None

        # 状态机
        self.state = "INIT"
        self.step_count = 0

        # 箱子生成参数
        self.min_boxes = 30
        self.max_boxes = 100
        self.current_box_count = 0

        # 时序控制
        self.box_spawn_steps = []  # 分批生成的时机
        self.hold_duration = 120  # 等待稳定时间（2秒）
        self.hold_timer = 0
        self.post_removal_wait = 60  # 移除后等待时间（1秒）
        self.post_removal_timer = 0

        # 数据采集相关
        self.round_index = 0  # 当前大循环轮次
        self.box_paths = []  # 所有箱子路径
        self.visible_boxes_to_test = []  # 待测试的可见箱子
        self.current_test_index = 0  # 当前测试的箱子索引

        # 场景状态
        self.initial_scene_state = None
        self.positions_before_removal = {}
        self.removed_box_path = None
        self.stability_threshold = 0.02  # 2cm位移阈值

        # 图像数据
        self.initial_rgb_image = None
        self.initial_depth_image = None
        self.initial_mask_data = None
        self.initial_id_to_labels = None

        # 测试结果记录
        self.round_results = []  # 当前轮次的所有测试结果

        # Replicator
        self.render_product = None
        self.rgb_annotator = None
        self.depth_annotator = None
        self.seg_annotator = None

        # 检测已有轮次，从下一轮继续
        self._detect_existing_rounds()

    def _detect_existing_rounds(self):
        """检测已有轮次数量，设置起始轮次"""
        if not os.path.exists(self.output_dir):
            return

        max_round = 0
        for name in os.listdir(self.output_dir):
            if name.startswith("round_"):
                try:
                    # 格式: round_0001_63
                    round_num = int(name.split("_")[1])
                    max_round = max(max_round, round_num)
                except (IndexError, ValueError):
                    pass

        if max_round > 0:
            self.round_index = max_round
            print(f"Found existing rounds, will continue from round {max_round + 1}")

    def _create_output_dirs(self):
        """创建输出目录结构（在知道箱子数量后调用）"""
        os.makedirs(self.output_dir, exist_ok=True)

        # 创建当前轮次目录，包含箱子数量
        self.current_round_dir = os.path.join(
            self.output_dir,
            f"round_{self.round_index:04d}_{self.current_box_count}"
        )
        os.makedirs(self.current_round_dir, exist_ok=True)
        os.makedirs(os.path.join(self.current_round_dir, "initial"), exist_ok=True)
        os.makedirs(os.path.join(self.current_round_dir, "removals"), exist_ok=True)

        print(f"Output directory: {self.current_round_dir}")

    def _create_cameras(self):
        """创建相机和annotators"""
        stage = omni.usd.get_context().get_stage()

        # 顶部相机
        top_cam_path = "/World/TopCamera"
        top_cam = UsdGeom.Camera.Define(stage, top_cam_path)
        top_xform = UsdGeom.Xformable(top_cam.GetPrim())
        top_xform.AddTranslateOp().Set(Gf.Vec3d(0.45, 0.0, 1.5))
        top_xform.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 180))
        top_cam.GetFocalLengthAttr().Set(18.0)

        # 设置Replicator
        self.render_product = rep.create.render_product(top_cam_path, (640, 480))

        # RGB
        self.rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb_annotator.attach([self.render_product])

        # Depth
        self.depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
        self.depth_annotator.attach([self.render_product])

        # Instance Segmentation
        self.seg_annotator = rep.AnnotatorRegistry.get_annotator("instance_segmentation")
        self.seg_annotator.attach([self.render_product])

        print("Cameras and annotators created")

    def setup_scene(self):
        """设置仿真场景"""
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        self.box_gen = BoxGenerator()
        self._create_cameras()
        print("Scene setup complete")

    def _calculate_spawn_steps(self, total_count: int) -> list:
        """计算分批生成箱子的时机"""
        # 分3批生成
        batch_size = total_count // 3
        remainder = total_count % 3

        batches = [batch_size, batch_size, batch_size]
        for i in range(remainder):
            batches[i] += 1

        return [
            (20, batches[0]),    # 第20步生成第一批
            (80, batches[1]),    # 第80步生成第二批
            (140, batches[2])    # 第140步生成第三批
        ]

    def spawn_boxes(self, count: int):
        """生成指定数量的箱子"""
        paths = self.box_gen.create_random_boxes(
            count=count,
            center=[0.45, 0.0],
            spread=0.15,
            drop_height=0.30,
            size_range=((0.06, 0.12), (0.06, 0.12), (0.06, 0.12)),
            mass_range=(0.3, 1.5)
        )
        self.box_paths.extend(paths)
        print(f"Spawned {count} boxes, total: {len(self.box_paths)}")

    def start_new_round(self):
        """开始新一轮数据采集"""
        self.round_index += 1

        # 随机箱子数量（必须在创建目录之前）
        self.current_box_count = np.random.randint(self.min_boxes, self.max_boxes + 1)

        # 创建输出目录（目录名包含箱子数量）
        self._create_output_dirs()

        print(f"\n{'='*60}")
        print(f"Round {self.round_index}: Generating {self.current_box_count} boxes")
        print(f"{'='*60}\n")

        # 计算分批生成时机
        self.box_spawn_steps = self._calculate_spawn_steps(self.current_box_count)

        # 重置状态
        self.box_paths = []
        self.visible_boxes_to_test = []
        self.current_test_index = 0
        self.round_results = []
        self.initial_scene_state = None
        self.initial_rgb_image = None
        self.initial_depth_image = None
        self.initial_mask_data = None
        self.hold_timer = 0
        self.step_count = 0
        self.state = "SPAWNING"

    def clear_scene(self):
        """清除场景中的所有箱子"""
        for path in list(self.box_paths):
            self.box_gen.delete_box(path)
        self.box_paths = []
        self.box_gen._boxes = {}
        self.box_gen._box_count = 0
        print("Scene cleared")

    def get_visible_boxes(self, min_visible_ratio: float = 0.20) -> list:
        """获取所有可见箱子，过滤遮挡太多的"""
        seg_data = self.seg_annotator.get_data()
        if seg_data is None:
            return []

        id_to_labels = seg_data.get("info", {}).get("idToLabels", {})
        mask = seg_data["data"]
        unique_ids = np.unique(mask)

        # 统计每个箱子的像素数
        box_pixels = {}
        for uid in unique_ids:
            if uid == 0:
                continue
            label_info = id_to_labels.get(str(uid), "")
            label_str = str(label_info)
            if "/World/box_" in label_str:
                box_pixels[uid] = np.sum(mask == uid)

        if not box_pixels:
            return []

        max_pixels = max(box_pixels.values())
        visible_boxes = []
        added_names = set()

        for uid, pixel_count in box_pixels.items():
            visible_ratio = pixel_count / max_pixels
            if visible_ratio < min_visible_ratio:
                continue

            label_info = id_to_labels.get(str(uid), {})
            label_str = str(label_info)

            for name, info in self.box_gen.boxes.items():
                if info["prim_path"] == label_str and name not in added_names:
                    visible_boxes.append({
                        "name": name,
                        "prim_path": info["prim_path"],
                        "uid": uid,
                        "pixel_count": pixel_count
                    })
                    added_names.add(name)
                    break

        print(f"Visible boxes: {len(visible_boxes)}")
        return visible_boxes

    def capture_initial_state(self):
        """捕获并保存初始状态（RGB、depth、掩码、可见箱子ID）"""
        initial_dir = os.path.join(self.current_round_dir, "initial")

        # 获取RGB图像
        rgb_data = self.rgb_annotator.get_data()
        if rgb_data is not None:
            self.initial_rgb_image = rgb_data[:, :, :3].copy()
            rgb_bgr = cv2.cvtColor(self.initial_rgb_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(initial_dir, "rgb.png"), rgb_bgr)

        # 获取深度图
        depth_data = self.depth_annotator.get_data()
        if depth_data is not None:
            self.initial_depth_image = depth_data.copy()
            # 归一化保存为可视化图像
            depth_vis = self.initial_depth_image.copy()
            depth_vis = np.clip(depth_vis, 0, 3)  # 截断到3米
            depth_vis = (depth_vis / 3.0 * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(initial_dir, "depth.png"), depth_vis)
            # 保存原始深度数据
            np.save(os.path.join(initial_dir, "depth.npy"), self.initial_depth_image)

        # 获取分割掩码
        seg_data = self.seg_annotator.get_data()
        if seg_data is not None:
            self.initial_mask_data = seg_data["data"].copy()
            self.initial_id_to_labels = seg_data.get("info", {}).get("idToLabels", {})
            # 保存掩码
            np.save(os.path.join(initial_dir, "mask.npy"), self.initial_mask_data)

        # 保存可见箱子信息
        visible_info = []
        for box in self.visible_boxes_to_test:
            visible_info.append({
                "name": box["name"],
                "prim_path": box["prim_path"],
                "uid": int(box["uid"]),
                "pixel_count": int(box["pixel_count"])
            })

        with open(os.path.join(initial_dir, "visible_boxes.json"), "w") as f:
            json.dump(visible_info, f, indent=2)

        print(f"Initial state saved to {initial_dir}")

    def check_stability(self) -> dict:
        """检测移除箱子后的稳定性"""
        positions_after = self.box_gen.get_all_box_positions(
            exclude_paths=[self.removed_box_path]
        )

        moved_boxes = []
        for path, pos_before in self.positions_before_removal.items():
            if path == self.removed_box_path:
                continue
            if path not in positions_after:
                continue

            pos_after = positions_after[path]
            displacement = np.linalg.norm(pos_after - pos_before)

            if displacement > self.stability_threshold:
                moved_boxes.append({
                    "path": path,
                    "name": path.split("/")[-1],
                    "displacement": float(displacement)
                })

        is_stable = len(moved_boxes) == 0
        return {
            "is_stable": is_stable,
            "stability_label": 1 if is_stable else 0,
            "moved_boxes": moved_boxes
        }

    def _get_box_mask_id(self, prim_path: str) -> int:
        """根据prim_path获取掩码中的ID"""
        for uid, label_info in self.initial_id_to_labels.items():
            if prim_path in str(label_info):
                return int(uid)
        return None

    def save_removal_result(self, test_index: int, removed_box: dict, stability_result: dict):
        """保存单次移除的结果

        保存内容：
        1. 半透明原图+红绿掩码标注图（绿=消失箱子，红=受影响箱子）
        2. 消失箱子的单独掩码
        3. 稳定性标签JSON
        """
        box_name = removed_box["name"]
        # 每个box单独一个文件夹，用纯数字命名方便PyTorch Dataset的get_item
        box_dir = os.path.join(
            self.current_round_dir, "removals", str(test_index)
        )
        os.makedirs(box_dir, exist_ok=True)

        # 1. 创建红绿掩码标注图
        if self.initial_rgb_image is not None and self.initial_mask_data is not None:
            rgb_bgr = cv2.cvtColor(self.initial_rgb_image, cv2.COLOR_RGB2BGR)
            overlay = rgb_bgr.copy()
            alpha = 0.4

            # 消失的箱子标绿色
            removed_id = self._get_box_mask_id(removed_box["prim_path"])
            if removed_id is not None:
                removed_mask = self.initial_mask_data == removed_id
                overlay[removed_mask] = [0, 255, 0]  # BGR绿色

            # 受影响的箱子标红色
            for moved_box in stability_result["moved_boxes"]:
                affected_id = self._get_box_mask_id(moved_box["path"])
                if affected_id is not None:
                    affected_mask = self.initial_mask_data == affected_id
                    overlay[affected_mask] = [0, 0, 255]  # BGR红色

            # 混合
            result = cv2.addWeighted(rgb_bgr, alpha, overlay, 1 - alpha, 0)
            cv2.imwrite(os.path.join(box_dir, "annotated.png"), result)

            # 2. 保存消失箱子的单独掩码
            if removed_id is not None:
                removed_mask_img = (self.initial_mask_data == removed_id).astype(np.uint8) * 255
                cv2.imwrite(os.path.join(box_dir, "mask.png"), removed_mask_img)

            # 3. 保存受影响箱子的掩码
            affected_mask_combined = np.zeros_like(self.initial_mask_data, dtype=np.uint8)
            for moved_box in stability_result["moved_boxes"]:
                affected_id = self._get_box_mask_id(moved_box["path"])
                if affected_id is not None:
                    affected_mask_combined[self.initial_mask_data == affected_id] = 255
            cv2.imwrite(os.path.join(box_dir, "affected_mask.png"), affected_mask_combined)

        # 3. 保存稳定性标签JSON
        result_data = {
            "removed_box": {
                "name": box_name,
                "prim_path": removed_box["prim_path"],
                "uid": int(removed_box["uid"])
            },
            "is_stable": stability_result["is_stable"],
            "stability_label": stability_result["stability_label"],
            "affected_boxes": stability_result["moved_boxes"]
        }

        with open(os.path.join(box_dir, "result.json"), "w") as f:
            json.dump(result_data, f, indent=2)

        status = "稳定" if stability_result["is_stable"] else "不稳定"
        print(f"  [{test_index}] {box_name}: {status}")

    def save_round_summary(self):
        """保存大循环总结图

        黄色掩码：移除后稳定的箱子
        蓝色掩码：移除后会导致坍塌的箱子
        """
        if self.initial_rgb_image is None or self.initial_mask_data is None:
            return

        rgb_bgr = cv2.cvtColor(self.initial_rgb_image, cv2.COLOR_RGB2BGR)
        overlay = rgb_bgr.copy()
        alpha = 0.4

        stable_boxes = []
        unstable_boxes = []

        for result in self.round_results:
            box_info = result["removed_box"]
            if result["is_stable"]:
                stable_boxes.append(box_info)
            else:
                unstable_boxes.append(box_info)

        # 稳定的箱子标黄色
        for box in stable_boxes:
            box_id = self._get_box_mask_id(box["prim_path"])
            if box_id is not None:
                mask = self.initial_mask_data == box_id
                overlay[mask] = [0, 255, 255]  # BGR黄色

        # 不稳定的箱子标蓝色
        for box in unstable_boxes:
            box_id = self._get_box_mask_id(box["prim_path"])
            if box_id is not None:
                mask = self.initial_mask_data == box_id
                overlay[mask] = [255, 0, 0]  # BGR蓝色

        result = cv2.addWeighted(rgb_bgr, alpha, overlay, 1 - alpha, 0)
        save_path = os.path.join(self.current_round_dir, "summary.png")
        cv2.imwrite(save_path, result)

        # 保存总结JSON
        summary_data = {
            "total_boxes": self.current_box_count,
            "visible_boxes_tested": len(self.round_results),
            "stable_count": len(stable_boxes),
            "unstable_count": len(unstable_boxes),
            "stable_boxes": [b["name"] for b in stable_boxes],
            "unstable_boxes": [b["name"] for b in unstable_boxes]
        }

        with open(os.path.join(self.current_round_dir, "summary.json"), "w") as f:
            json.dump(summary_data, f, indent=2)

        print(f"\nRound summary saved: {len(stable_boxes)} stable, {len(unstable_boxes)} unstable")

    def run(self, num_rounds: int = 10):
        """运行数据采集

        Args:
            num_rounds: 采集轮数
        """
        self.setup_scene()
        self.world.reset()

        print(f"Starting dataset collection for {num_rounds} rounds...")

        # 开始第一轮
        self.start_new_round()
        spawn_batch_index = 0

        while simulation_app.is_running() and self.round_index <= num_rounds:
            self.world.step(render=True)
            self.step_count += 1

            # 状态机
            if self.state == "SPAWNING":
                self._handle_spawning_state(spawn_batch_index)
                # 检查是否完成所有批次
                if spawn_batch_index < len(self.box_spawn_steps):
                    step, count = self.box_spawn_steps[spawn_batch_index]
                    if self.step_count == step:
                        self.spawn_boxes(count)
                        spawn_batch_index += 1
                        if spawn_batch_index >= len(self.box_spawn_steps):
                            self.state = "STABILIZING"
                            self.hold_timer = 0
                            print("All boxes spawned, waiting for stabilization...")

            elif self.state == "STABILIZING":
                self._handle_stabilizing_state()

            elif self.state == "TESTING":
                self._handle_testing_state()

            elif self.state == "WAITING_STABILITY":
                self._handle_waiting_stability_state()

            elif self.state == "RESTORING":
                self._handle_restoring_state()

            elif self.state == "ROUND_COMPLETE":
                self._handle_round_complete_state(num_rounds)
                spawn_batch_index = 0

        simulation_app.close()

    def _handle_spawning_state(self, spawn_batch_index):
        """处理箱子生成状态"""
        pass  # 生成逻辑在run()中处理

    def _handle_stabilizing_state(self):
        """处理等待稳定状态"""
        self.hold_timer += 1
        if self.hold_timer >= self.hold_duration:
            # 保存初始场景状态
            self.initial_scene_state = self.box_gen.save_scene_state()
            # 获取可见箱子
            self.visible_boxes_to_test = self.get_visible_boxes()

            if not self.visible_boxes_to_test:
                print("No visible boxes, skipping this round")
                self.state = "ROUND_COMPLETE"
                return

            # 保存初始状态
            self.capture_initial_state()
            print(f"Starting tests on {len(self.visible_boxes_to_test)} visible boxes")

            # 开始测试
            self.current_test_index = 0
            self.state = "TESTING"

    def _handle_testing_state(self):
        """处理测试状态 - 移除一个箱子"""
        if self.current_test_index >= len(self.visible_boxes_to_test):
            # 所有箱子测试完成
            self.state = "ROUND_COMPLETE"
            return

        # 选择当前要测试的箱子
        current_box = self.visible_boxes_to_test[self.current_test_index]
        self.removed_box_path = current_box["prim_path"]

        # 记录移除前位置
        self.positions_before_removal = self.box_gen.get_all_box_positions()

        # 删除箱子
        self.box_gen.delete_box(self.removed_box_path)

        # 进入等待稳定状态
        self.post_removal_timer = 0
        self.state = "WAITING_STABILITY"

    def _handle_waiting_stability_state(self):
        """处理等待稳定状态 - 检测稳定性并保存结果"""
        self.post_removal_timer += 1
        if self.post_removal_timer >= self.post_removal_wait:
            # 检测稳定性
            stability_result = self.check_stability()

            # 获取当前测试的箱子信息
            current_box = self.visible_boxes_to_test[self.current_test_index]

            # 保存结果
            self.save_removal_result(
                self.current_test_index,
                current_box,
                stability_result
            )

            # 记录结果
            self.round_results.append({
                "removed_box": current_box,
                "is_stable": stability_result["is_stable"],
                "affected_boxes": stability_result["moved_boxes"]
            })

            # 恢复场景准备下一次测试
            self.current_test_index += 1
            if self.current_test_index < len(self.visible_boxes_to_test):
                self.box_gen.restore_scene_state(self.initial_scene_state)
                self.hold_timer = 0
                self.state = "RESTORING"
            else:
                self.state = "ROUND_COMPLETE"

    def _handle_restoring_state(self):
        """处理场景恢复状态"""
        self.hold_timer += 1
        if self.hold_timer >= 60:  # 等待1秒让场景稳定
            self.state = "TESTING"

    def _handle_round_complete_state(self, num_rounds: int):
        """处理轮次完成状态"""
        # 保存总结图
        self.save_round_summary()

        print(f"\n{'='*60}")
        print(f"Round {self.round_index} complete!")
        print(f"{'='*60}\n")

        # 检查是否继续下一轮
        if self.round_index < num_rounds:
            # 清除场景，开始新一轮
            self.clear_scene()
            self.start_new_round()
        else:
            print("All rounds complete!")
            self.state = "DONE"


if __name__ == "__main__":
    collector = DatasetCollector(output_dir=args.output)
    collector.run(num_rounds=args.rounds)
