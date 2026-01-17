"""
BoxWorld 数据集采集脚本
无头模式批量采集训练数据
"""

import argparse

# 解析参数（必须在SimulationApp之前）
parser = argparse.ArgumentParser(description="BoxWorld Dataset Collector")
parser.add_argument("--rounds", type=int, default=100, help="Number of rounds")
parser.add_argument("--output", type=str, default="dataset", help="Output dir")
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--gui", action="store_true", help="Run with GUI")
parser.add_argument("--mode", type=str, default="messy", choices=["messy", "neat"],
                    help="Dataset mode: 'messy' (random drop) or 'neat' (stacked)")
parser.add_argument("--min-boxes", type=int, default=30, help="Min boxes per round")
parser.add_argument("--max-boxes", type=int, default=100, help="Max boxes per round")
args = parser.parse_args()

from isaacsim import SimulationApp

headless_mode = args.headless and not args.gui
simulation_app = SimulationApp({"headless": headless_mode})

import numpy as np
import cv2
import os
import json
import time
from isaacsim.core.api import World
import omni.usd
import omni.replicator.core as rep
from pxr import UsdLux, UsdGeom, Gf

import sys
sys.path.append("src")
sys.path.append(".")
from box_generator import BoxGenerator
from lib import CameraManager, StabilityChecker, ImageUtils, SceneBuilder

# 九相机配置：八个方向 + 正上方
# vertical_flip: 是否垂直翻转图像（解决某些视角图像颠倒问题）
MULTI_CAMERA_CONFIGS = {
    "top": {"position": (0.45, 0.0, 1.5), "rotation": (0, 0, 180), "vertical_flip": True},
    "north": {"position": (0.45, 1.0, 1.5), "rotation": (-30, 0, 0), "vertical_flip": True},
    "south": {"position": (0.45, -1.0, 1.5), "rotation": (-30, 0, 180), "vertical_flip": True},
    "east": {"position": (1.45, 0.0, 1.5), "rotation": (-30, 0, -90), "vertical_flip": True},
    "west": {"position": (-0.55, 0.0, 1.5), "rotation": (-30, 0, 90), "vertical_flip": True},
    "northeast": {"position": (1.15, 0.70, 1.5), "rotation": (-30, 0, -45), "vertical_flip": True},
    "northwest": {"position": (-0.25, 0.70, 1.5), "rotation": (-30, 0, 45), "vertical_flip": True},
    "southeast": {"position": (1.15, -0.70, 1.5), "rotation": (-30, 0, -135), "vertical_flip": True},
    "southwest": {"position": (-0.25, -0.70, 1.5), "rotation": (-30, 0, 135), "vertical_flip": True},
}


class DatasetCollector:
    """数据集采集器（无头模式）- 每个round保存9组数据（9个视角）"""

    def __init__(self, output_dir: str = "dataset", mode: str = "messy",
                 min_boxes: int = 30, max_boxes: int = 100):
        self.world = None
        self.box_gen = None
        self.scene_builder = SceneBuilder()
        self.camera = CameraManager()
        self.multi_cameras = {}  # 九视角相机
        self.stability_checker = StabilityChecker(threshold=0.02)

        self.output_dir = output_dir
        self.current_round_base_dir = None  # round基础目录
        self.mode = mode  # "messy" 或 "neat"

        # 状态机
        self.state = "INIT"
        self.step_count = 0

        # 箱子参数
        self.min_boxes = min_boxes
        self.max_boxes = max_boxes
        self.current_box_count = 0
        self.box_spawn_steps = []
        self.box_paths = []

        # 时序控制
        self.hold_duration = 60  # 箱子生成后保持（1秒）
        self.hold_timer = 0
        self.max_wait_frames = 180  # 初始环境稳定（3秒）
        self.wait_frame_count = 0

        # 采集相关
        self.round_index = 0
        self.visible_boxes_to_test = []
        self.current_test_index = 0
        self.removed_box_path = None

        # 场景状态
        self.initial_scene_state = None
        self.round_results = []

        # 每个视角的图像数据
        self.camera_data = {}  # {视角名: {rgb, depth, mask, id_to_labels}}

        # 统计信息
        self.total_rounds = 0
        self.start_time = None
        self.round_start_time = None
        self.total_samples = 0

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

    def setup_scene(self):
        """设置场景"""
        self.world = World(stage_units_in_meters=1.0)
        self.scene_builder.create_textured_floor(size=5.0)
        self.box_gen = BoxGenerator(texture_dir="assets/cardboard_textures_processed")
        self.camera.create_top_camera(with_depth=True)
        self.create_multi_cameras()

    def create_multi_cameras(self):
        """创建九个方向的相机（含RGB、depth、segmentation）"""
        stage = omni.usd.get_context().get_stage()

        for name, config in MULTI_CAMERA_CONFIGS.items():
            cam_path = f"/World/Camera_{name}"
            camera = UsdGeom.Camera.Define(stage, cam_path)
            xform = UsdGeom.Xformable(camera.GetPrim())
            xform.AddTranslateOp().Set(Gf.Vec3d(*config["position"]))
            xform.AddRotateXYZOp().Set(Gf.Vec3f(*config["rotation"]))
            camera.GetFocalLengthAttr().Set(18.0)

            render_product = rep.create.render_product(cam_path, (640, 480))

            # RGB annotator
            rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            rgb_annotator.attach([render_product])

            # Depth annotator
            depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
            depth_annotator.attach([render_product])

            # Segmentation annotator
            seg_annotator = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation")
            seg_annotator.attach([render_product])

            self.multi_cameras[name] = {
                "path": cam_path,
                "render_product": render_product,
                "rgb_annotator": rgb_annotator,
                "depth_annotator": depth_annotator,
                "seg_annotator": seg_annotator,
                "vertical_flip": config.get("vertical_flip", False),
            }

    def get_visible_boxes_for_view(self, view_name: str, min_visible_ratio: float = 0.2, collect_stats: bool = False) -> set:
        """
        获取单个视角可见的箱子prim_path集合

        使用基于3D投影的可见性判断：实际像素数 / 理论投影像素数

        Args:
            view_name: 视角名称
            min_visible_ratio: 最小可见比例阈值
            collect_stats: 是否收集统计数据

        Returns:
            visible_paths: 可见箱子路径集合
            如果 collect_stats=True，还会在 self.visibility_stats[view_name] 中保存统计数据
        """
        cam_data = self.camera_data.get(view_name)
        if not cam_data:
            return set()

        mask = cam_data["mask"]
        id_to_labels = cam_data["id_to_labels"]

        if mask is None:
            return set()

        # 获取相机配置
        cam_config = MULTI_CAMERA_CONFIGS.get(view_name, {})
        cam_position = cam_config.get("position", (0, 0, 1.5))

        unique_ids = np.unique(mask)
        visible_paths = set()

        # 收集统计数据
        if collect_stats:
            if not hasattr(self, 'visibility_stats'):
                self.visibility_stats = {}
            self.visibility_stats[view_name] = []

        for uid in unique_ids:
            if uid == 0:
                continue
            label_str = str(id_to_labels.get(str(uid), ""))
            if "/World/box_" not in label_str:
                continue

            # 获取实际像素数
            actual_pixels = int(np.sum(mask == uid))

            # 查找对应的箱子信息
            box_info = None
            box_name = None
            box_prim_path = None
            for name, info in self.box_gen.boxes.items():
                if label_str.startswith(info["prim_path"] + "/") or label_str == info["prim_path"]:
                    box_info = info
                    box_name = name
                    box_prim_path = info["prim_path"]
                    break

            if box_info is None:
                continue

            # 获取箱子当前位置
            box_position = self.box_gen.get_box_position(box_name)
            if box_position is None:
                continue

            # 估算理论投影像素数
            expected_pixels = ImageUtils.estimate_projected_pixels(
                box_size=box_info["size"],
                box_position=box_position,
                camera_position=cam_position,
                focal_length=18.0,
                image_size=(640, 480)
            )

            # 计算可见比例
            visibility_ratio = actual_pixels / expected_pixels if expected_pixels > 0 else 0

            # 收集统计数据（不做任何过滤）
            if collect_stats:
                self.visibility_stats[view_name].append({
                    "box_name": box_name,
                    "prim_path": box_prim_path,
                    "actual_pixels": actual_pixels,
                    "expected_pixels": int(expected_pixels),
                    "visibility_ratio": round(visibility_ratio, 4),
                    "box_size": [round(s, 4) for s in box_info["size"]],
                    "box_position": [round(p, 4) for p in box_position]
                })

            # 判断是否可见
            if expected_pixels > 0 and visibility_ratio >= min_visible_ratio and actual_pixels >= 300:
                visible_paths.add(box_prim_path)

        return visible_paths

    def get_all_views_visible_boxes(self, min_visible_ratio: float = 0.2) -> list:
        """获取所有视角可见箱子的并集"""
        all_visible_paths = set()

        # 收集所有视角可见的箱子路径
        for view_name in self.camera_data.keys():
            visible_paths = self.get_visible_boxes_for_view(view_name, min_visible_ratio)
            all_visible_paths.update(visible_paths)

        # 转换为箱子信息列表
        visible_boxes = []
        for name, info in self.box_gen.boxes.items():
            if info["prim_path"] in all_visible_paths:
                visible_boxes.append({
                    "name": name,
                    "prim_path": info["prim_path"],
                    "uid": 0
                })

        return visible_boxes

    def randomize_lighting(self):
        """随机化环境光"""
        stage = omni.usd.get_context().get_stage()

        # 查找或创建 DomeLight (环境光)
        dome_light_path = "/World/DomeLight"
        prim = stage.GetPrimAtPath(dome_light_path)
        if prim.IsValid():
            dome_light = UsdLux.DomeLight(prim)
        else:
            dome_light = UsdLux.DomeLight.Define(stage, dome_light_path)

        # 随机强度
        intensity = np.random.uniform(300, 1200)
        dome_light.GetIntensityAttr().Set(intensity)

        # 随机旋转 (改变阴影方向)
        xform = UsdGeom.Xformable(dome_light)
        xform.ClearXformOpOrder()
        xform.AddRotateXYZOp().Set(Gf.Vec3f(
            np.random.uniform(0, 360),
            np.random.uniform(0, 360),
            0
        ))

        # 随机色温 (冷光/暖光)
        enable_temp = dome_light.GetEnableColorTemperatureAttr()
        enable_temp.Set(True)
        temp = np.random.uniform(4000, 8000)
        dome_light.GetColorTemperatureAttr().Set(temp)

    def start_new_round(self):
        """开始新一轮"""
        self.round_index += 1
        self.round_start_time = time.time()
        self.current_box_count = np.random.randint(self.min_boxes, self.max_boxes + 1)

        # 随机化光照和地面
        self.randomize_lighting()
        self.scene_builder.randomize_floor_texture()

        # 基础目录名（不含视角后缀），包含模式标识
        mode_suffix = "neat" if self.mode == "neat" else "messy"
        self.current_round_base_dir = os.path.join(
            self.output_dir, f"round_{self.round_index:03d}_{self.current_box_count}_{mode_suffix}"
        )

        # 为每个视角创建目录
        for view_name in MULTI_CAMERA_CONFIGS.keys():
            view_dir = f"{self.current_round_base_dir}_{view_name}"
            os.makedirs(view_dir, exist_ok=True)

        if self.mode == "neat":
            # 整齐模式：一次性生成所有箱子，不需要分批
            self.box_spawn_steps = [(20, self.current_box_count)]
        else:
            # 散乱模式：分批掉落
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
        self.camera_data = {}
        self.hold_timer = 0
        self.step_count = 0
        self.state = "SPAWNING"

    def spawn_boxes(self, count: int):
        """生成箱子"""
        if self.mode == "neat":
            # 整齐堆叠模式
            paths = self.box_gen.create_neat_stacked_boxes(
                count=count,
                center=[0.45, 0.0],
                base_size=(0.12, 0.10, 0.08),
                size_variation=0.025,
                grid_spacing=0.02,
                max_layers=6,
                mass_range=(0.3, 1.5),
                position_noise=0.01,
                rotation_noise=5.0
            )
        else:
            # 散乱掉落模式
            paths = self.box_gen.create_random_boxes(
                count=count,
                center=[0.45, 0.0],
                spread=0.15,
                drop_height=0.30,
                size_range=((0.05, 0.20), (0.05, 0.20), (0.05, 0.20)),
                mass_range=(0.3, 1.5)
            )
        self.box_paths.extend(paths)

    def clear_scene(self):
        """清除场景"""
        for path in list(self.box_paths):
            self.box_gen.delete_box(path)
        self.box_paths = []
        self.box_gen._boxes = {}
        self.box_gen._box_count = 0

    def capture_initial_state(self):
        """捕获初始状态 - 为每个视角保存RGB、depth、mask，并记录各视角可见箱子"""
        for view_name, cam_info in self.multi_cameras.items():
            view_dir = f"{self.current_round_base_dir}_{view_name}"
            vertical_flip = cam_info.get("vertical_flip", False)

            # 获取RGB
            rgb_annotator = cam_info["rgb_annotator"]
            rgb_data = rgb_annotator.get_data()
            rgb = rgb_data[:, :, :3].copy() if rgb_data is not None else None
            if rgb is not None and vertical_flip:
                rgb = cv2.flip(rgb, 0)

            # 获取depth和mask
            depth_data = cam_info["depth_annotator"].get_data() if "depth_annotator" in cam_info else None
            # depth与RGB同步翻转，保持空间对齐
            if depth_data is not None and vertical_flip:
                depth_data = np.flip(depth_data, axis=0)
            mask_data, id_to_labels = None, {}
            if "seg_annotator" in cam_info:
                seg_data = cam_info["seg_annotator"].get_data()
                if seg_data is not None:
                    mask_data = seg_data["data"].copy()
                    # mask与RGB同步翻转，保证标注对齐
                    if mask_data is not None and vertical_flip:
                        mask_data = np.flip(mask_data, axis=0)
                    id_to_labels = seg_data.get("info", {}).get("idToLabels", {})

            # 存储到camera_data
            self.camera_data[view_name] = {
                "rgb": rgb,
                "depth": depth_data,
                "mask": mask_data,
                "id_to_labels": id_to_labels
            }

            # 保存初始图像
            if rgb is not None:
                noisy_rgb = ImageUtils.add_camera_noise(rgb)
                ImageUtils.save_rgb(noisy_rgb, os.path.join(view_dir, "rgb.png"))

            if depth_data is not None:
                noisy_depth = ImageUtils.add_depth_noise(depth_data)
                ImageUtils.save_depth(noisy_depth, os.path.join(view_dir, "depth.png"))
                np.save(os.path.join(view_dir, "depth.npy"), noisy_depth)

            if mask_data is not None:
                np.save(os.path.join(view_dir, "mask.npy"), mask_data)

        # 获取所有视角可见箱子的并集（用于测试）
        self.visible_boxes_to_test = self.get_all_views_visible_boxes()

        # 为每个视角保存该视角可见的箱子信息和掩码
        for view_name in self.camera_data.keys():
            view_dir = f"{self.current_round_base_dir}_{view_name}"
            cam_data = self.camera_data[view_name]
            mask_data = cam_data["mask"]
            id_to_labels = cam_data["id_to_labels"]

            # 获取该视角可见的箱子路径，同时收集统计数据
            view_visible_paths = self.get_visible_boxes_for_view(view_name, collect_stats=True)
            self.camera_data[view_name]["visible_paths"] = view_visible_paths

            # 保存可见性统计数据到JSON（用于分析阈值）
            if hasattr(self, 'visibility_stats') and view_name in self.visibility_stats:
                stats_file = os.path.join(view_dir, "visibility_stats.json")
                with open(stats_file, "w") as f:
                    json.dump(self.visibility_stats[view_name], f, indent=2)

            if mask_data is not None:
                # 保存该视角可见箱子掩码
                visible_mask = np.zeros_like(mask_data, dtype=np.uint8)
                for box_path in view_visible_paths:
                    box_id = ImageUtils.get_mask_id(box_path, id_to_labels)
                    if box_id is not None:
                        visible_mask[mask_data == box_id] = 255
                cv2.imwrite(os.path.join(view_dir, "visible_mask.png"), visible_mask)

            # 保存该视角可见箱子列表（含 bbox）
            view_visible_info = []
            for name, info in self.box_gen.boxes.items():
                if info["prim_path"] in view_visible_paths:
                    box_data = {"name": name, "prim_path": info["prim_path"]}
                    # 计算 bbox
                    box_id = ImageUtils.get_mask_id(info["prim_path"], id_to_labels)
                    if box_id is not None and mask_data is not None:
                        bbox = ImageUtils.get_bbox_from_mask(mask_data, box_id)
                        if bbox:
                            box_data["bbox"] = bbox
                    view_visible_info.append(box_data)
            with open(os.path.join(view_dir, "visible_boxes.json"), "w") as f:
                json.dump(view_visible_info, f, indent=2)

    def save_removal_result(self, test_index: int, box: dict, result: dict):
        """保存移除结果 - 只为该视角可见的箱子保存数据"""
        for view_name, cam_data in self.camera_data.items():
            view_visible_paths = cam_data.get("visible_paths", set())

            # 如果被移除的箱子在该视角不可见，跳过保存
            if box["prim_path"] not in view_visible_paths:
                continue

            view_dir = f"{self.current_round_base_dir}_{view_name}"
            box_dir = os.path.join(view_dir, "removals", str(test_index))
            os.makedirs(box_dir, exist_ok=True)

            rgb = cam_data["rgb"]
            mask = cam_data["mask"]
            id_to_labels = cam_data["id_to_labels"]

            if rgb is not None and mask is not None:
                # 标注图（红绿掩码）
                annotated = ImageUtils.create_annotated_image(
                    rgb, mask, id_to_labels, box["prim_path"], result["moved_boxes"]
                )
                cv2.imwrite(os.path.join(box_dir, "annotated.png"), annotated)

                # 消失箱子掩码
                removed_id = ImageUtils.get_mask_id(box["prim_path"], id_to_labels)
                if removed_id is not None:
                    removed_mask = (mask == removed_id).astype(np.uint8) * 255
                    cv2.imwrite(os.path.join(box_dir, "removed_mask.png"), removed_mask)

                # 受影响箱子掩码（只保存该视角可见的受影响箱子）
                affected_mask = np.zeros_like(mask, dtype=np.uint8)
                visible_affected = []
                for affected_box in result["moved_boxes"]:
                    box_path = affected_box.get("path", affected_box.get("prim_path", ""))
                    if box_path in view_visible_paths:
                        affected_id = ImageUtils.get_mask_id(box_path, id_to_labels)
                        if affected_id is not None:
                            affected_mask[mask == affected_id] = 255
                        visible_affected.append(affected_box)
                cv2.imwrite(os.path.join(box_dir, "affected_mask.png"), affected_mask)

                # 保存结果JSON（只包含该视角可见的受影响箱子）
                result_data = {
                    "removed_box": {"name": box["name"], "prim_path": box["prim_path"]},
                    "is_stable": result["is_stable"],
                    "stability_label": result["stability_label"],
                    "affected_boxes": visible_affected
                }
                with open(os.path.join(box_dir, "result.json"), "w") as f:
                    json.dump(result_data, f, indent=2)

    def save_round_summary(self):
        """保存轮次总结 - 每个视角只显示该视角可见的箱子"""
        stable = [r["removed_box"] for r in self.round_results if r["is_stable"]]
        unstable = [r["removed_box"] for r in self.round_results if not r["is_stable"]]

        # 为每个视角保存总结图
        for view_name, cam_data in self.camera_data.items():
            view_dir = f"{self.current_round_base_dir}_{view_name}"
            view_visible_paths = cam_data.get("visible_paths", set())
            rgb = cam_data["rgb"]
            mask = cam_data["mask"]
            id_to_labels = cam_data["id_to_labels"]

            # 过滤出该视角可见的箱子
            view_stable = [b for b in stable if b["prim_path"] in view_visible_paths]
            view_unstable = [b for b in unstable if b["prim_path"] in view_visible_paths]

            if rgb is not None and mask is not None:
                summary = ImageUtils.create_summary_image(
                    rgb, mask, id_to_labels, view_stable, view_unstable
                )
                cv2.imwrite(os.path.join(view_dir, "summary.png"), summary)

            # 每个视角保存自己的总结JSON
            summary_data = {
                "total_boxes": self.current_box_count,
                "view_visible": len(view_visible_paths),
                "view_tested": len(view_stable) + len(view_unstable),
                "view_stable": len(view_stable),
                "view_unstable": len(view_unstable)
            }
            with open(os.path.join(view_dir, "summary.json"), "w") as f:
                json.dump(summary_data, f, indent=2)

        # 统计本轮信息
        round_time = time.time() - self.round_start_time
        # 计算实际保存的样本数（每个视角只保存该视角可见的箱子）
        round_samples = sum(
            len([b for b in self.round_results if b["removed_box"]["prim_path"] in cam_data.get("visible_paths", set())])
            for cam_data in self.camera_data.values()
        )
        self.total_samples += round_samples

        print(f"[{self.round_index}/{self.total_rounds}] "
              f"boxes={self.current_box_count}, tested={len(self.round_results)}, "
              f"stable={len(stable)}, unstable={len(unstable)}, "
              f"time={round_time:.1f}s")

    def run(self, num_rounds: int = 10):
        """运行采集"""
        self.total_rounds = num_rounds
        self.start_time = time.time()

        self.setup_scene()
        self.world.reset()

        start_round = self.round_index + 1
        mode_desc = "neat (stacked)" if self.mode == "neat" else "messy (random drop)"
        print(f"Dataset collection: mode={mode_desc}, rounds {start_round}-{num_rounds}, output={self.output_dir}")

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
                            self.stability_checker.reset_convergence()
                            self.hold_timer = 0

            elif self.state == "STABILIZING":
                self.hold_timer += 1
                converged = self.stability_checker.check_converged(
                    self.box_gen, min_stable_frames=30
                )
                if converged or self.hold_timer >= 300:  # 收敛或最多等5秒
                    self.initial_scene_state = self.box_gen.save_scene_state()
                    # capture_initial_state 会设置 self.visible_boxes_to_test（所有视角并集）
                    self.capture_initial_state()
                    if not self.visible_boxes_to_test:
                        self.state = "ROUND_COMPLETE"
                    else:
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
                    self.stability_checker.reset_convergence()
                    self.wait_frame_count = 0
                    self.state = "WAITING_STABILITY"

            elif self.state == "WAITING_STABILITY":
                self.wait_frame_count += 1
                converged = self.stability_checker.check_converged(
                    self.box_gen, min_stable_frames=30, excluded_path=self.removed_box_path
                )
                if converged or self.wait_frame_count >= 120:  # 移除后最多等待2秒
                    result = self.stability_checker.check(self.box_gen, self.removed_box_path)
                    box = self.visible_boxes_to_test[self.current_test_index]
                    self.save_removal_result(self.current_test_index, box, result)
                    self.round_results.append({"removed_box": box, "is_stable": result["is_stable"]})

                    self.current_test_index += 1
                    if self.current_test_index < len(self.visible_boxes_to_test):
                        self.box_gen.restore_scene_state(self.initial_scene_state)
                        self.stability_checker.reset_convergence()
                        self.wait_frame_count = 0
                        self.state = "RESTORING"
                    else:
                        self.state = "ROUND_COMPLETE"

            elif self.state == "RESTORING":
                self.wait_frame_count += 1
                converged = self.stability_checker.check_converged(
                    self.box_gen, min_stable_frames=30
                )
                if converged or self.wait_frame_count >= 120:  # 恢复后最多等待2秒
                    self.state = "TESTING"

            elif self.state == "ROUND_COMPLETE":
                self.save_round_summary()
                if self.round_index < num_rounds:
                    self.clear_scene()
                    self.start_new_round()
                    spawn_batch_index = 0
                else:
                    total_time = time.time() - self.start_time
                    print(f"\nComplete! samples={self.total_samples}, time={total_time/60:.1f}min")
                    break

        simulation_app.close()


if __name__ == "__main__":
    collector = DatasetCollector(
        output_dir=args.output,
        mode=args.mode,
        min_boxes=args.min_boxes,
        max_boxes=args.max_boxes
    )
    collector.run(num_rounds=args.rounds)
