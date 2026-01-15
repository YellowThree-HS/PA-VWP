"""
BoxWorld MVP - 主仿真脚本
纸箱堆叠稳定性测试
适配Isaac Sim 5.1版本
"""

from isaacsim import SimulationApp

# 启动仿真应用
simulation_app = SimulationApp({"headless": False})

import numpy as np
from isaacsim.core.api import World
from pxr import UsdGeom, UsdPhysics, Gf, Sdf
import omni.usd
import omni.replicator.core as rep

# 导入自定义模块
import sys
sys.path.append("src")
from box_generator import BoxGenerator


class BoxPickSimulation:
    """纸箱堆叠稳定性测试仿真"""

    def __init__(self):
        self.world = None
        self.box_gen = None

        # 状态机
        self.state = "INIT"
        self.step_count = 0
        self.box_spawn_step = 20  # 第一批箱子的时机
        self.box_spawn_step_2 = 80  # 第二批箱子（1秒后，60步）
        self.box_spawn_step_3 = 140  # 第三批箱子（再1秒后）
        self.hold_duration = 5 * 60  # 保持5秒 (60Hz)
        self.hold_timer = 0

        # 多箱子相关
        self.box_paths = []  # 所有纸箱路径

        # 箱子移除相关
        self.removed_box_path = None  # 被移除的箱子路径
        self.removal_force_applied = False  # 是否已施加移除力
        self.post_removal_wait = 30  # 移除后等待时间（0.5秒）
        self.post_removal_timer = 0

        # 稳定性检测相关
        self.positions_before_removal = {}  # 移除前的位置快照
        self.stability_threshold = 0.02  # 位置变化阈值（2cm）

        # 循环测试相关
        self.initial_scene_state = None  # 初始场景状态
        self.test_iteration = 0  # 当前测试轮次
        self.visible_boxes_to_test = []  # 待测试的可见箱子列表
        self.total_boxes_to_test = 0  # 总共要测试的箱子数
        self.test_results = []  # 测试结果记录

        # 初始图像数据（只保存一次）
        self.initial_rgb_image = None
        self.initial_mask_data = None
        self.initial_id_to_labels = None

        # Replicator相关
        self.render_product = None
        self.rgb_annotator = None
        self.seg_annotator = None

    def _create_cameras(self):
        """创建顶部深度相机"""
        stage = omni.usd.get_context().get_stage()

        # 顶部相机 - 俯视箱子堆
        top_cam_path = "/World/TopCamera"
        top_cam = UsdGeom.Camera.Define(stage, top_cam_path)
        top_xform = UsdGeom.Xformable(top_cam.GetPrim())
        # 位置：箱子堆上方1.5m
        top_xform.AddTranslateOp().Set(Gf.Vec3d(0.45, 0.0, 1.5))
        # 旋转：(0, 0, 180) 向下看
        top_xform.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 180))
        top_cam.GetFocalLengthAttr().Set(18.0)

        # 设置Replicator获取RGB和实例分割掩码
        self.render_product = rep.create.render_product(top_cam_path, (640, 480))
        self.rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb_annotator.attach([self.render_product])
        self.seg_annotator = rep.AnnotatorRegistry.get_annotator("instance_segmentation")
        self.seg_annotator.attach([self.render_product])

        print("Camera created: TopCamera (1.5m height, looking down)")

    def setup_scene(self):
        """设置仿真场景（不含箱子，箱子稍后动态生成）"""
        # 创建世界
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        # 初始化箱子生成器（稍后动态生成箱子）
        self.box_gen = BoxGenerator()

        # 创建深度相机
        self._create_cameras()

        print("Scene setup complete (without box)!")

    def spawn_box(self, batch: int = 1):
        """分批生成随机大小的纸箱从半空中掉落，共100个

        Args:
            batch: 第几批 (1, 2, 3)
        """
        # 每批数量：34 + 33 + 33 = 100
        batch_counts = {1: 34, 2: 33, 3: 33}
        count = batch_counts.get(batch, 34)

        # 箱子尺寸：确保不太扁，每个维度都在5-15cm之间
        # 这样任意一个面都能被吸盘吸住
        paths = self.box_gen.create_random_boxes(
            count=count,
            center=[0.45, 0.0],
            spread=0.15,  # 稍微增大散布范围
            drop_height=0.30,  # 从30cm高度开始掉落
            size_range=((0.06, 0.12), (0.06, 0.12), (0.06, 0.12)),
            mass_range=(0.3, 1.5)
        )
        self.box_paths.extend(paths)
        print(f"Batch {batch}: Spawned {count} boxes, total: {len(self.box_paths)}")

    def capture_initial_state(self):
        """捕获并保存初始RGB图像和掩码数据"""
        import cv2
        import os
        from datetime import datetime

        os.makedirs("result", exist_ok=True)

        # 获取RGB图像
        rgb_data = self.rgb_annotator.get_data()
        if rgb_data is not None:
            self.initial_rgb_image = rgb_data[:, :, :3].copy()
            # 保存一张初始RGB
            rgb_bgr = cv2.cvtColor(self.initial_rgb_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite("result/initial_rgb.png", rgb_bgr)
            print("Initial RGB saved to result/initial_rgb.png")

        # 获取分割掩码
        seg_data = self.seg_annotator.get_data()
        if seg_data is not None:
            self.initial_mask_data = seg_data["data"].copy()
            self.initial_id_to_labels = seg_data.get("info", {}).get("idToLabels", {})

    def get_all_visible_boxes(self, min_visible_ratio: float = 0.20) -> list:
        """获取所有在掩码中可见的箱子列表，过滤掉遮挡太多的

        Args:
            min_visible_ratio: 最小可见比例，低于此值的箱子会被跳过
        """
        seg_data = self.seg_annotator.get_data()
        if seg_data is None:
            print("[DEBUG] seg_data is None!")
            return []

        id_to_labels = seg_data.get("info", {}).get("idToLabels", {})
        mask = seg_data["data"]
        unique_ids = np.unique(mask)

        # 先找出所有有效箱子的uid和像素数
        box_pixels = {}
        for uid in unique_ids:
            if uid == 0:
                continue
            label_info = id_to_labels.get(str(uid), "")
            label_str = str(label_info)
            # 检查是否是箱子（路径包含/World/box_）
            if "/World/box_" in label_str:
                box_pixels[uid] = np.sum(mask == uid)

        if not box_pixels:
            print("[DEBUG] No box pixels found!")
            return []

        # 用箱子中最大像素数作为参考
        max_pixels = max(box_pixels.values())
        print(f"[DEBUG] 箱子像素范围: {min(box_pixels.values())} - {max_pixels}")

        visible_boxes = []
        skipped_count = 0

        added_names = set()  # 防止重复添加
        for uid, pixel_count in box_pixels.items():
            visible_ratio = pixel_count / max_pixels

            if visible_ratio < min_visible_ratio:
                skipped_count += 1
                continue

            label_info = id_to_labels.get(str(uid), {})
            label_str = str(label_info)

            for name, info in self.box_gen.boxes.items():
                # 精确匹配路径，避免 box_1 匹配到 box_10, box_11 等
                if info["prim_path"] == label_str and name not in added_names:
                    visible_boxes.append((name, info["prim_path"]))
                    added_names.add(name)
                    break

        if skipped_count > 0:
            print(f"跳过 {skipped_count} 个遮挡过多的箱子（可见比例 < {min_visible_ratio*100:.0f}%）")

        print(f"可见箱子数量: {len(visible_boxes)}")
        return visible_boxes

    def save_annotated_result(self, iteration: int, removed_path: str, affected_boxes: list):
        """保存标注结果图：消失箱子(绿色)、受影响箱子(红色)"""
        import cv2

        if self.initial_rgb_image is None or self.initial_mask_data is None:
            print("No initial data available")
            return

        # 转换为BGR
        rgb_bgr = cv2.cvtColor(self.initial_rgb_image, cv2.COLOR_RGB2BGR)

        # 创建半透明叠加层
        overlay = rgb_bgr.copy()
        alpha = 0.4

        # 找到被移除箱子的mask id
        removed_id = None
        for uid, label_info in self.initial_id_to_labels.items():
            if removed_path in str(label_info):
                removed_id = int(uid)
                break

        # 被移除的箱子标绿色
        if removed_id is not None:
            removed_mask = self.initial_mask_data == removed_id
            overlay[removed_mask] = [0, 255, 0]  # BGR绿色

        # 受影响的箱子标红色
        for box_info in affected_boxes:
            box_path = box_info["path"]
            for uid, label_info in self.initial_id_to_labels.items():
                if box_path in str(label_info):
                    affected_mask = self.initial_mask_data == int(uid)
                    overlay[affected_mask] = [0, 0, 255]  # BGR红色
                    break

        # 混合
        result = cv2.addWeighted(rgb_bgr, alpha, overlay, 1 - alpha, 0)

        # 保存
        removed_name = removed_path.split("/")[-1]
        save_path = f"result/iter{iteration:02d}_{removed_name}.png"
        cv2.imwrite(save_path, result)
        print(f"Annotated result saved to {save_path}")

    def select_next_box_for_removal(self):
        """从待测试列表中选择下一个箱子"""
        if not self.visible_boxes_to_test:
            return False

        name, prim_path = self.visible_boxes_to_test.pop(0)
        pos = self.box_gen.get_box_position(name)
        print(f"Selected box for removal: {name} at position {pos}")
        self.removed_box_path = prim_path
        return True

    def apply_removal_to_selected_box(self):
        """删除已选择的箱子"""
        if self.removed_box_path is None:
            print("No box selected for removal")
            return False

        # 直接删除箱子
        self.box_gen.delete_box(self.removed_box_path)
        return True

    def check_stability(self) -> dict:
        """
        检测移除箱子后其他箱子的稳定性

        Returns:
            dict: 稳定性检测结果
        """
        # 获取当前位置（排除被移除的箱子）
        positions_after = self.box_gen.get_all_box_positions(
            exclude_paths=[self.removed_box_path]
        )

        moved_boxes = []
        stable_boxes = []

        for path, pos_before in self.positions_before_removal.items():
            if path == self.removed_box_path:
                continue
            if path not in positions_after:
                continue

            pos_after = positions_after[path]
            displacement = np.linalg.norm(pos_after - pos_before)

            box_name = path.split("/")[-1]
            if displacement > self.stability_threshold:
                moved_boxes.append({
                    "name": box_name,
                    "path": path,
                    "displacement": displacement,
                    "pos_before": pos_before,
                    "pos_after": pos_after
                })
            else:
                stable_boxes.append(box_name)

        # 判断整体稳定性
        is_stable = len(moved_boxes) == 0

        return {
            "is_stable": is_stable,
            "moved_boxes": moved_boxes,
            "stable_boxes": stable_boxes,
            "total_boxes": len(positions_after)
        }

    def print_stability_report(self, result: dict):
        """打印稳定性检测报告"""
        print("\n" + "=" * 50)
        print("稳定性检测报告")
        print("=" * 50)

        removed_name = self.removed_box_path.split("/")[-1]
        print(f"被移除的箱子: {removed_name}")
        print(f"剩余箱子数量: {result['total_boxes']}")
        print(f"位移阈值: {self.stability_threshold * 100:.1f} cm")
        print("-" * 50)

        if result["is_stable"]:
            print("结果: ✓ 稳定")
            print("所有箱子位置变化均在阈值内")
        else:
            print(f"结果: ✗ 不稳定")
            print(f"发生位移的箱子数量: {len(result['moved_boxes'])}")
            print("\n位移详情:")
            for box in result["moved_boxes"]:
                print(f"  - {box['name']}: 位移 {box['displacement']*100:.2f} cm")

        print("=" * 50 + "\n")

    def print_final_summary(self):
        """打印所有测试轮次的总结"""
        print("\n" + "=" * 60)
        print("最终测试总结")
        print("=" * 60)
        print(f"总测试轮次: {len(self.test_results)}")

        stable_count = sum(1 for r in self.test_results if r["is_stable"])
        unstable_count = len(self.test_results) - stable_count

        print(f"稳定次数: {stable_count}")
        print(f"不稳定次数: {unstable_count}")
        print("-" * 60)

        for i, result in enumerate(self.test_results):
            status = "稳定" if result["is_stable"] else "不稳定"
            moved = len(result["moved_boxes"])
            print(f"轮次 {i+1}: 移除 {result['removed_box']} -> {status} (位移箱子: {moved})")

        print("=" * 60 + "\n")

    def run(self):
        """运行仿真"""
        self.setup_scene()

        # 重置世界
        self.world.reset()

        print("Starting simulation...")

        # 主仿真循环
        while simulation_app.is_running():
            self.world.step(render=True)
            self.step_count += 1

            # 状态机控制
            if self.state == "INIT":
                # 在指定时机生成箱子（分三批）
                if self.step_count == self.box_spawn_step:
                    self.spawn_box(batch=1)

                if self.step_count == self.box_spawn_step_2:
                    self.spawn_box(batch=2)

                if self.step_count == self.box_spawn_step_3:
                    self.spawn_box(batch=3)
                    self.state = "HOLD"
                    self.hold_timer = 0
                    print("All boxes dropped, holding for 20 seconds...")

            elif self.state == "HOLD":
                self.hold_timer += 1
                # 等待2秒让箱子稳定
                if self.hold_timer == 120:
                    # 第一次进入时保存初始场景状态和图像
                    if self.initial_scene_state is None:
                        self.initial_scene_state = self.box_gen.save_scene_state()
                        self.capture_initial_state()
                        # 获取所有可见箱子
                        self.visible_boxes_to_test = self.get_all_visible_boxes()
                        self.total_boxes_to_test = len(self.visible_boxes_to_test)
                        print(f"\n{'='*50}")
                        print(f"开始循环测试，共 {self.total_boxes_to_test} 个可见箱子")
                        print(f"{'='*50}\n")

                # 额外等待30帧(0.5秒)再开始测试，确保位置稳定
                if self.hold_timer == 150:
                    # 检查是否还有箱子待测试
                    if not self.visible_boxes_to_test:
                        print("没有可见箱子可测试，结束")
                        self.print_final_summary()
                        self.state = "DONE"
                    else:
                        self.test_iteration += 1
                        print(f"\n--- 测试轮次 {self.test_iteration}/{self.total_boxes_to_test} ---")

                        # 选择下一个要移除的箱子
                        self.select_next_box_for_removal()
                        # 记录移除前位置
                        self.positions_before_removal = self.box_gen.get_all_box_positions()
                        # 进入移除状态
                        self.state = "REMOVE_BOX"

            elif self.state == "REMOVE_BOX":
                if not self.removal_force_applied:
                    # 删除箱子
                    self.apply_removal_to_selected_box()
                    self.removal_force_applied = True
                    self.post_removal_timer = 0
                else:
                    # 等待箱子稳定
                    self.post_removal_timer += 1
                    if self.post_removal_timer >= self.post_removal_wait:
                        # 进行稳定性检测
                        stability_result = self.check_stability()
                        self.print_stability_report(stability_result)

                        # 记录测试结果
                        removed_name = self.removed_box_path.split("/")[-1]
                        stability_result["removed_box"] = removed_name
                        self.test_results.append(stability_result)

                        # 保存标注结果图（绿色=消失，红色=受影响）
                        self.save_annotated_result(
                            self.test_iteration,
                            self.removed_box_path,
                            stability_result["moved_boxes"]
                        )

                        # 判断是否继续下一轮
                        if self.visible_boxes_to_test:
                            # 还有箱子待测试，恢复场景
                            print(f"\n恢复场景到初始状态...")
                            self.box_gen.restore_scene_state(self.initial_scene_state)
                            # 重置状态变量
                            self.removal_force_applied = False
                            self.removed_box_path = None
                            self.hold_timer = 0
                            self.state = "RESTORE_WAIT"
                        else:
                            # 所有测试完成
                            self.print_final_summary()
                            self.state = "DONE"

            elif self.state == "RESTORE_WAIT":
                # 等待场景恢复稳定
                self.hold_timer += 1
                if self.hold_timer >= 60:  # 等待1秒
                    self.hold_timer = 0
                    self.state = "HOLD"

            elif self.state == "DONE":
                self.hold_timer += 1
                if self.hold_timer >= self.hold_duration:
                    print("Simulation complete, closing...")
                    break

        simulation_app.close()


if __name__ == "__main__":
    sim = BoxPickSimulation()
    sim.run()
