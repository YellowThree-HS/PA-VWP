"""
BoxWorld MVP - 主仿真脚本
UR5机器人使用吸盘抓取纸箱
适配Isaac Sim 5.1版本
"""

from isaacsim import SimulationApp

# 启动仿真应用
simulation_app = SimulationApp({"headless": False})

import numpy as np
from isaacsim.core.api import World
from pxr import UsdGeom, UsdPhysics, Gf
import omni.usd

# 导入自定义模块
import sys
sys.path.append("src")
from ur5_controller import UR5eController
from suction_gripper import SuctionGripper
from box_generator import BoxGenerator


class BoxPickSimulation:
    """纸箱抓取仿真"""

    def __init__(self):
        self.world = None
        self.ur5 = None
        self.gripper = None
        self.box_gen = None
        self.box_prim_path = None

        # 状态机
        self.state = "INIT"
        self.step_count = 0
        self.grasp_delay = 0  # 抓取延迟计数
        self.home_steps = 10  # UR5运动到直立姿态所需步数
        self.box_spawn_step = 20  # 生成箱子的时机
        self.stable_steps = 300  # 箱子落地稳定所需步数（增加等待时间让10个箱子稳定）

        # 多箱子相关
        self.box_paths = []  # 所有纸箱路径
        self.target_box_path = None  # 目标纸箱（最上面的）
        self.target_box_size = None  # 目标纸箱尺寸

        # 完成后计时器
        self.done_timer = 0
        self.shutdown_delay = 20 * 60  # 20秒 (假设60Hz仿真频率)

    def _create_base(self, height: float = 0.15, radius: float = 0.1):
        """创建UR5基座"""
        stage = omni.usd.get_context().get_stage()

        # 创建圆柱体基座
        base_path = "/World/UR5_Base"
        cylinder = UsdGeom.Cylinder.Define(stage, base_path)
        cylinder.GetRadiusAttr().Set(radius)
        cylinder.GetHeightAttr().Set(height)
        cylinder.GetAxisAttr().Set("Z")

        # 设置位置（圆柱体中心在height/2处）
        xform = UsdGeom.Xformable(cylinder.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, height / 2))

        # 设置颜色（深灰色）
        cylinder.GetDisplayColorAttr().Set([(0.3, 0.3, 0.3)])

        # 添加碰撞
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())

    def _create_fence(self, center: list = None, width: float = 0.4, depth: float = 0.3, height: float = 0.2):
        """
        创建三面围栏，让箱子聚集在机械臂前方

        Args:
            center: 围栏中心位置 [x, y]
            width: 围栏宽度（左右方向）
            depth: 围栏深度（前后方向）
            height: 围栏高度
        """
        if center is None:
            center = [0.4, 0.0]

        stage = omni.usd.get_context().get_stage()
        wall_thickness = 0.02  # 墙壁厚度

        # 后墙（远离机械臂的一侧）
        back_path = "/World/Fence/BackWall"
        back_xform = UsdGeom.Xform.Define(stage, back_path)
        back_pos = [center[0] + depth / 2 + wall_thickness / 2, center[1], height / 2]
        back_xform.AddTranslateOp().Set(Gf.Vec3d(*back_pos))

        back_cube = UsdGeom.Cube.Define(stage, f"{back_path}/geometry")
        back_cube.GetSizeAttr().Set(1.0)
        back_cube.AddScaleOp().Set(Gf.Vec3f(wall_thickness, width, height))
        back_cube.GetDisplayColorAttr().Set([Gf.Vec3f(0.5, 0.5, 0.5)])
        UsdPhysics.CollisionAPI.Apply(back_cube.GetPrim())

        # 左墙
        left_path = "/World/Fence/LeftWall"
        left_xform = UsdGeom.Xform.Define(stage, left_path)
        left_pos = [center[0], center[1] - width / 2 - wall_thickness / 2, height / 2]
        left_xform.AddTranslateOp().Set(Gf.Vec3d(*left_pos))

        left_cube = UsdGeom.Cube.Define(stage, f"{left_path}/geometry")
        left_cube.GetSizeAttr().Set(1.0)
        left_cube.AddScaleOp().Set(Gf.Vec3f(depth, wall_thickness, height))
        left_cube.GetDisplayColorAttr().Set([Gf.Vec3f(0.5, 0.5, 0.5)])
        UsdPhysics.CollisionAPI.Apply(left_cube.GetPrim())

        # 右墙
        right_path = "/World/Fence/RightWall"
        right_xform = UsdGeom.Xform.Define(stage, right_path)
        right_pos = [center[0], center[1] + width / 2 + wall_thickness / 2, height / 2]
        right_xform.AddTranslateOp().Set(Gf.Vec3d(*right_pos))

        right_cube = UsdGeom.Cube.Define(stage, f"{right_path}/geometry")
        right_cube.GetSizeAttr().Set(1.0)
        right_cube.AddScaleOp().Set(Gf.Vec3f(depth, wall_thickness, height))
        right_cube.GetDisplayColorAttr().Set([Gf.Vec3f(0.5, 0.5, 0.5)])
        UsdPhysics.CollisionAPI.Apply(right_cube.GetPrim())

        print(f"Fence created at center {center}, size {width}x{depth}x{height}m")

    def setup_scene(self):
        """设置仿真场景（不含箱子，箱子稍后动态生成）"""
        # 创建世界
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        # 创建UR5基座 (15cm高)
        self._create_base(height=0.15, radius=0.1)

        # 创建三面围栏（让箱子聚集）
        self._create_fence(center=[0.45, 0.0], width=0.6, depth=0.5, height=0.25)

        # 加载UR5e机器人（放在基座上方）
        self.ur5 = UR5eController("ur5e")
        self.ur5.load_robot(
            prim_path="/World/UR5",
            position=[0.0, 0.0, 0.15]  # 基座高度
        )

        # 创建吸盘 (使用Isaac Sim SurfaceGripper)
        self.gripper = SuctionGripper(
            name="suction",
            parent_prim_path="/World/UR5/tool0",
            grip_threshold=0.03,  # 3cm内才能吸附（保留1cm余量）
            force_limit=1000.0,
            torque_limit=1000.0
        )

        # 初始化箱子生成器（稍后动态生成箱子）
        self.box_gen = BoxGenerator()

        print("Scene setup complete (without box)!")

    def spawn_box(self):
        """生成50个随机大小的纸箱从半空中掉落"""
        # 在围栏范围内生成纸箱（围栏宽0.6m，深0.5m）
        self.box_paths = self.box_gen.create_random_boxes(
            count=30,
            center=[0.45, 0.0],  # 围栏中心
            spread=0.20,  # 扩大散布范围
            drop_height=0.30,  # 从30cm高度开始掉落
            size_range=((0.05, 0.18), (0.04, 0.15), (0.03, 0.12)),  # 随机大小，上限增大
            mass_range=(0.3, 2.0)
        )
        print(f"Spawned {len(self.box_paths)} boxes into the fence")

    def get_box_position(self):
        """获取箱子当前位置"""
        stage = omni.usd.get_context().get_stage()
        box_prim = stage.GetPrimAtPath(self.box_prim_path)
        if box_prim.IsValid():
            xformable = UsdGeom.Xformable(box_prim)
            transform = xformable.ComputeLocalToWorldTransform(0)
            pos = transform.ExtractTranslation()
            return np.array([pos[0], pos[1], pos[2]])
        return None

    def run(self):
        """运行仿真"""
        self.setup_scene()

        # 重置世界
        self.world.reset()
        self.ur5.initialize()

        # 初始化吸盘 (必须在world.reset之后)
        self.gripper.initialize(self.world)

        # 设置UR5到直立姿态
        self.ur5.set_home_position()

        print("Starting pick and place simulation...")
        print("Step 1: UR5 moving to upright position...")

        # 主仿真循环
        while simulation_app.is_running():
            self.world.step(render=True)
            self.step_count += 1

            # 每步更新轨迹插值
            self.ur5.update_motion()

            # 更新吸盘状态
            self.gripper.update()

            # 状态机控制
            if self.state == "INIT":
                # 等待UR5到达直立姿态
                if self.step_count == self.home_steps:
                    print("UR5 is now upright.")
                    print(f"Step 2: Spawning 50 boxes at step {self.box_spawn_step}...")

                # 在指定时机生成箱子
                if self.step_count == self.box_spawn_step:
                    self.spawn_box()
                    print(f"Step 3: Waiting for boxes to fall and stabilize (until step {self.stable_steps})...")

                # 箱子稳定后开始抓取
                if self.step_count == self.stable_steps:
                    # 找到最上面的纸箱
                    topmost = self.box_gen.get_topmost_box()
                    if topmost is None:
                        print("Error: Cannot find any box")
                        continue

                    self.target_box_path, box_pos, self.target_box_size = topmost
                    box_height = self.target_box_size[2]

                    print(f"Found topmost box: {self.target_box_path}")
                    print(f"Box position: {box_pos}, size: {self.target_box_size}")

                    # 计算路径点（添加安全高度避障）
                    box_top_z = box_pos[2] + box_height / 2
                    safe_height = 0.50  # 安全高度
                    self.safe_pos = np.array([box_pos[0], box_pos[1], safe_height])
                    self.approach_pos = np.array([box_pos[0], box_pos[1], box_top_z + 0.10])
                    self.pick_pos = np.array([box_pos[0], box_pos[1], box_top_z + 0.02])
                    self.lift_pos = np.array([box_pos[0], box_pos[1], 0.40])  # 提升到40cm高度

                    print(f"Step 4: Moving to safe height {self.safe_pos}")
                    self.ur5.move_to_pose(self.safe_pos)
                    self.state = "SAFE_HEIGHT"

            elif self.state == "SAFE_HEIGHT" and self.ur5.is_motion_complete():
                ee_pos, _ = self.ur5.get_end_effector_pose()
                if ee_pos is not None:
                    print(f"Reached safe height. EE position: {ee_pos}")
                print(f"Step 5: Approaching box at {self.approach_pos}")
                self.ur5.move_to_pose(self.approach_pos)
                self.state = "APPROACH"

            elif self.state == "APPROACH" and self.ur5.is_motion_complete():
                ee_pos, _ = self.ur5.get_end_effector_pose()
                if ee_pos is not None:
                    print(f"Reached approach. EE position: {ee_pos}")
                print(f"Step 6: Moving to pick position {self.pick_pos}")
                self.ur5.move_to_pose(self.pick_pos)
                self.state = "PICK"

            elif self.state == "PICK" and self.ur5.is_motion_complete():
                ee_pos, _ = self.ur5.get_end_effector_pose()
                print(f"=== PICK位置对比 ===")
                print(f"目标位置: {self.pick_pos}")
                print(f"实际EE位置: {ee_pos}")
                print(f"====================")
                print("Step 7: Activating suction gripper")
                self.gripper.activate(self.target_box_path)
                self.state = "GRASP"
                self.grasp_delay = 0

            elif self.state == "GRASP":
                self.grasp_delay += 1
                if self.grasp_delay > 30:
                    print("Step 8: Lifting box")
                    self.ur5.move_to_pose(self.lift_pos)
                    self.state = "LIFT"

            elif self.state == "LIFT" and self.ur5.is_motion_complete():
                self.state = "DONE"
                print("Task complete! Box lifted successfully.")
                print("Simulation will close in 20 seconds...")

            elif self.state == "DONE":
                self.done_timer += 1
                if self.done_timer >= self.shutdown_delay:
                    print("20 seconds elapsed. Closing simulation...")
                    break

        simulation_app.close()


if __name__ == "__main__":
    sim = BoxPickSimulation()
    sim.run()
