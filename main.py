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
        self.stable_steps = 100  # 箱子落地稳定所需步数（box_spawn_step + 100）

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

    def setup_scene(self):
        """设置仿真场景（不含箱子，箱子稍后动态生成）"""
        # 创建世界
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()

        # 创建UR5基座 (15cm高)
        self._create_base(height=0.15, radius=0.1)

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
        """在空中10cm处生成箱子"""
        box_size = [0.15, 0.10, 0.08]  # 更小的箱子
        # 箱子底部距地面10cm，所以中心高度 = 0.1 + 高度/2
        spawn_height = 0.1 + box_size[2] / 2

        # 箱子位置：在UR5前方0.4m处（工作范围内）
        self.box_prim_path = self.box_gen.create_box(
            name="cardboard_box",
            position=[0.4, 0.0, spawn_height],
            size=box_size,
            mass=1.0,
            color=[0.6, 0.4, 0.2, 1.0]
        )
        print(f"Box spawned at height {spawn_height}m (10cm above ground)")

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

        # 箱子参数
        box_height = 0.08  # 与spawn_box中的box_size[2]一致

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
                    print(f"Step 2: Spawning box at 10cm height at step {self.box_spawn_step}...")

                # 在指定时机生成箱子
                if self.step_count == self.box_spawn_step:
                    self.spawn_box()
                    print(f"Step 3: Waiting for box to fall and stabilize (until step {self.stable_steps})...")

                # 箱子稳定后开始抓取
                if self.step_count == self.stable_steps:
                    box_pos = self.get_box_position()
                    if box_pos is None:
                        print("Error: Cannot get box position")
                        continue

                    print(f"Box stabilized at position: {box_pos}")

                    # 计算路径点（添加安全高度避障）
                    # 注意: box_pos是箱子中心位置，箱子顶面 = box_pos[2] + box_height/2
                    box_top_z = box_pos[2] + box_height / 2
                    safe_height = 0.45  # 安全高度（降低以避免IK翻转）
                    self.safe_pos = np.array([box_pos[0], box_pos[1], safe_height])
                    self.approach_pos = np.array([box_pos[0], box_pos[1], box_top_z + 0.10])
                    self.pick_pos = np.array([box_pos[0], box_pos[1], box_top_z + 0.02])  # 顶面上方2cm
                    self.lift_pos = np.array([box_pos[0], box_pos[1], 0.25])  # 抬升高度降低

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
                self.gripper.activate(self.box_prim_path)
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
