"""
UR5e机器人控制器模块
负责UR5e机器人的加载、关节控制和运动规划
适配Isaac Sim 5.1
使用CuRobo进行无碰撞逆运动学求解
"""

from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
import numpy as np
from typing import List, Optional
from scipy.spatial.transform import Rotation as R

# CuRobo imports
import torch
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.types import WorldConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml


class UR5eController:
    """UR5e机器人控制器 - 使用CuRobo无碰撞IK"""

    def __init__(self, name: str = "ur5e"):
        self.name = name
        self.robot = None
        self.articulation_controller = None
        self._prim_path = None

        self._joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]

        # 轨迹插值相关
        self._target_joint_positions = None
        self._start_joint_positions = None
        self._interpolation_progress = 1.0  # 1.0表示已完成
        self._interpolation_speed = 0.02    # 每步插值进度

        # 基座高度 (会在load_robot时更新)
        self._base_height = 0.0
        self._base_position = [0.0, 0.0, 0.0]

        # CuRobo IK求解器
        self._ik_solver = None
        self._tensor_args = TensorDeviceType(device=torch.device("cuda:0"), dtype=torch.float32)

    def load_robot(
        self,
        prim_path: str = "/World/UR5",
        usd_path: str = None,
        position: List[float] = None,
        orientation: List[float] = None,
        initial_joint_positions: List[float] = None
    ) -> SingleArticulation:
        """
        加载UR5e机器人到场景

        Args:
            prim_path: USD场景中的路径
            usd_path: UR5e USD文件路径
            position: 机器人基座位置 [x, y, z]
            orientation: 机器人基座朝向 [w, x, y, z]
            initial_joint_positions: 初始关节角度 [6个值]

        Returns:
            SingleArticulation: 加载的机器人对象
        """
        from isaacsim.core.utils.nucleus import get_assets_root_path

        if usd_path is None:
            assets_root = get_assets_root_path()
            usd_path = assets_root + "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"

        if position is None:
            position = [0.0, 0.0, 0.0]
        if orientation is None:
            orientation = [1.0, 0.0, 0.0, 0.0]

        # 默认竖直向上姿态
        # shoulder_lift = -90度 使上臂竖直向上
        # elbow = 0 保持前臂与上臂同向
        # wrist1 = -90度 使末端朝上
        if initial_joint_positions is None:
            initial_joint_positions = [0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0]

        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

        # 保存路径和位置信息
        self._prim_path = prim_path
        self._base_position = position
        self._base_height = position[2]

        # 设置机器人基座位置
        self._set_robot_position(prim_path, position)

        # 设置初始关节角度 (在创建Articulation之前)
        self._set_initial_joint_angles(prim_path, initial_joint_positions)

        self.robot = SingleArticulation(
            prim_path=prim_path,
            name=self.name
        )

        return self.robot

    def _set_robot_position(self, prim_path: str, position: List[float]):
        """设置机器人基座位置"""
        import omni.usd
        from pxr import UsdGeom, Gf

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)

        if prim.IsValid():
            xform = UsdGeom.Xformable(prim)
            # 清除现有变换操作
            xform.ClearXformOpOrder()
            # 添加平移
            xform.AddTranslateOp().Set(Gf.Vec3d(position[0], position[1], position[2]))

    def _set_initial_joint_angles(self, prim_path: str, joint_positions: List[float]):
        """在仿真开始前设置初始关节角度"""
        import omni.usd
        from pxr import UsdPhysics

        stage = omni.usd.get_context().get_stage()

        for i, joint_name in enumerate(self._joint_names):
            joint_prim_path = f"{prim_path}/{joint_name}"
            joint_prim = stage.GetPrimAtPath(joint_prim_path)

            if joint_prim.IsValid():
                # 设置关节的初始位置
                drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
                if drive:
                    drive.GetTargetPositionAttr().Set(np.degrees(joint_positions[i]))

    def initialize(self):
        """初始化机器人控制器和CuRobo IK求解器"""
        if self.robot is None:
            raise RuntimeError("Robot not loaded. Call load_robot() first.")
        self.robot.initialize()
        self.articulation_controller = self.robot.get_articulation_controller()

        # 初始化CuRobo IK求解器
        self._init_curobo_ik()
        print("[IK] Using CuRobo collision-free IK solver for UR5e")

    def _init_curobo_ik(self):
        """初始化CuRobo无碰撞IK求解器"""
        # 加载UR5e机器人配置
        robot_cfg_path = join_path(get_robot_configs_path(), "ur5e.yml")
        robot_cfg = load_yaml(robot_cfg_path)["robot_cfg"]

        # 创建世界配置（可添加障碍物）
        world_cfg = WorldConfig.from_dict({
            "cuboid": {
                "ground": {
                    "dims": [2.0, 2.0, 0.01],
                    "pose": [0.0, 0.0, -0.005, 1.0, 0.0, 0.0, 0.0]
                }
            }
        })

        # 配置IK求解器
        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=20,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self._tensor_args,
            use_cuda_graph=True,
        )

        self._ik_solver = IKSolver(ik_config)
        print("[CuRobo] IK solver initialized with collision checking")

    def set_home_position(self):
        """设置UR5e到竖直向上的初始姿态"""
        home_joints = np.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0])
        self.set_joint_positions(home_joints)

    def get_joint_positions(self) -> np.ndarray:
        """获取当前关节位置"""
        return self.robot.get_joint_positions()

    def get_joint_velocities(self) -> np.ndarray:
        """获取当前关节速度"""
        return self.robot.get_joint_velocities()

    def set_joint_positions(self, positions: np.ndarray):
        """
        设置目标关节位置

        Args:
            positions: 6个关节的目标位置(弧度)
        """
        self.robot.set_joint_positions(positions)

    def get_end_effector_pose(self) -> tuple:
        """
        获取末端执行器位姿

        Returns:
            tuple: (position, orientation) 位置和四元数
        """
        from pxr import UsdGeom
        import omni.usd

        stage = omni.usd.get_context().get_stage()

        # 尝试多个可能的路径
        possible_paths = [
            "/World/UR5/tool0",
            "/World/UR5/ee_link",
            "/World/UR5/wrist_3_link"
        ]

        for path in possible_paths:
            ee_prim = stage.GetPrimAtPath(path)
            if ee_prim.IsValid():
                xformable = UsdGeom.Xformable(ee_prim)
                transform = xformable.ComputeLocalToWorldTransform(0)
                pos = transform.ExtractTranslation()
                rot = transform.ExtractRotation()
                print(f"[EE] Found at {path}: pos={[pos[0], pos[1], pos[2]]}")
                return np.array([pos[0], pos[1], pos[2]]), rot

        print(f"[EE] No valid end effector found!")
        return None, None

    def move_to_pose(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray = None
    ) -> np.ndarray:
        """
        移动到目标位姿（使用IK求解和轨迹插值）

        Args:
            target_position: 目标位置 [x, y, z]
            target_orientation: 目标朝向四元数 [w, x, y, z]

        Returns:
            np.ndarray: 目标关节角度
        """
        # 使用IK求解目标关节角度
        joint_pos = self._solve_ik(target_position, target_orientation)

        # 启动插值运动
        self._start_joint_positions = self.get_joint_positions().copy()
        self._target_joint_positions = joint_pos
        self._interpolation_progress = 0.0

        return joint_pos

    def _solve_ik(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray = None
    ) -> np.ndarray:
        """
        使用CuRobo求解无碰撞逆运动学

        Args:
            target_position: 目标位置 [x, y, z] (世界坐标)
            target_orientation: 目标朝向四元数 [w, x, y, z]，默认末端朝下

        Returns:
            np.ndarray: 6个关节角度
        """
        print(f"[CuRobo IK] Target position (world): {target_position}")

        # 默认末端朝下，保持当前rz不变
        if target_orientation is None:
            # 获取当前末端姿态，提取rz
            _, current_rot = self.get_end_effector_pose()
            current_rz = current_rot.GetAxis()[2] * current_rot.GetAngle() * np.pi / 180.0

            # 构造四元数: rx=180°, ry=0°, rz=当前值 (外旋XYZ)
            rx, ry = np.pi, 0.0
            # 四元数 = Rz * Ry * Rx (外旋等价于内旋逆序)
            cx, sx = np.cos(rx/2), np.sin(rx/2)
            cy, sy = np.cos(ry/2), np.sin(ry/2)
            cz, sz = np.cos(current_rz/2), np.sin(current_rz/2)

            # ZYX内旋顺序计算四元数 [w, x, y, z]
            w = cx*cy*cz + sx*sy*sz
            x = sx*cy*cz - cx*sy*sz
            y = cx*sy*cz + sx*cy*sz
            z = cx*cy*sz - sx*sy*cz
            target_orientation = np.array([w, x, y, z])

        # 转换为相对于机器人基座的坐标
        rel_pos = target_position - np.array(self._base_position)

        # 创建CuRobo Pose目标
        device = self._tensor_args.device
        goal_pose = Pose(
            position=torch.tensor(np.array([rel_pos]), dtype=torch.float32, device=device),
            quaternion=torch.tensor(np.array([target_orientation]), dtype=torch.float32, device=device)
        )

        # 使用当前关节位置作为seed，让IK找到构型变化最小的解
        current_joints = self.get_joint_positions()
        seed_config = torch.tensor(
            current_joints.reshape(1, 1, -1),
            dtype=torch.float32,
            device=device
        )

        # 求解IK
        result = self._ik_solver.solve_single(goal_pose, seed_config=seed_config)

        if result.success.item():
            joint_positions = result.solution[0].cpu().numpy()
            print(f"[CuRobo IK] Solution found: {joint_positions}")
            return joint_positions
        else:
            print("[CuRobo IK] Warning: No collision-free solution found, using best effort")
            joint_positions = result.solution[0].cpu().numpy()
            return joint_positions

    def update_motion(self):
        """
        更新轨迹插值，每个仿真步调用一次
        使用平滑的S曲线插值
        """
        if self._interpolation_progress >= 1.0:
            return

        # 更新插值进度
        self._interpolation_progress += self._interpolation_speed
        self._interpolation_progress = min(self._interpolation_progress, 1.0)

        # S曲线平滑 (smoothstep)
        t = self._interpolation_progress
        smooth_t = t * t * (3 - 2 * t)

        # 线性插值关节角度
        current_pos = (
            self._start_joint_positions * (1 - smooth_t) +
            self._target_joint_positions * smooth_t
        )

        self.set_joint_positions(current_pos)

    def is_motion_complete(self) -> bool:
        """检查运动是否完成"""
        return self._interpolation_progress >= 1.0

    def set_interpolation_speed(self, speed: float):
        """设置插值速度 (0.01-0.1)"""
        self._interpolation_speed = np.clip(speed, 0.01, 0.1)
