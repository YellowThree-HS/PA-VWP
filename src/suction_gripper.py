"""
吸盘末端执行器模块
使用FixedJoint + 距离检测实现真空吸盘功能
适配Isaac Sim 5.1
"""

import numpy as np
from pxr import UsdPhysics, Gf, UsdGeom
import omni.usd


class SuctionGripper:
    """真空吸盘控制器 - 基于FixedJoint实现"""

    def __init__(
        self,
        name: str = "suction_gripper",
        parent_prim_path: str = "/World/UR5/tool0",
        grip_threshold: float = 0.05,
        force_limit: float = 1000.0,
        torque_limit: float = 1000.0
    ):
        """
        初始化吸盘

        Args:
            name: 吸盘名称
            parent_prim_path: 末端执行器prim路径
            grip_threshold: 吸附距离阈值(米)，物体必须在此距离内才能被吸附
            force_limit: 吸附力限制(N) - 暂未使用
            torque_limit: 扭矩限制(Nm) - 暂未使用
        """
        self.name = name
        self.parent_prim_path = parent_prim_path
        # 用于获取位置的备选路径
        self.ee_prim_path = "/World/UR5/wrist_3_link"
        self.grip_threshold = grip_threshold
        self.force_limit = force_limit
        self.torque_limit = torque_limit

        self._is_closed = False
        self._attached_object = None
        self._fixed_joint_path = None

    def initialize(self, world):
        """初始化吸盘 (兼容接口)"""
        print(f"SuctionGripper initialized: threshold={self.grip_threshold}m")

    def get_gripper_position(self) -> np.ndarray:
        """获取吸盘当前世界坐标位置"""
        stage = omni.usd.get_context().get_stage()

        # 尝试多个可能的路径
        paths_to_try = [self.parent_prim_path, self.ee_prim_path]

        for path in paths_to_try:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                xformable = UsdGeom.Xformable(prim)
                transform = xformable.ComputeLocalToWorldTransform(0)
                pos = transform.ExtractTranslation()
                if pos[0] != 0 or pos[1] != 0 or pos[2] != 0:
                    return np.array([pos[0], pos[1], pos[2]])

        print("Warning: Could not get gripper position")
        return np.zeros(3)

    def _get_object_position(self, prim_path: str) -> np.ndarray:
        """获取物体位置"""
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            xformable = UsdGeom.Xformable(prim)
            transform = xformable.ComputeLocalToWorldTransform(0)
            pos = transform.ExtractTranslation()
            return np.array([pos[0], pos[1], pos[2]])
        return None

    def _get_object_top_z(self, target_prim_path: str) -> float:
        """获取物体顶面的Z坐标"""
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(target_prim_path)
        if not prim.IsValid():
            return None

        # 获取物体中心位置
        xformable = UsdGeom.Xformable(prim)
        transform = xformable.ComputeLocalToWorldTransform(0)
        center_pos = transform.ExtractTranslation()

        # 获取几何体尺寸（查找geometry子节点）
        geom_path = f"{target_prim_path}/geometry"
        geom_prim = stage.GetPrimAtPath(geom_path)
        if geom_prim.IsValid():
            # 尝试获取缩放（箱子使用Cube + Scale实现尺寸）
            geom_xform = UsdGeom.Xformable(geom_prim)
            scale = geom_xform.GetOrderedXformOps()
            for op in scale:
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    s = op.Get()
                    # s[2]是高度，顶面 = 中心Z + 高度/2
                    return center_pos[2] + s[2] / 2

        # 如果无法获取尺寸，返回中心位置
        return center_pos[2]

    def _disable_collision_between(self, stage, prim_path_a: str, prim_path_b: str):
        """禁用两个物体之间的碰撞"""
        # 使用 FilteredPairsAPI 禁用特定物体对之间的碰撞
        physics_scene_path = "/World/physicsScene"
        physics_scene = stage.GetPrimAtPath(physics_scene_path)

        if not physics_scene.IsValid():
            # 尝试查找场景中的 PhysicsScene
            for prim in stage.Traverse():
                if prim.GetTypeName() == "PhysicsScene":
                    physics_scene = prim
                    break

        if physics_scene.IsValid():
            from pxr import PhysxSchema
            # 应用 FilteredPairsAPI
            if not physics_scene.HasAPI(PhysxSchema.PhysxSceneAPI):
                PhysxSchema.PhysxSceneAPI.Apply(physics_scene)

            # 创建碰撞过滤对
            filtered_pairs_api = UsdPhysics.FilteredPairsAPI.Apply(physics_scene)
            filtered_pairs_api.GetFilteredPairsRel().AddTarget(prim_path_a)
            filtered_pairs_api.GetFilteredPairsRel().AddTarget(prim_path_b)
            print(f"  Collision disabled between {prim_path_a} and {prim_path_b}")
        else:
            print("  Warning: PhysicsScene not found, collision filtering skipped")

    def _calculate_distance(self, target_prim_path: str) -> float:
        """计算吸盘到目标物体顶面的垂直距离"""
        gripper_pos = self.get_gripper_position()
        target_pos = self._get_object_position(target_prim_path)
        if target_pos is None:
            return float('inf')

        # 获取箱子顶面Z坐标
        top_z = self._get_object_top_z(target_prim_path)
        if top_z is None:
            top_z = target_pos[2]

        # 计算水平距离和垂直距离（到顶面）
        horizontal_dist = np.sqrt(
            (gripper_pos[0] - target_pos[0])**2 +
            (gripper_pos[1] - target_pos[1])**2
        )
        vertical_dist = abs(gripper_pos[2] - top_z)

        print(f"  Gripper pos: {gripper_pos}")
        print(f"  Target center: {target_pos}, top_z: {top_z:.4f}")
        print(f"  Horizontal dist: {horizontal_dist:.4f}m, Vertical dist to top: {vertical_dist:.4f}m")

        # 主要检查垂直距离（吸盘朝下，到顶面的距离）
        return vertical_dist

    def activate(self, target_prim_path: str) -> bool:
        """
        激活吸盘，吸附目标物体（带距离检测）

        Args:
            target_prim_path: 要吸附的物体路径

        Returns:
            bool: 是否成功吸附
        """
        if self._is_closed:
            print("Gripper already closed")
            return False

        # 检查距离
        distance = self._calculate_distance(target_prim_path)
        print(f"Distance to target: {distance:.4f}m (threshold: {self.grip_threshold}m)")

        if distance > self.grip_threshold:
            print(f"Target too far! Distance {distance:.3f}m > threshold {self.grip_threshold}m")
            return False

        stage = omni.usd.get_context().get_stage()
        target_prim = stage.GetPrimAtPath(target_prim_path)

        if not target_prim.IsValid():
            print(f"Target prim not found: {target_prim_path}")
            return False

        # 禁用末端和箱子之间的碰撞
        self._disable_collision_between(stage, self.ee_prim_path, target_prim_path)

        # 创建固定关节连接吸盘和物体
        body0_path = self.ee_prim_path
        self._fixed_joint_path = f"{body0_path}/suction_joint"
        joint_prim = stage.DefinePrim(self._fixed_joint_path, "PhysicsFixedJoint")

        # 设置关节的两个连接体
        fixed_joint = UsdPhysics.FixedJoint(joint_prim)
        fixed_joint.CreateBody0Rel().SetTargets([body0_path])
        fixed_joint.CreateBody1Rel().SetTargets([target_prim_path])

        # 获取 wrist_3_link 的世界变换，用于计算正确的局部坐标偏移
        gripper_prim = stage.GetPrimAtPath(body0_path)
        gripper_xform = UsdGeom.Xformable(gripper_prim)
        gripper_world_transform = gripper_xform.ComputeLocalToWorldTransform(0)
        gripper_world_inv = gripper_world_transform.GetInverse()

        # 获取箱子的世界变换
        target_xform = UsdGeom.Xformable(target_prim)
        target_world_transform = target_xform.ComputeLocalToWorldTransform(0)

        # 获取箱子世界位置
        target_pos = self._get_object_position(target_prim_path)
        target_world_pos = Gf.Vec3d(target_pos[0], target_pos[1], target_pos[2])

        # 将箱子世界位置转换到 wrist_3_link 局部坐标系
        local_pos = gripper_world_inv.Transform(target_world_pos)

        # 计算相对旋转: gripper_world_inv * target_world_rot
        gripper_rot = gripper_world_transform.ExtractRotation().GetQuat()
        target_rot = target_world_transform.ExtractRotation().GetQuat()
        # 相对旋转 = gripper^-1 * target
        gripper_rot_inv = gripper_rot.GetConjugate()
        rel_rot = gripper_rot_inv * target_rot
        rel_rot_normalized = rel_rot.GetNormalized()

        # 设置局部坐标偏移
        fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(local_pos[0], local_pos[1], local_pos[2]))
        fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))

        # 设置相对旋转，保持箱子当前朝向
        fixed_joint.CreateLocalRot0Attr().Set(Gf.Quatf(
            rel_rot_normalized.GetReal(),
            rel_rot_normalized.GetImaginary()[0],
            rel_rot_normalized.GetImaginary()[1],
            rel_rot_normalized.GetImaginary()[2]
        ))
        fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

        self._attached_object = target_prim_path
        self._is_closed = True

        print(f"Suction activated: attached to {target_prim_path}")
        return True

    def deactivate(self) -> bool:
        """
        释放吸盘，放开物体

        Returns:
            bool: 是否成功释放
        """
        if not self._is_closed:
            return False

        stage = omni.usd.get_context().get_stage()

        # 删除固定关节
        if self._fixed_joint_path:
            joint_prim = stage.GetPrimAtPath(self._fixed_joint_path)
            if joint_prim.IsValid():
                stage.RemovePrim(self._fixed_joint_path)

        self._fixed_joint_path = None
        self._attached_object = None
        self._is_closed = False

        print("Suction deactivated")
        return True

    def update(self):
        """每帧更新 (兼容接口)"""
        pass

    def close(self) -> bool:
        """激活吸盘 (兼容接口，需要先设置target)"""
        return self._is_closed

    def open(self) -> bool:
        """释放吸盘 (兼容接口)"""
        return self.deactivate()

    @property
    def is_closed(self) -> bool:
        """吸盘是否处于吸附状态"""
        return self._is_closed

    @property
    def attached_object(self) -> str:
        """当前吸附的物体路径"""
        return self._attached_object
