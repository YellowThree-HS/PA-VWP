"""
纸箱生成器模块
在场景中创建可抓取的纸箱
适配Isaac Sim 5.1
"""

import numpy as np
from pxr import UsdGeom, UsdPhysics, Gf
import omni.usd


class BoxGenerator:
    """纸箱生成器"""

    def __init__(self):
        self._boxes = {}
        self._box_count = 0

    def create_box(
        self,
        name: str = None,
        position: list = None,
        size: list = None,
        mass: float = 1.0,
        color: list = None
    ) -> str:
        """
        创建一个纸箱

        Args:
            name: 纸箱名称
            position: 位置 [x, y, z]
            size: 尺寸 [长, 宽, 高]
            mass: 质量(kg)
            color: RGBA颜色

        Returns:
            str: 纸箱的prim路径
        """
        if name is None:
            name = f"box_{self._box_count}"
            self._box_count += 1

        if position is None:
            position = [0.5, 0.0, 0.1]
        if size is None:
            size = [0.3, 0.2, 0.15]
        if color is None:
            color = [0.6, 0.4, 0.2, 1.0]

        prim_path = f"/World/{name}"
        stage = omni.usd.get_context().get_stage()

        # 创建Xform作为纸箱根节点
        xform = UsdGeom.Xform.Define(stage, prim_path)
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))

        # 创建立方体几何
        cube_path = f"{prim_path}/geometry"
        cube = UsdGeom.Cube.Define(stage, cube_path)
        cube.GetSizeAttr().Set(1.0)

        # 设置缩放以匹配尺寸
        cube.AddScaleOp().Set(Gf.Vec3f(*size))

        # 设置颜色
        cube.GetDisplayColorAttr().Set([Gf.Vec3f(color[0], color[1], color[2])])

        # 添加刚体物理
        rigid_body = UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        rigid_body.CreateRigidBodyEnabledAttr(True)

        # 添加碰撞
        collision = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

        # 设置质量
        mass_api = UsdPhysics.MassAPI.Apply(xform.GetPrim())
        mass_api.CreateMassAttr(mass)

        # 保存纸箱信息
        self._boxes[name] = {
            "prim_path": prim_path,
            "position": position,
            "size": size,
            "mass": mass
        }

        print(f"Box created: {prim_path}")
        return prim_path

    def get_box_position(self, name: str) -> np.ndarray:
        """获取纸箱当前位置"""
        if name not in self._boxes:
            return None

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self._boxes[name]["prim_path"])

        if prim.IsValid():
            xformable = UsdGeom.Xformable(prim)
            transform = xformable.ComputeLocalToWorldTransform(0)
            pos = transform.ExtractTranslation()
            return np.array([pos[0], pos[1], pos[2]])
        return None

    @property
    def boxes(self) -> dict:
        """获取所有纸箱信息"""
        return self._boxes

    def get_box_prim_path(self, name: str) -> str:
        """获取纸箱的prim路径"""
        if name in self._boxes:
            return self._boxes[name]["prim_path"]
        return None
