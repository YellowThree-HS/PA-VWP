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

    def create_random_boxes(
        self,
        count: int = 10,
        center: list = None,
        spread: float = 0.3,
        drop_height: float = 0.5,
        size_range: tuple = ((0.08, 0.15), (0.06, 0.12), (0.05, 0.10)),
        mass_range: tuple = (0.5, 2.0)
    ) -> list:
        """
        批量生成随机大小的纸箱从半空中掉落

        Args:
            count: 纸箱数量
            center: 掉落中心位置 [x, y]
            spread: 水平散布范围
            drop_height: 掉落起始高度
            size_range: 尺寸范围 ((长min,长max), (宽min,宽max), (高min,高max))
            mass_range: 质量范围 (min, max)

        Returns:
            list: 所有纸箱的prim路径列表
        """
        if center is None:
            center = [0.4, 0.0]

        box_paths = []
        colors = [
            [0.6, 0.4, 0.2, 1.0],   # 棕色
            [0.7, 0.5, 0.3, 1.0],   # 浅棕色
            [0.5, 0.35, 0.2, 1.0],  # 深棕色
            [0.65, 0.45, 0.25, 1.0], # 中棕色
        ]

        for i in range(count):
            # 随机尺寸
            size = [
                np.random.uniform(size_range[0][0], size_range[0][1]),
                np.random.uniform(size_range[1][0], size_range[1][1]),
                np.random.uniform(size_range[2][0], size_range[2][1])
            ]

            # 随机位置（在中心点周围散布，高度递增避免初始碰撞）
            pos_x = center[0] + np.random.uniform(-spread, spread)
            pos_y = center[1] + np.random.uniform(-spread, spread)
            pos_z = drop_height + i * 0.05 + size[2] / 2  # 递增高度（每个+5cm）

            # 随机质量
            mass = np.random.uniform(mass_range[0], mass_range[1])

            # 随机颜色
            color = colors[i % len(colors)]

            # 创建纸箱
            path = self.create_box(
                name=f"box_{i}",
                position=[pos_x, pos_y, pos_z],
                size=size,
                mass=mass,
                color=color
            )
            box_paths.append(path)

        print(f"Created {count} random boxes dropping from height {drop_height}m")
        return box_paths

    def get_topmost_box(self, exclude_paths: list = None) -> tuple:
        """
        获取最上面的纸箱（Z坐标最高）

        Args:
            exclude_paths: 要排除的箱子路径列表

        Returns:
            tuple: (prim_path, position, size) 或 None
        """
        if not self._boxes:
            return None

        if exclude_paths is None:
            exclude_paths = []

        topmost = None
        max_z = -float('inf')

        stage = omni.usd.get_context().get_stage()

        for name, info in self._boxes.items():
            if info["prim_path"] in exclude_paths:
                continue
            prim = stage.GetPrimAtPath(info["prim_path"])
            if prim.IsValid():
                xformable = UsdGeom.Xformable(prim)
                transform = xformable.ComputeLocalToWorldTransform(0)
                pos = transform.ExtractTranslation()
                # 计算箱子顶面高度
                top_z = pos[2] + info["size"][2] / 2
                if top_z > max_z:
                    max_z = top_z
                    topmost = (
                        info["prim_path"],
                        np.array([pos[0], pos[1], pos[2]]),
                        info["size"]
                    )

        return topmost

    def get_unstable_graspable_box(self, exclude_paths: list = None) -> tuple:
        """
        获取边缘支撑箱子（抓取后会导致其他箱子倒塌）
        返回箱子信息和最佳吸取方向

        Args:
            exclude_paths: 要排除的箱子路径列表

        Returns:
            tuple: (prim_path, position, size, grasp_info) 或 None
            grasp_info: {"direction": "top"/"side", "approach_vector": [x,y,z]}
        """
        if not self._boxes:
            return None

        if exclude_paths is None:
            exclude_paths = []

        stage = omni.usd.get_context().get_stage()

        # 收集所有箱子信息
        box_infos = []
        for name, info in self._boxes.items():
            if info["prim_path"] in exclude_paths:
                continue
            prim = stage.GetPrimAtPath(info["prim_path"])
            if prim.IsValid():
                xformable = UsdGeom.Xformable(prim)
                transform = xformable.ComputeLocalToWorldTransform(0)
                pos = transform.ExtractTranslation()
                box_infos.append({
                    "prim_path": info["prim_path"],
                    "pos": np.array([pos[0], pos[1], pos[2]]),
                    "size": info["size"],
                    "top_z": pos[2] + info["size"][2] / 2,
                    "bottom_z": pos[2] - info["size"][2] / 2
                })

        if not box_infos:
            return None

        # 分析每个箱子的可抓取性和不稳定性
        candidates = []
        for box in box_infos:
            analysis = self._analyze_box_graspability(box, box_infos)
            if analysis["has_box_above"] and analysis["grasp_direction"]:
                candidates.append({
                    **box,
                    **analysis
                })

        if not candidates:
            # 没找到理想的，返回最底部有暴露面的箱子
            return self._get_fallback_box(box_infos)

        # 优先选择支撑箱子数量多的（倒塌效果更明显）
        candidates.sort(key=lambda x: x["boxes_above_count"], reverse=True)
        best = candidates[0]

        grasp_info = {
            "direction": best["grasp_direction"],
            "approach_vector": best["approach_vector"],
            "grasp_point": best["grasp_point"]
        }

        return (best["prim_path"], best["pos"], best["size"], grasp_info)

    def _analyze_box_graspability(self, box: dict, all_boxes: list) -> dict:
        """
        分析单个箱子的可抓取性
        检查顶面和四个侧面是否暴露，以及上方是否有箱子
        """
        result = {
            "has_box_above": False,
            "boxes_above_count": 0,
            "grasp_direction": None,
            "approach_vector": None,
            "grasp_point": None,
            "top_exposed": True,
            "side_exposed": {"front": True, "back": True, "left": True, "right": True}
        }

        box_pos = box["pos"]
        box_size = box["size"]
        half_size = np.array(box_size) / 2

        # 检查每个其他箱子
        for other in all_boxes:
            if other["prim_path"] == box["prim_path"]:
                continue

            other_pos = other["pos"]
            other_size = other["size"]
            other_half = np.array(other_size) / 2

            # 检查是否在上方（有重叠）
            if other_pos[2] > box_pos[2]:
                dx = abs(other_pos[0] - box_pos[0])
                dy = abs(other_pos[1] - box_pos[1])
                overlap_x = half_size[0] + other_half[0]
                overlap_y = half_size[1] + other_half[1]

                if dx < overlap_x * 0.8 and dy < overlap_y * 0.8:
                    result["has_box_above"] = True
                    result["boxes_above_count"] += 1
                    # 顶面被遮挡
                    if dx < half_size[0] and dy < half_size[1]:
                        result["top_exposed"] = False

            # 检查侧面遮挡（同一高度层的箱子）
            z_overlap = (abs(other_pos[2] - box_pos[2]) <
                        (half_size[2] + other_half[2]) * 0.8)
            if z_overlap:
                # front (x+方向)
                if (other_pos[0] > box_pos[0] and
                    other_pos[0] - box_pos[0] < half_size[0] + other_half[0] + 0.05):
                    if abs(other_pos[1] - box_pos[1]) < half_size[1] + other_half[1]:
                        result["side_exposed"]["front"] = False
                # back (x-方向)
                if (other_pos[0] < box_pos[0] and
                    box_pos[0] - other_pos[0] < half_size[0] + other_half[0] + 0.05):
                    if abs(other_pos[1] - box_pos[1]) < half_size[1] + other_half[1]:
                        result["side_exposed"]["back"] = False
                # right (y+方向)
                if (other_pos[1] > box_pos[1] and
                    other_pos[1] - box_pos[1] < half_size[1] + other_half[1] + 0.05):
                    if abs(other_pos[0] - box_pos[0]) < half_size[0] + other_half[0]:
                        result["side_exposed"]["right"] = False
                # left (y-方向)
                if (other_pos[1] < box_pos[1] and
                    box_pos[1] - other_pos[1] < half_size[1] + other_half[1] + 0.05):
                    if abs(other_pos[0] - box_pos[0]) < half_size[0] + other_half[0]:
                        result["side_exposed"]["left"] = False

        # 确定最佳抓取方向
        if result["top_exposed"]:
            result["grasp_direction"] = "top"
            result["approach_vector"] = np.array([0, 0, -1])
            result["grasp_point"] = np.array([
                box_pos[0], box_pos[1], box_pos[2] + half_size[2]
            ])
        else:
            # 选择暴露的侧面，优先选择朝向机械臂的方向(back, x-方向)
            side_priority = ["back", "left", "right", "front"]
            side_vectors = {
                "front": np.array([1, 0, 0]),
                "back": np.array([-1, 0, 0]),
                "left": np.array([0, -1, 0]),
                "right": np.array([0, 1, 0])
            }
            side_offsets = {
                "front": np.array([half_size[0], 0, 0]),
                "back": np.array([-half_size[0], 0, 0]),
                "left": np.array([0, -half_size[1], 0]),
                "right": np.array([0, half_size[1], 0])
            }

            for side in side_priority:
                if result["side_exposed"][side]:
                    result["grasp_direction"] = f"side_{side}"
                    result["approach_vector"] = -side_vectors[side]
                    result["grasp_point"] = box_pos + side_offsets[side]
                    break

        return result

    def _get_fallback_box(self, box_infos: list) -> tuple:
        """当没有理想候选时，返回一个有暴露面的箱子"""
        for box in sorted(box_infos, key=lambda x: x["top_z"]):
            analysis = self._analyze_box_graspability(box, box_infos)
            if analysis["grasp_direction"]:
                grasp_info = {
                    "direction": analysis["grasp_direction"],
                    "approach_vector": analysis["approach_vector"],
                    "grasp_point": analysis["grasp_point"]
                }
                return (box["prim_path"], box["pos"], box["size"], grasp_info)
        return None
