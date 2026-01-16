"""
纸箱生成器模块
在场景中创建可抓取的纸箱
适配Isaac Sim 5.1
"""

import numpy as np
from pathlib import Path
from pxr import UsdGeom, UsdPhysics, Gf, Semantics, UsdShade, Sdf, PhysxSchema
import omni.usd


class BoxGenerator:
    """纸箱生成器"""

    def __init__(self, texture_dir: str = None):
        self._boxes = {}
        self._box_count = 0
        self._textures = []

        # 加载纹理文件列表
        if texture_dir is None:
            # 默认纹理目录
            texture_dir = Path(__file__).parent.parent / "assets" / "cardboard_textures"
        else:
            texture_dir = Path(texture_dir)

        if texture_dir.exists():
            extensions = {'.jpg', '.jpeg', '.png'}
            self._textures = sorted([
                str(f.absolute()) for f in texture_dir.iterdir()
                if f.suffix.lower() in extensions
            ])

    def _create_beveled_box_mesh(self, size: list, bevel_radius: float = 0.005):
        """
        创建带倒角的箱子网格数据

        Args:
            size: 箱子尺寸 [长, 宽, 高]
            bevel_radius: 倒角半径 (单位：米，默认5mm)

        Returns:
            tuple: (points, normals, face_vertex_counts, face_vertex_indices, uvs, uv_indices)
        """
        # 半尺寸
        hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
        r = bevel_radius

        # 内边界（倒角后的平面边界）
        bx, by, bz = hx - r, hy - r, hz - r

        # 24个顶点：每个角切成3个顶点
        points = [
            # 角0: (-x, -y, -z)
            Gf.Vec3f(-bx, -hy, -hz),  # 0: 沿X方向
            Gf.Vec3f(-hx, -by, -hz),  # 1: 沿Y方向
            Gf.Vec3f(-hx, -hy, -bz),  # 2: 沿Z方向
            # 角1: (+x, -y, -z)
            Gf.Vec3f(bx, -hy, -hz),   # 3
            Gf.Vec3f(hx, -by, -hz),   # 4
            Gf.Vec3f(hx, -hy, -bz),   # 5
            # 角2: (+x, +y, -z)
            Gf.Vec3f(hx, by, -hz),    # 6
            Gf.Vec3f(bx, hy, -hz),    # 7
            Gf.Vec3f(hx, hy, -bz),    # 8
            # 角3: (-x, +y, -z)
            Gf.Vec3f(-bx, hy, -hz),   # 9
            Gf.Vec3f(-hx, by, -hz),   # 10
            Gf.Vec3f(-hx, hy, -bz),   # 11
            # 角4: (-x, -y, +z)
            Gf.Vec3f(-bx, -hy, hz),   # 12
            Gf.Vec3f(-hx, -by, hz),   # 13
            Gf.Vec3f(-hx, -hy, bz),   # 14
            # 角5: (+x, -y, +z)
            Gf.Vec3f(bx, -hy, hz),    # 15
            Gf.Vec3f(hx, -by, hz),    # 16
            Gf.Vec3f(hx, -hy, bz),    # 17
            # 角6: (+x, +y, +z)
            Gf.Vec3f(hx, by, hz),     # 18
            Gf.Vec3f(bx, hy, hz),     # 19
            Gf.Vec3f(hx, hy, bz),     # 20
            # 角7: (-x, +y, +z)
            Gf.Vec3f(-bx, hy, hz),    # 21
            Gf.Vec3f(-hx, by, hz),    # 22
            Gf.Vec3f(-hx, hy, bz),    # 23
        ]

        # 面定义（顶点索引，从外部看逆时针为正面）
        faces = [
            # 6个主面 (四边形) - 从外部看逆时针
            [3, 7, 9, 0],       # 底面 -Z (从下往上看)
            [12, 21, 19, 15],   # 顶面 +Z (从上往下看)
            [1, 13, 14, 2],     # 左面 -X (从左往右看)
            [16, 18, 6, 4],     # 右面 +X (从右往左看)
            [13, 1, 4, 16],     # 前面 -Y (从前往后看)
            [22, 23, 11, 10],   # 后面 +Y (从后往前看)

            # 12条边的倒角面 (四边形)
            # 底面Z-的4条边 (斜面朝外)
            [1, 4, 3, 0],       # 底-前
            [4, 6, 7, 3],       # 底-右
            [6, 10, 9, 7],      # 底-后
            [10, 1, 0, 9],      # 底-左
            # 顶面Z+的4条边
            [15, 16, 13, 12],   # 顶-前
            [19, 18, 16, 15],   # 顶-右
            [21, 22, 18, 19],   # 顶-后
            [12, 13, 22, 21],   # 顶-左
            # 竖直4条边
            [14, 17, 5, 2],     # 前左 (-Y, -X)
            [17, 20, 8, 5],     # 前右 (-Y, +X)
            [20, 23, 11, 8],    # 后右 (+Y, +X)
            [23, 14, 2, 11],    # 后左 (+Y, -X)

            # 8个角的三角面 (朝外的小三角)
            [2, 1, 0],          # 角0 (-x, -y, -z)
            [4, 5, 3],          # 角1 (+x, -y, -z)
            [8, 7, 6],          # 角2 (+x, +y, -z)
            [10, 11, 9],        # 角3 (-x, +y, -z)
            [13, 14, 12],       # 角4 (-x, -y, +z)
            [17, 16, 15],       # 角5 (+x, -y, +z)
            [19, 20, 18],       # 角6 (+x, +y, +z)
            [23, 22, 21],       # 角7 (-x, +y, +z)
        ]

        face_vertex_counts = []
        face_vertex_indices = []
        for face in faces:
            face_vertex_counts.append(len(face))
            face_vertex_indices.extend(face)

        # 计算法线（每个面一个法线）
        normals = []
        # 6个主面法线
        main_normals = [
            Gf.Vec3f(0, 0, -1),   # 底面 -Z
            Gf.Vec3f(0, 0, 1),    # 顶面 +Z
            Gf.Vec3f(-1, 0, 0),   # 左面 -X
            Gf.Vec3f(1, 0, 0),    # 右面 +X
            Gf.Vec3f(0, -1, 0),   # 前面 -Y
            Gf.Vec3f(0, 1, 0),    # 后面 +Y
        ]
        # 12条边的倒角面法线（45度斜面）
        sqrt2 = 0.7071
        edge_normals = [
            Gf.Vec3f(0, -sqrt2, -sqrt2),   # 底-前
            Gf.Vec3f(sqrt2, 0, -sqrt2),    # 底-右
            Gf.Vec3f(0, sqrt2, -sqrt2),    # 底-后
            Gf.Vec3f(-sqrt2, 0, -sqrt2),   # 底-左
            Gf.Vec3f(0, -sqrt2, sqrt2),    # 顶-前
            Gf.Vec3f(sqrt2, 0, sqrt2),     # 顶-右
            Gf.Vec3f(0, sqrt2, sqrt2),     # 顶-后
            Gf.Vec3f(-sqrt2, 0, sqrt2),    # 顶-左
            Gf.Vec3f(-sqrt2, -sqrt2, 0),   # 前左
            Gf.Vec3f(sqrt2, -sqrt2, 0),    # 前右
            Gf.Vec3f(sqrt2, sqrt2, 0),     # 后右
            Gf.Vec3f(-sqrt2, sqrt2, 0),    # 后左
        ]
        # 8个角的法线（指向角的方向）
        sqrt3 = 0.5774
        corner_normals = [
            Gf.Vec3f(-sqrt3, -sqrt3, -sqrt3),  # 角0
            Gf.Vec3f(sqrt3, -sqrt3, -sqrt3),   # 角1
            Gf.Vec3f(sqrt3, sqrt3, -sqrt3),    # 角2
            Gf.Vec3f(-sqrt3, sqrt3, -sqrt3),   # 角3
            Gf.Vec3f(-sqrt3, -sqrt3, sqrt3),   # 角4
            Gf.Vec3f(sqrt3, -sqrt3, sqrt3),    # 角5
            Gf.Vec3f(sqrt3, sqrt3, sqrt3),     # 角6
            Gf.Vec3f(-sqrt3, sqrt3, sqrt3),    # 角7
        ]

        # 为每个面的每个顶点分配法线
        for i, face in enumerate(faces):
            if i < 6:
                normal = main_normals[i]
            elif i < 18:
                normal = edge_normals[i - 6]
            else:
                normal = corner_normals[i - 18]
            for _ in face:
                normals.append(normal)

        # UV坐标（faceVarying，每个面顶点独立UV）
        uvs = []
        uv_indices = []
        uv_idx = 0
        for face in faces:
            n = len(face)
            if n == 4:
                uvs.extend([
                    Gf.Vec2f(0, 0), Gf.Vec2f(1, 0),
                    Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)
                ])
            else:  # 三角形
                uvs.extend([
                    Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(0.5, 1)
                ])
            for j in range(n):
                uv_indices.append(uv_idx + j)
            uv_idx += n

        return points, normals, face_vertex_counts, face_vertex_indices, uvs, uv_indices

    def create_box(
        self,
        name: str = None,
        position: list = None,
        size: list = None,
        mass: float = 1.0,
        color: list = None,
        texture_path: str = None
    ) -> str:
        """
        创建一个纸箱

        Args:
            name: 纸箱名称
            position: 位置 [x, y, z]
            size: 尺寸 [长, 宽, 高]
            mass: 质量(kg)
            color: RGBA颜色 (仅在无纹理时使用)
            texture_path: 纹理图片路径

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

        # 创建带倒角的网格几何（用于渲染）
        mesh_path = f"{prim_path}/geometry"
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)

        # 生成倒角网格数据
        points, normals, face_counts, face_indices, uvs, uv_indices = \
            self._create_beveled_box_mesh(size, bevel_radius=0.003)

        # 设置网格属性
        mesh.GetPointsAttr().Set(points)
        mesh.GetFaceVertexCountsAttr().Set(face_counts)
        mesh.GetFaceVertexIndicesAttr().Set(face_indices)

        # 设置法线（faceVarying插值）
        mesh.GetNormalsAttr().Set(normals)
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)

        # 设置UV坐标
        uv_primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
        )
        uv_primvar.Set(uvs)
        uv_primvar.SetIndices(uv_indices)

        # 创建粗糙纸箱材质
        mat_path = f"{prim_path}/material"
        material = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/shader")
        shader.CreateIdAttr("UsdPreviewSurface")

        # 如果有纹理，使用纹理贴图
        if texture_path:
            # 创建纹理采样器
            tex_shader = UsdShade.Shader.Define(stage, f"{mat_path}/diffuse_texture")
            tex_shader.CreateIdAttr("UsdUVTexture")
            tex_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_path)
            tex_shader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tex_shader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            tex_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

            # 创建UV读取器
            uv_reader = UsdShade.Shader.Define(stage, f"{mat_path}/uv_reader")
            uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
            uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            uv_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

            # 连接UV到纹理
            tex_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                uv_reader.ConnectableAPI(), "result"
            )

            # 连接纹理到shader的diffuseColor
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                tex_shader.ConnectableAPI(), "rgb"
            )
        else:
            # 无纹理时使用纯色
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(color[0], color[1], color[2])
            )

        # 设置粗糙度（0.8-1.0 很粗糙）
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
        # 设置金属度（纸箱不是金属）
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

        # 连接shader到material
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        # 绑定材质到几何体
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)

        # 添加刚体物理
        rigid_body = UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        rigid_body.CreateRigidBodyEnabledAttr(True)

        # 创建独立的碰撞体（简单Box，不渲染）
        collider_path = f"{prim_path}/collider"
        collider = UsdGeom.Cube.Define(stage, collider_path)
        collider.GetSizeAttr().Set(1.0)
        collider.AddScaleOp().Set(Gf.Vec3f(*size))
        # 隐藏碰撞体，不参与渲染
        collider.GetPurposeAttr().Set(UsdGeom.Tokens.guide)

        # 添加碰撞到独立的碰撞体
        collision = UsdPhysics.CollisionAPI.Apply(collider.GetPrim())

        # 添加物理材质 - 高摩擦、低弹性，防止箱子弹开
        phys_mat_path = f"{prim_path}/physics_material"
        phys_mat_prim = stage.DefinePrim(phys_mat_path)
        phys_mat = UsdPhysics.MaterialAPI.Apply(phys_mat_prim)
        phys_mat.CreateStaticFrictionAttr(0.8)   # 静摩擦
        phys_mat.CreateDynamicFrictionAttr(0.6)  # 动摩擦
        phys_mat.CreateRestitutionAttr(0.01)     # 极低弹性，几乎不反弹

        # 使用PhysxSchema绑定物理材质到碰撞体
        physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(collider.GetPrim())
        collision_prim = collider.GetPrim()
        rel = collision_prim.CreateRelationship("material:binding:physics", False)
        rel.SetTargets([Sdf.Path(phys_mat_path)])

        # 添加刚体阻尼 - 减缓运动，让箱子更快稳定
        physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(xform.GetPrim())
        physx_rb.CreateLinearDampingAttr(0.5)    # 线性阻尼
        physx_rb.CreateAngularDampingAttr(0.5)   # 角阻尼

        # 设置质量
        mass_api = UsdPhysics.MassAPI.Apply(xform.GetPrim())
        mass_api.CreateMassAttr(mass)

        # 添加语义标签用于实例分割
        sem = Semantics.SemanticsAPI.Apply(xform.GetPrim(), "Semantics")
        sem.CreateSemanticTypeAttr().Set("class")
        sem.CreateSemanticDataAttr().Set("box")

        # 保存纸箱信息
        self._boxes[name] = {
            "prim_path": prim_path,
            "position": position,
            "size": size,
            "mass": mass
        }

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
        drop_height: float = 0.2,
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
        # 记录本批次已生成箱子的位置和尺寸，用于碰撞检测
        spawned_boxes = []

        for i in range(count):
            # 随机尺寸
            size = [
                np.random.uniform(size_range[0][0], size_range[0][1]),
                np.random.uniform(size_range[1][0], size_range[1][1]),
                np.random.uniform(size_range[2][0], size_range[2][1])
            ]

            # 尝试找到不重叠的位置
            max_attempts = 50
            pos_x, pos_y, pos_z = None, None, None

            for attempt in range(max_attempts):
                # 随机位置（在中心点周围散布）
                trial_x = center[0] + np.random.uniform(-spread, spread)
                trial_y = center[1] + np.random.uniform(-spread, spread)
                # 高度根据已生成箱子数量递增，确保足够间距
                layer = len(spawned_boxes) // 5
                trial_z = drop_height + layer * 0.20 + size[2] / 2  # 层间距增加到20cm

                # 检查与已生成箱子是否重叠
                overlap = False
                for other_pos, other_size in spawned_boxes:
                    # 计算两个箱子在各轴上的距离
                    dx = abs(trial_x - other_pos[0])
                    dy = abs(trial_y - other_pos[1])
                    dz = abs(trial_z - other_pos[2])

                    # 计算最小安全距离（两个箱子半尺寸之和 + 小间隙）
                    min_dx = (size[0] + other_size[0]) / 2 + 0.01
                    min_dy = (size[1] + other_size[1]) / 2 + 0.01
                    min_dz = (size[2] + other_size[2]) / 2 + 0.01

                    # 如果三个轴都重叠，则箱子重叠
                    if dx < min_dx and dy < min_dy and dz < min_dz:
                        overlap = True
                        break

                if not overlap:
                    pos_x, pos_y, pos_z = trial_x, trial_y, trial_z
                    break

            # 如果找不到不重叠的位置，放到更高的层
            if pos_x is None:
                pos_x = center[0] + np.random.uniform(-spread, spread)
                pos_y = center[1] + np.random.uniform(-spread, spread)
                pos_z = drop_height + (len(spawned_boxes) // 3 + 1) * 0.25 + size[2] / 2

            # 随机质量
            mass = np.random.uniform(mass_range[0], mass_range[1])

            # 记录本箱子位置和尺寸
            spawned_boxes.append(([pos_x, pos_y, pos_z], size))

            # 随机选择纹理（如果有的话）
            texture_path = None
            if self._textures:
                texture_path = np.random.choice(self._textures)

            # 随机纸箱颜色（仅在无纹理时使用）
            base_r, base_g, base_b = 0.72, 0.53, 0.35
            variation = 0.05
            color = [
                base_r + np.random.uniform(-variation, variation),
                base_g + np.random.uniform(-variation, variation),
                base_b + np.random.uniform(-variation, variation),
                1.0
            ]

            # 创建纸箱
            path = self.create_box(
                position=[pos_x, pos_y, pos_z],
                size=size,
                mass=mass,
                color=color,
                texture_path=texture_path
            )
            box_paths.append(path)

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

    def get_random_box(self, exclude_paths: list = None) -> tuple:
        """
        随机选择一个纸箱

        Args:
            exclude_paths: 要排除的箱子路径列表

        Returns:
            tuple: (name, prim_path, position) 或 None
        """
        if not self._boxes:
            return None

        if exclude_paths is None:
            exclude_paths = []

        # 过滤可用的箱子
        available = [
            (name, info) for name, info in self._boxes.items()
            if info["prim_path"] not in exclude_paths
        ]

        if not available:
            return None

        # 随机选择
        name, info = available[np.random.randint(len(available))]
        pos = self.get_box_position(name)

        return (name, info["prim_path"], pos)

    def delete_box(self, prim_path: str) -> bool:
        """
        从场景中删除指定纸箱

        Args:
            prim_path: 纸箱的prim路径

        Returns:
            bool: 是否成功删除
        """
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)

        if not prim.IsValid():
            return False

        # 从stage中删除prim
        stage.RemovePrim(prim_path)

        # 从跟踪列表中移除
        for name, info in list(self._boxes.items()):
            if info["prim_path"] == prim_path:
                del self._boxes[name]
                break

        return True

    def remove_box_from_tracking(self, name: str):
        """从跟踪列表中移除箱子（不删除物理对象）"""
        if name in self._boxes:
            del self._boxes[name]

    def get_all_box_positions(self, exclude_paths: list = None) -> dict:
        """
        获取所有箱子的当前位置快照

        Args:
            exclude_paths: 要排除的箱子路径列表

        Returns:
            dict: {prim_path: position_array}
        """
        if exclude_paths is None:
            exclude_paths = []

        positions = {}
        stage = omni.usd.get_context().get_stage()

        for name, info in self._boxes.items():
            if info["prim_path"] in exclude_paths:
                continue
            prim = stage.GetPrimAtPath(info["prim_path"])
            if prim.IsValid():
                xformable = UsdGeom.Xformable(prim)
                transform = xformable.ComputeLocalToWorldTransform(0)
                pos = transform.ExtractTranslation()
                positions[info["prim_path"]] = np.array([pos[0], pos[1], pos[2]])

        return positions

    def save_scene_state(self) -> dict:
        """
        保存当前场景中所有箱子的状态（位置、旋转）

        Returns:
            dict: 场景状态快照
        """
        state = {}
        stage = omni.usd.get_context().get_stage()

        for name, info in self._boxes.items():
            prim = stage.GetPrimAtPath(info["prim_path"])
            if prim.IsValid():
                xformable = UsdGeom.Xformable(prim)
                transform = xformable.ComputeLocalToWorldTransform(0)
                pos = transform.ExtractTranslation()
                # 提取旋转四元数
                rot_matrix = transform.ExtractRotationMatrix()
                quat = rot_matrix.ExtractRotation().GetQuaternion()

                state[name] = {
                    "prim_path": info["prim_path"],
                    "position": [pos[0], pos[1], pos[2]],
                    "quaternion": [quat.GetReal(), quat.GetImaginary()[0],
                                   quat.GetImaginary()[1], quat.GetImaginary()[2]],
                    "size": info["size"],
                    "mass": info["mass"]
                }

        return state

    def restore_scene_state(self, state: dict):
        """
        恢复场景到保存的状态

        Args:
            state: 之前保存的场景状态
        """
        stage = omni.usd.get_context().get_stage()

        for name, box_state in state.items():
            prim_path = box_state["prim_path"]
            prim = stage.GetPrimAtPath(prim_path)

            if not prim.IsValid():
                # 箱子被删除了，需要重新创建
                self._recreate_box(name, box_state)
            else:
                # 箱子存在，恢复位置
                self._reset_box_transform(prim, box_state)

    def _recreate_box(self, name: str, box_state: dict):
        """重新创建被删除的箱子"""
        # 使用保存的参数重新创建
        self.create_box(
            name=name,
            position=box_state["position"],
            size=box_state["size"],
            mass=box_state["mass"]
        )

    def _reset_box_transform(self, prim, box_state: dict):
        """重置箱子的位置、旋转和速度"""
        xformable = UsdGeom.Xformable(prim)

        # 清除现有变换操作
        xformable.ClearXformOpOrder()

        # 恢复位置
        xformable.AddTranslateOp().Set(Gf.Vec3d(*box_state["position"]))

        # 恢复旋转（使用四元数）
        quat = box_state["quaternion"]
        xformable.AddOrientOp().Set(Gf.Quatf(quat[0], quat[1], quat[2], quat[3]))

        # 重置刚体速度为0
        rigid_body = UsdPhysics.RigidBodyAPI(prim)
        if rigid_body:
            rigid_body.GetVelocityAttr().Set(Gf.Vec3f(0, 0, 0))
            rigid_body.GetAngularVelocityAttr().Set(Gf.Vec3f(0, 0, 0))
