"""场景构建模块 - 地面、光源等基础场景元素"""

import random
from pathlib import Path
from pxr import UsdGeom, UsdPhysics, UsdShade, UsdLux, Gf, Sdf
import omni.usd


class SceneBuilder:
    """场景构建器"""

    def __init__(self, floor_texture_dir: str = "assets/floor_processed"):
        self.floor_texture_dir = Path(floor_texture_dir)

    def create_textured_floor(self, size: float = 5.0):
        """创建带随机纹理的地面"""
        stage = omni.usd.get_context().get_stage()

        # 创建光源
        self.create_lights(stage)

        # 创建地面mesh
        floor_path = "/World/Floor"
        floor_mesh = UsdGeom.Mesh.Define(stage, floor_path)

        # 定义顶点
        half = size / 2
        points = [
            Gf.Vec3f(-half, -half, 0),
            Gf.Vec3f(half, -half, 0),
            Gf.Vec3f(half, half, 0),
            Gf.Vec3f(-half, half, 0)
        ]
        floor_mesh.GetPointsAttr().Set(points)

        # 定义面
        floor_mesh.GetFaceVertexCountsAttr().Set([4])
        floor_mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])

        # UV坐标
        texcoords = UsdGeom.PrimvarsAPI(floor_mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying
        )
        texcoords.Set([
            Gf.Vec2f(0, 0),
            Gf.Vec2f(size, 0),
            Gf.Vec2f(size, size),
            Gf.Vec2f(0, size)
        ])

        # 法线
        floor_mesh.GetNormalsAttr().Set([Gf.Vec3f(0, 0, 1)] * 4)

        # 添加碰撞
        UsdPhysics.CollisionAPI.Apply(floor_mesh.GetPrim())

        # 随机选择纹理
        self._apply_random_floor_texture(floor_path)

        print(f"Created {size}m x {size}m textured floor")

    def _apply_random_floor_texture(self, floor_path: str):
        """为地面应用随机纹理"""
        stage = omni.usd.get_context().get_stage()

        texture_files = list(self.floor_texture_dir.glob("*.jpg"))
        if not texture_files:
            print("Warning: No floor textures found")
            return

        texture_path = random.choice(texture_files)
        texture_abs_path = str(texture_path.absolute()).replace("\\", "/")

        # 创建材质
        mat_path = "/World/Materials/FloorMaterial"
        material = UsdShade.Material.Define(stage, mat_path)

        # 创建 UsdPreviewSurface shader
        shader_path = f"{mat_path}/Shader"
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")

        # 创建 UV 读取节点
        st_reader_path = f"{mat_path}/stReader"
        st_reader = UsdShade.Shader.Define(stage, st_reader_path)
        st_reader.CreateIdAttr("UsdPrimvarReader_float2")
        st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        # 创建纹理采样器
        tex_path = f"{mat_path}/DiffuseTexture"
        tex_sampler = UsdShade.Shader.Define(stage, tex_path)
        tex_sampler.CreateIdAttr("UsdUVTexture")
        tex_sampler.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_abs_path)
        tex_sampler.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex_sampler.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex_sampler.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

        # 连接 UV 到纹理采样器
        tex_sampler.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            st_reader.ConnectableAPI(), "result"
        )

        # 连接纹理到 shader
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex_sampler.ConnectableAPI(), "rgb"
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)

        # 设置材质输出
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        # 绑定材质到地面
        floor_prim = stage.GetPrimAtPath(floor_path)
        UsdShade.MaterialBindingAPI(floor_prim).Bind(material)

    def randomize_floor_texture(self):
        """随机更换地面纹理"""
        stage = omni.usd.get_context().get_stage()

        texture_files = list(self.floor_texture_dir.glob("*.jpg"))
        if not texture_files:
            return

        texture_path = random.choice(texture_files)
        texture_abs_path = str(texture_path.absolute()).replace("\\", "/")

        # 更新纹理采样器的文件路径
        tex_path = "/World/Materials/FloorMaterial/DiffuseTexture"
        tex_prim = stage.GetPrimAtPath(tex_path)
        if tex_prim.IsValid():
            tex_sampler = UsdShade.Shader(tex_prim)
            tex_sampler.GetInput("file").Set(texture_abs_path)

    def create_lights(self, stage=None):
        """创建场景光源"""
        if stage is None:
            stage = omni.usd.get_context().get_stage()

        # 主光源 - 圆顶光（环境光）
        dome_path = "/World/DomeLight"
        dome_light = UsdLux.DomeLight.Define(stage, dome_path)
        dome_light.GetIntensityAttr().Set(500)

        # 方向光 - 模拟太阳光
        sun_path = "/World/SunLight"
        sun_light = UsdLux.DistantLight.Define(stage, sun_path)
        sun_light.GetIntensityAttr().Set(1000)
        sun_xform = UsdGeom.Xformable(sun_light.GetPrim())
        sun_xform.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

        print("Lights created: DomeLight + SunLight")
