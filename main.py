"""
BoxWorld MVP - 可视化仿真脚本
场景预览：生成箱子堆叠并保存九视角图像
"""

from isaacsim import SimulationApp

# 启动仿真应用（可视化模式）
simulation_app = SimulationApp({"headless": False})

import numpy as np
import os
from isaacsim.core.api import World
import omni.usd
import omni.replicator.core as rep
from pxr import UsdLux, UsdGeom, Gf

# 导入自定义模块
import sys
sys.path.append("src")
sys.path.append(".")
from box_generator import BoxGenerator
from lib import SceneBuilder, ImageUtils


# 九相机配置：八个方向 + 正上方
# 相机围绕中心点 (0.45, 0.0) 布置，距离约1.0m，高度1.5m
# 旋转角度：rx控制俯仰（负值向下看），rz控制水平朝向
MULTI_CAMERA_CONFIGS = {
    "top": {  # 正上方
        "position": (0.45, 0.0, 1.5),
        "rotation": (0, 0, 180),
    },
    "north": {  # 北（+Y方向）- 相机在+Y，看向-Y
        "position": (0.45, 1.0, 1.5),
        "rotation": (-30, 0, 0),
    },
    "south": {  # 南（-Y方向）- 相机在-Y，看向+Y
        "position": (0.45, -1.0, 1.5),
        "rotation": (-30, 0, 180),
    },
    "east": {  # 东（+X方向）- 相机在+X，看向-X
        "position": (1.45, 0.0, 1.5),
        "rotation": (-30, 0, -90),
    },
    "west": {  # 西（-X方向）- 相机在-X，看向+X
        "position": (-0.55, 0.0, 1.5),
        "rotation": (-30, 0, 90),
    },
    "northeast": {  # 东北 - 相机在+X+Y，看向西南
        "position": (1.15, 0.70, 1.5),
        "rotation": (-30, 0, -45),
    },
    "northwest": {  # 西北 - 相机在-X+Y，看向东南
        "position": (-0.25, 0.70, 1.5),
        "rotation": (-30, 0, 45),
    },
    "southeast": {  # 东南 - 相机在+X-Y，看向西北
        "position": (1.15, -0.70, 1.5),
        "rotation": (-30, 0, -135),
    },
    "southwest": {  # 西南 - 相机在-X-Y，看向东北
        "position": (-0.25, -0.70, 1.5),
        "rotation": (-30, 0, 135),
    },
}


class BoxSceneViewer:
    """箱子场景预览：生成箱子并保存九视角图像"""

    def __init__(self):
        self.world = None
        self.box_gen = None
        self.scene_builder = SceneBuilder()
        self.multi_cameras = {}

        # 状态机
        self.state = "INIT"
        self.step_count = 0
        self.box_spawn_steps = [20, 80, 140]  # 分三批生成
        self.box_paths = []
        self.images_saved = False

    def setup_scene(self):
        """设置仿真场景"""
        self.world = World(stage_units_in_meters=1.0)
        self.scene_builder.create_textured_floor(size=5.0)
        self.box_gen = BoxGenerator(texture_dir="assets/cardboard_textures_processed")
        self.create_multi_cameras()
        self.randomize_lighting()
        print("Scene setup complete!")

    def create_multi_cameras(self):
        """创建九个方向的相机"""
        stage = omni.usd.get_context().get_stage()

        for name, config in MULTI_CAMERA_CONFIGS.items():
            cam_path = f"/World/Camera_{name}"
            camera = UsdGeom.Camera.Define(stage, cam_path)
            xform = UsdGeom.Xformable(camera.GetPrim())
            xform.AddTranslateOp().Set(Gf.Vec3d(*config["position"]))
            xform.AddRotateXYZOp().Set(Gf.Vec3f(*config["rotation"]))
            camera.GetFocalLengthAttr().Set(18.0)

            # 创建 render product 和 annotator
            render_product = rep.create.render_product(cam_path, (640, 480))
            rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            rgb_annotator.attach([render_product])

            self.multi_cameras[name] = {
                "path": cam_path,
                "render_product": render_product,
                "rgb_annotator": rgb_annotator,
            }

        print(f"Created {len(self.multi_cameras)} cameras")

    def randomize_lighting(self):
        """随机化环境光"""
        stage = omni.usd.get_context().get_stage()

        dome_light_path = "/World/DomeLight"
        prim = stage.GetPrimAtPath(dome_light_path)
        if prim.IsValid():
            dome_light = UsdLux.DomeLight(prim)
        else:
            dome_light = UsdLux.DomeLight.Define(stage, dome_light_path)

        intensity = np.random.uniform(300, 1200)
        dome_light.GetIntensityAttr().Set(intensity)

        xform = UsdGeom.Xformable(dome_light)
        xform.ClearXformOpOrder()
        xform.AddRotateXYZOp().Set(Gf.Vec3f(
            np.random.uniform(0, 360),
            np.random.uniform(0, 360),
            0
        ))

        enable_temp = dome_light.GetEnableColorTemperatureAttr()
        enable_temp.Set(True)
        temp = np.random.uniform(4000, 8000)
        dome_light.GetColorTemperatureAttr().Set(temp)

        print(f"Lighting randomized: Int={intensity:.1f}, Temp={temp:.1f}")

    def spawn_box(self, batch: int = 1):
        """分批生成箱子"""
        batch_counts = {1: 34, 2: 33, 3: 33}
        count = batch_counts.get(batch, 34)
        paths = self.box_gen.create_random_boxes(
            count=count,
            center=[0.45, 0.0],
            spread=0.15,
            drop_height=0.30,
            size_range=((0.06, 0.12), (0.06, 0.12), (0.06, 0.12)),
            mass_range=(0.3, 1.5)
        )
        self.box_paths.extend(paths)
        print(f"Batch {batch}: Spawned {count} boxes, total: {len(self.box_paths)}")

    def save_multi_camera_images(self):
        """保存九个相机的RGB图像"""
        os.makedirs("result/multi_view", exist_ok=True)
        for name, cam_info in self.multi_cameras.items():
            rgb_annotator = cam_info["rgb_annotator"]
            data = rgb_annotator.get_data()
            if data is not None:
                rgb = data[:, :, :3].copy()
                noisy_rgb = ImageUtils.add_camera_noise(rgb)
                ImageUtils.save_rgb(noisy_rgb, f"result/multi_view/{name}.png")
        print(f"Saved {len(self.multi_cameras)} camera images to result/multi_view/")

    def run(self):
        """运行仿真"""
        self.setup_scene()
        self.world.reset()
        print("Starting simulation...")

        while simulation_app.is_running():
            self.world.step(render=True)
            self.step_count += 1

            # 分批生成箱子
            if self.step_count == self.box_spawn_steps[0]:
                self.spawn_box(batch=1)
            elif self.step_count == self.box_spawn_steps[1]:
                self.spawn_box(batch=2)
            elif self.step_count == self.box_spawn_steps[2]:
                self.spawn_box(batch=3)
                print("All boxes dropped, waiting for stabilization...")

            # 等待箱子稳定后保存图像（约4秒后）
            if self.step_count == 380 and not self.images_saved:
                self.save_multi_camera_images()
                self.images_saved = True
                print("Images saved! You can close the window now.")

        simulation_app.close()


if __name__ == "__main__":
    viewer = BoxSceneViewer()
    viewer.run()
