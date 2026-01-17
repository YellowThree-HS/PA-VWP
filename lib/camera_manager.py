"""相机和标注器管理模块"""

import numpy as np
import cv2
from pxr import UsdGeom, Gf
import omni.usd
import omni.replicator.core as rep


class CameraManager:
    """相机和Replicator标注器管理"""

    def __init__(self):
        self.render_product = None
        self.rgb_annotator = None
        self.depth_annotator = None
        self.seg_annotator = None
        self.vertical_flip = False

    def create_top_camera(
        self,
        position: tuple = (0.45, 0.0, 1.5),
        rotation: tuple = (0, 0, 180),
        focal_length: float = 18.0,
        resolution: tuple = (640, 480),
        with_depth: bool = False,
        vertical_flip: bool = False
    ):
        """创建顶部相机并设置标注器

        Args:
            position: 相机位置 (x, y, z)
            rotation: 相机旋转角度 (rx, ry, rz)
            focal_length: 焦距
            resolution: 分辨率 (width, height)
            with_depth: 是否启用深度标注器
            vertical_flip: 是否垂直翻转图像（解决相机倒挂导致的图像颠倒）
        """
        self.vertical_flip = vertical_flip
        stage = omni.usd.get_context().get_stage()

        # 创建相机
        cam_path = "/World/TopCamera"
        camera = UsdGeom.Camera.Define(stage, cam_path)
        xform = UsdGeom.Xformable(camera.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))
        camera.GetFocalLengthAttr().Set(focal_length)

        # 设置Replicator
        self.render_product = rep.create.render_product(cam_path, resolution)

        # RGB标注器
        self.rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb_annotator.attach([self.render_product])

        # 实例分割标注器
        self.seg_annotator = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation")
        self.seg_annotator.attach([self.render_product])

        # 深度标注器（可选）
        if with_depth:
            self.depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
            self.depth_annotator.attach([self.render_product])

    def get_rgb(self) -> np.ndarray:
        """获取RGB图像（自动应用垂直翻转如果设置了的话）"""
        if self.rgb_annotator is None:
            return None
        data = self.rgb_annotator.get_data()
        if data is not None:
            img = data[:, :, :3].copy()
            if getattr(self, 'vertical_flip', False):
                img = cv2.flip(img, 0)
            return img
        return None

    def get_depth(self) -> np.ndarray:
        """获取深度图"""
        if self.depth_annotator is None:
            return None
        data = self.depth_annotator.get_data()
        if data is not None:
            return data.copy()
        return None

    def get_segmentation(self) -> tuple:
        """获取实例分割数据

        Returns:
            (mask, id_to_labels) 或 (None, None)
        """
        if self.seg_annotator is None:
            return None, None
        data = self.seg_annotator.get_data()
        if data is not None:
            mask = data["data"].copy()
            id_to_labels = data.get("info", {}).get("idToLabels", {})
            return mask, id_to_labels
        return None, None

    def get_visible_boxes(self, box_gen, min_visible_ratio: float = 0.3) -> list:
        """获取所有可见箱子列表

        Args:
            box_gen: BoxGenerator实例
            min_visible_ratio: 最小可见比例阈值

        Returns:
            可见箱子列表 [{name, prim_path, uid, pixel_count}, ...]
        """
        mask, id_to_labels = self.get_segmentation()
        if mask is None:
            return []

        unique_ids = np.unique(mask)

        # 统计每个箱子的像素数
        box_pixels = {}
        for uid in unique_ids:
            if uid == 0:
                continue
            label_str = str(id_to_labels.get(str(uid), ""))
            if "/World/box_" in label_str:
                box_pixels[uid] = np.sum(mask == uid)

        if not box_pixels:
            return []

        max_pixels = max(box_pixels.values())
        visible_boxes = []
        added_names = set()

        for uid, pixel_count in box_pixels.items():
            if pixel_count / max_pixels < min_visible_ratio:
                continue

            label_str = str(id_to_labels.get(str(uid), {}))
            for name, info in box_gen.boxes.items():
                if info["prim_path"] == label_str and name not in added_names:
                    visible_boxes.append({
                        "name": name,
                        "prim_path": info["prim_path"],
                        "uid": uid,
                        "pixel_count": pixel_count
                    })
                    added_names.add(name)
                    break

        return visible_boxes
