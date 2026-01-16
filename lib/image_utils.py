"""图像处理工具模块"""

import cv2
import numpy as np
import os


class ImageUtils:
    """图像处理工具类"""

    @staticmethod
    def get_mask_id(prim_path: str, id_to_labels: dict) -> int:
        """根据prim_path获取掩码中的ID"""
        for uid, label_info in id_to_labels.items():
            if prim_path in str(label_info):
                return int(uid)
        return None

    @staticmethod
    def save_rgb(rgb_image: np.ndarray, path: str):
        """保存RGB图像（RGB转BGR）"""
        bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)

    @staticmethod
    def save_depth(depth: np.ndarray, path: str, max_depth: float = 3.0):
        """保存深度图（归一化可视化）"""
        depth_vis = np.clip(depth, 0, max_depth)
        depth_vis = (depth_vis / max_depth * 255).astype(np.uint8)
        cv2.imwrite(path, depth_vis)

    @staticmethod
    def create_annotated_image(
        rgb_image: np.ndarray,
        mask_data: np.ndarray,
        id_to_labels: dict,
        removed_path: str,
        affected_boxes: list,
        alpha: float = 0.4
    ) -> np.ndarray:
        """创建标注图（绿色=消失，红色=受影响）"""
        rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        overlay = rgb_bgr.copy()

        # 消失的箱子标绿色
        removed_id = ImageUtils.get_mask_id(removed_path, id_to_labels)
        if removed_id is not None:
            overlay[mask_data == removed_id] = [0, 255, 0]

        # 受影响的箱子标红色
        for box in affected_boxes:
            box_path = box.get("path", box.get("prim_path", ""))
            affected_id = ImageUtils.get_mask_id(box_path, id_to_labels)
            if affected_id is not None:
                overlay[mask_data == affected_id] = [0, 0, 255]

        return cv2.addWeighted(rgb_bgr, alpha, overlay, 1 - alpha, 0)

    @staticmethod
    def add_camera_noise(
        image: np.ndarray,
        gaussian_std: float = None,
        shot_noise: bool = True,
        color_jitter: bool = True
    ) -> np.ndarray:
        """
        添加相机噪声，模拟真实相机

        Args:
            image: RGB图像 (uint8)
            gaussian_std: 高斯噪声标准差，None则随机[2,10]
            shot_noise: 是否添加散粒噪声（泊松噪声）
            color_jitter: 是否添加颜色抖动
        """
        img = image.astype(np.float32)

        # 1. 散粒噪声 (Shot noise / Poisson noise)
        if shot_noise:
            noise_level = np.random.uniform(0.005, 0.02)
            shot = np.random.poisson(img * noise_level) / noise_level - img
            img = img + shot * 0.075

        # 2. 高斯噪声 (读取噪声)
        if gaussian_std is None:
            gaussian_std = np.random.uniform(0.5, 2.5)
        gaussian = np.random.normal(0, gaussian_std, img.shape)
        img = img + gaussian

        # 3. 颜色抖动 (白平衡偏移)
        if color_jitter:
            color_shift = np.random.uniform(0.9875, 1.0125, 3).astype(np.float32)
            img = img * color_shift

        return np.clip(img, 0, 255).astype(np.uint8)

    @staticmethod
    def add_depth_noise(depth: np.ndarray) -> np.ndarray:
        """
        添加深度图噪声，模拟真实深度相机

        Args:
            depth: 深度图 (float32, 单位：米)
        """
        noisy = depth.copy().astype(np.float32)

        # 1. 高斯噪声 (测量噪声)
        gaussian_std = np.random.uniform(0.002, 0.008)
        noisy += np.random.normal(0, gaussian_std, noisy.shape)

        # 2. 距离相关噪声 (远处噪声更大)
        distance_noise = np.random.uniform(0.001, 0.003)
        noisy += np.random.normal(0, 1, noisy.shape) * noisy * distance_noise

        # 3. 随机缺失 (模拟反射失败)
        if np.random.random() < 0.3:
            dropout_rate = np.random.uniform(0.001, 0.005)
            mask = np.random.random(noisy.shape) < dropout_rate
            noisy[mask] = 0

        return np.maximum(noisy, 0)

    @staticmethod
    def estimate_projected_pixels(
        box_size: list,
        box_position: np.ndarray,
        camera_position: tuple,
        focal_length: float = 18.0,
        image_size: tuple = (640, 480),
        horizontal_aperture: float = 20.955
    ) -> float:
        """
        估算箱子在相机视角下的理论投影像素数

        基于相机到箱子的方向向量，计算各个面的可见投影面积

        Args:
            box_size: 箱子尺寸 [长, 宽, 高] 对应 [x, y, z]
            box_position: 箱子世界坐标位置
            camera_position: 相机位置 (x, y, z)
            focal_length: 焦距 (mm)
            image_size: 图像尺寸 (width, height)
            horizontal_aperture: 水平孔径 (mm)，Isaac Sim 默认 20.955mm

        Returns:
            float: 估算的理论像素数
        """
        cam_pos = np.array(camera_position)
        box_pos = np.array(box_position)

        # 计算从箱子指向相机的方向向量（归一化）
        view_dir = cam_pos - box_pos
        distance = np.linalg.norm(view_dir)
        if distance < 0.1:
            distance = 0.1
        view_dir = view_dir / distance

        # 箱子六个面的法向量和面积
        # 面法向量指向外部，只有当 view_dir 与法向量夹角 < 90° 时才可见
        faces = [
            (np.array([1, 0, 0]), box_size[1] * box_size[2]),   # +X 面 (yz)
            (np.array([-1, 0, 0]), box_size[1] * box_size[2]),  # -X 面 (yz)
            (np.array([0, 1, 0]), box_size[0] * box_size[2]),   # +Y 面 (xz)
            (np.array([0, -1, 0]), box_size[0] * box_size[2]),  # -Y 面 (xz)
            (np.array([0, 0, 1]), box_size[0] * box_size[1]),   # +Z 面 (xy) 顶面
            (np.array([0, 0, -1]), box_size[0] * box_size[1]),  # -Z 面 (xy) 底面
        ]

        # 计算可见面的投影面积之和
        total_projected_area = 0.0
        for normal, area in faces:
            # cos(theta) = view_dir · normal
            cos_theta = np.dot(view_dir, normal)
            if cos_theta > 0:
                # 投影面积 = 实际面积 * cos(theta)
                total_projected_area += area * cos_theta

        # 计算每米对应多少像素
        # 透视投影公式: pixels_per_meter = (focal_length / horizontal_aperture) * image_width / distance
        # 注意: focal_length 和 horizontal_aperture 单位都是 mm，可以约掉
        pixels_per_meter = (focal_length / horizontal_aperture) * image_size[0] / distance

        # 理论像素数
        return total_projected_area * (pixels_per_meter ** 2)

    @staticmethod
    def get_bbox_from_mask(mask: np.ndarray, mask_id: int) -> dict:
        """
        从 mask 中计算指定 ID 的 bbox

        Args:
            mask: 分割掩码
            mask_id: 目标 ID

        Returns:
            dict: {"x1": int, "y1": int, "x2": int, "y2": int} 或 None
        """
        binary_mask = (mask == mask_id).astype(np.uint8)
        coords = np.where(binary_mask > 0)
        if len(coords[0]) == 0:
            return None
        y1, y2 = int(coords[0].min()), int(coords[0].max())
        x1, x2 = int(coords[1].min()), int(coords[1].max())
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    @staticmethod
    def create_summary_image(
        rgb_image: np.ndarray,
        mask_data: np.ndarray,
        id_to_labels: dict,
        stable_boxes: list,
        unstable_boxes: list,
        alpha: float = 0.4
    ) -> np.ndarray:
        """创建总结图（黄色=稳定，蓝色=不稳定）"""
        rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        overlay = rgb_bgr.copy()

        # 稳定的箱子标黄色
        for box in stable_boxes:
            box_path = box.get("prim_path", "")
            box_id = ImageUtils.get_mask_id(box_path, id_to_labels)
            if box_id is not None:
                overlay[mask_data == box_id] = [0, 255, 255]

        # 不稳定的箱子标蓝色
        for box in unstable_boxes:
            box_path = box.get("prim_path", "")
            box_id = ImageUtils.get_mask_id(box_path, id_to_labels)
            if box_id is not None:
                overlay[mask_data == box_id] = [255, 0, 0]

        return cv2.addWeighted(rgb_bgr, alpha, overlay, 1 - alpha, 0)
