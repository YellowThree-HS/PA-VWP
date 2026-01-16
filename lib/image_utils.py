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
