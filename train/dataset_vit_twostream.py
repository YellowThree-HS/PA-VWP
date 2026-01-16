"""
BoxWorld ViT 双流网络数据集
- Global Stream: RGB + Mask (4通道)
- Local Stream: RGB only (3通道，裁剪放大)
"""

import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class ViTTwoStreamDataset(Dataset):
    """ViT双流网络数据集

    每个样本返回:
    - global_input: (4, 224, 224) - 全局视图 RGB + Mask
    - local_input: (3, 224, 224) - 局部裁剪 RGB
    - label: 稳定性标签
    """

    def __init__(self, data_dir: str, image_size: int = 224,
                 padding_ratio: float = 1.5, augment: bool = False):
        """
        Args:
            data_dir: 数据集根目录
            image_size: 输出图像大小
            padding_ratio: 局部裁剪时边界框的扩展比例 (1.5-2.0)
            augment: 是否启用数据增强
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.padding_ratio = padding_ratio
        self.augment = augment
        self.samples = []

        # ImageNet归一化参数
        self.mean_rgb = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std_rgb = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self._scan_samples()

    def _scan_samples(self):
        """扫描数据目录，收集所有样本"""
        for round_dir in sorted(os.listdir(self.data_dir)):
            round_path = os.path.join(self.data_dir, round_dir)
            if not os.path.isdir(round_path):
                continue

            initial_dir = os.path.join(round_path, "initial")
            removals_dir = os.path.join(round_path, "removals")

            if not os.path.exists(initial_dir) or not os.path.exists(removals_dir):
                continue

            rgb_path = os.path.join(initial_dir, "rgb.png")
            if not os.path.exists(rgb_path):
                continue

            for sample_idx in sorted(os.listdir(removals_dir)):
                sample_dir = os.path.join(removals_dir, sample_idx)
                if not os.path.isdir(sample_dir):
                    continue

                mask_path = os.path.join(sample_dir, "mask.png")
                result_path = os.path.join(sample_dir, "result.json")

                if os.path.exists(mask_path) and os.path.exists(result_path):
                    self.samples.append({
                        "rgb_path": rgb_path,
                        "mask_path": mask_path,
                        "result_path": result_path
                    })

        print(f"Found {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def _get_bbox_with_padding(self, mask: np.ndarray):
        """从mask获取带padding的边界框

        Args:
            mask: 二值掩码 (H, W)

        Returns:
            (x1, y1, x2, y2): 带padding的边界框坐标
        """
        h, w = mask.shape

        rows = np.any(mask > 0, axis=1)
        cols = np.any(mask > 0, axis=0)

        if not rows.any() or not cols.any():
            return 0, 0, w, h

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # 计算padding (扩展1.5-2倍)
        box_w = x_max - x_min
        box_h = y_max - y_min
        pad_w = int(box_w * (self.padding_ratio - 1) / 2)
        pad_h = int(box_h * (self.padding_ratio - 1) / 2)

        x1 = max(0, x_min - pad_w)
        y1 = max(0, y_min - pad_h)
        x2 = min(w, x_max + pad_w)
        y2 = min(h, y_max + pad_h)

        return x1, y1, x2, y2

    def _augment_global(self, rgb, mask):
        """全局视图数据增强"""
        if not self.augment:
            return rgb, mask

        # 随机水平翻转
        if np.random.rand() > 0.5:
            rgb = cv2.flip(rgb, 1)
            mask = cv2.flip(mask, 1)

        # 随机垂直翻转
        if np.random.rand() > 0.5:
            rgb = cv2.flip(rgb, 0)
            mask = cv2.flip(mask, 0)

        return rgb, mask

    def _augment_local(self, rgb):
        """局部视图数据增强 - 更激进的颜色抖动"""
        if not self.augment:
            return rgb

        # 颜色抖动
        if np.random.rand() > 0.5:
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] *= np.random.uniform(0.8, 1.2)  # 饱和度
            hsv[:, :, 2] *= np.random.uniform(0.8, 1.2)  # 亮度
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return rgb

    def _normalize_global(self, rgb, mask):
        """归一化全局输入 (4通道)"""
        rgb = cv2.resize(rgb, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size))

        # RGB归一化
        rgb_norm = rgb.astype(np.float32) / 255.0
        rgb_norm = (rgb_norm - self.mean_rgb) / self.std_rgb

        # Mask归一化到0-1
        mask_norm = (mask > 127).astype(np.float32)

        # 拼接为4通道 (H, W, 4) -> (4, H, W)
        combined = np.concatenate([rgb_norm, mask_norm[:, :, np.newaxis]], axis=2)
        tensor = torch.from_numpy(combined).permute(2, 0, 1)

        return tensor

    def _normalize_local(self, rgb):
        """归一化局部输入 (3通道)"""
        rgb = cv2.resize(rgb, (self.image_size, self.image_size))

        # RGB归一化
        rgb_norm = rgb.astype(np.float32) / 255.0
        rgb_norm = (rgb_norm - self.mean_rgb) / self.std_rgb

        # (H, W, 3) -> (3, H, W)
        tensor = torch.from_numpy(rgb_norm).permute(2, 0, 1)

        return tensor

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 读取RGB图像
        rgb = cv2.imread(sample["rgb_path"])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # 读取掩码
        mask = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)

        # 读取标签
        with open(sample["result_path"], "r") as f:
            result = json.load(f)
        label = torch.tensor(result["stability_label"], dtype=torch.float32)

        # === Global Stream ===
        rgb_aug, mask_aug = self._augment_global(rgb.copy(), mask.copy())
        global_input = self._normalize_global(rgb_aug, mask_aug)

        # === Local Stream ===
        x1, y1, x2, y2 = self._get_bbox_with_padding(mask)
        local_rgb = rgb[y1:y2, x1:x2].copy()

        # 处理边界情况
        if local_rgb.shape[0] < 10 or local_rgb.shape[1] < 10:
            local_rgb = rgb.copy()

        local_rgb = self._augment_local(local_rgb)
        local_input = self._normalize_local(local_rgb)

        return global_input, local_input, label
