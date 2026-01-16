"""
BoxWorld 双流网络数据集
支持返回全局视图和局部裁剪
"""

import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class TwoStreamDataset(Dataset):
    """双流网络数据集

    每个样本返回:
    - global_input: 全局视图 (4, 224, 224)
    - local_input: 局部裁剪 (4, 224, 224)
    - label: 稳定性标签
    """

    def __init__(self, data_dir: str, image_size: int = 224, padding_ratio: float = 0.5):
        """
        Args:
            data_dir: 数据集根目录
            image_size: 输出图像大小
            padding_ratio: 局部裁剪时边界框的扩展比例
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.padding_ratio = padding_ratio
        self.samples = []

        # ImageNet归一化参数
        self.mean = np.array([0.485, 0.456, 0.406, 0.0], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225, 1.0], dtype=np.float32)

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

        # 找到mask的边界框
        rows = np.any(mask > 0, axis=1)
        cols = np.any(mask > 0, axis=0)

        if not rows.any() or not cols.any():
            # mask为空，返回整个图像
            return 0, 0, w, h

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # 计算padding
        box_w = x_max - x_min
        box_h = y_max - y_min
        pad_w = int(box_w * self.padding_ratio)
        pad_h = int(box_h * self.padding_ratio)

        # 扩展边界框
        x1 = max(0, x_min - pad_w)
        y1 = max(0, y_min - pad_h)
        x2 = min(w, x_max + pad_w)
        y2 = min(h, y_max + pad_h)

        return x1, y1, x2, y2

    def _normalize(self, rgb: np.ndarray, mask: np.ndarray):
        """归一化并拼接为4通道tensor

        Args:
            rgb: RGB图像 (H, W, 3), uint8
            mask: 掩码 (H, W), uint8

        Returns:
            tensor: (4, H, W) 归一化后的tensor
        """
        # Resize
        rgb = cv2.resize(rgb, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size))

        # 归一化
        rgb_norm = rgb.astype(np.float32) / 255.0
        mask_norm = (mask > 127).astype(np.float32)

        # 拼接为4通道
        combined = np.concatenate([rgb_norm, mask_norm[:, :, np.newaxis]], axis=2)

        # 转为tensor (H, W, 4) -> (4, H, W)
        tensor = torch.from_numpy(combined).permute(2, 0, 1)

        # ImageNet归一化
        mean = torch.tensor(self.mean).view(4, 1, 1)
        std = torch.tensor(self.std).view(4, 1, 1)
        tensor = (tensor - mean) / std

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
        # 全局视图：整个场景resize到224x224
        global_input = self._normalize(rgb, mask)

        # === Local Stream ===
        # 局部裁剪：根据mask边界框裁剪
        x1, y1, x2, y2 = self._get_bbox_with_padding(mask)
        local_rgb = rgb[y1:y2, x1:x2]
        local_mask = mask[y1:y2, x1:x2]
        local_input = self._normalize(local_rgb, local_mask)

        return global_input, local_input, label
