"""
BoxWorld 稳定性预测数据集
"""

import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class BoxStabilityDataset(Dataset):
    """箱子稳定性预测数据集

    每个样本包含:
    - RGB图像 (H, W, 3)
    - 目标箱子掩码 (H, W, 1)
    - 稳定性标签 (0或1)
    """

    def __init__(self, data_dir: str, transform=None, image_size: int = 224):
        """
        Args:
            data_dir: 数据集根目录
            transform: 图像变换
            image_size: 输出图像大小
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.samples = []

        # 默认变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406, 0.0],  # 4通道，mask通道不归一化
                    std=[0.229, 0.224, 0.225, 1.0]
                )
            ])
        else:
            self.transform = transform

        # 扫描所有样本
        self._scan_samples()

    def _scan_samples(self):
        """扫描数据目录，收集所有样本"""
        for round_dir in sorted(os.listdir(self.data_dir)):
            round_path = os.path.join(self.data_dir, round_dir)
            if not os.path.isdir(round_path):
                continue

            # 检查是否有initial和removals目录
            initial_dir = os.path.join(round_path, "initial")
            removals_dir = os.path.join(round_path, "removals")

            if not os.path.exists(initial_dir) or not os.path.exists(removals_dir):
                continue

            rgb_path = os.path.join(initial_dir, "rgb.png")
            if not os.path.exists(rgb_path):
                continue

            # 扫描removals下的每个样本
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
        label = result["stability_label"]

        # Resize
        rgb = cv2.resize(rgb, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size))

        # 归一化mask到0-1
        mask = (mask > 127).astype(np.float32)

        # 拼接为4通道 (H, W, 4)
        rgb_normalized = rgb.astype(np.float32) / 255.0
        combined = np.concatenate([rgb_normalized, mask[:, :, np.newaxis]], axis=2)

        # 转换为tensor并应用变换
        combined = torch.from_numpy(combined).permute(2, 0, 1)  # (4, H, W)

        # 对RGB通道进行归一化
        mean = torch.tensor([0.485, 0.456, 0.406, 0.0]).view(4, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225, 1.0]).view(4, 1, 1)
        combined = (combined - mean) / std

        label = torch.tensor(label, dtype=torch.float32)

        return combined, label
