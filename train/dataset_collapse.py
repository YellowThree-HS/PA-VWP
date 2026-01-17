"""
CollapseNet 数据集
支持多任务学习: 稳定性分类 + 受影响区域分割
"""

import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CollapseDataset(Dataset):
    """坍塌预测多任务数据集

    每个样本包含:
    - RGB图像 (H, W, 3)
    - 目标箱子掩码 (H, W, 1)
    - 稳定性标签 (0或1)
    - 受影响区域掩码 (H, W, 1) - 坍塌时受影响区域

    数据目录结构:
        dataset/
        ├── round_000/
        │   ├── initial/
        │   │   └── rgb.png
        │   └── removals/
        │       ├── 0/
        │       │   ├── mask.png          # 目标箱子掩码
        │       │   ├── affected_mask.png # 受影响区域掩码 (新增)
        │       │   └── result.json       # 包含 stability_label
        │       └── 1/
        │           ├── mask.png
        │           ├── affected_mask.png
        │           └── result.json
    """

    def __init__(
        self,
        data_dir: str,
        transform=None,
        image_size: int = 224,
        use_segmentation: bool = True,
    ):
        """
        Args:
            data_dir: 数据集根目录
            transform: 图像变换
            image_size: 输出图像大小
            use_segmentation: 是否加载分割掩码
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.use_segmentation = use_segmentation
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

                if not os.path.exists(mask_path) or not os.path.exists(result_path):
                    continue

                # 检查分割掩码是否存在
                affected_path = os.path.join(sample_dir, "affected_mask.png")
                if self.use_segmentation and not os.path.exists(affected_path):
                    print(f"Warning: Missing affected_mask.png in {sample_dir}")
                    continue

                self.samples.append({
                    "rgb_path": rgb_path,
                    "mask_path": mask_path,
                    "affected_path": affected_path if self.use_segmentation else None,
                    "result_path": result_path
                })

        print(f"Found {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # === 1. 读取RGB图像 ===
        rgb = cv2.imread(sample["rgb_path"])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # === 2. 读取目标掩码 ===
        mask = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)

        # === 3. 读取稳定性标签 ===
        with open(sample["result_path"], "r") as f:
            result = json.load(f)
        stability_label = result["stability_label"]

        # === 4. 读取分割掩码 (受影响区域) ===
        if self.use_segmentation and sample["affected_path"] is not None:
            affected_mask = cv2.imread(sample["affected_path"], cv2.IMREAD_GRAYSCALE)
        else:
            affected_mask = np.zeros_like(mask)

        # === 5. 图像预处理 ===
        # Resize
        rgb = cv2.resize(rgb, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        affected_mask = cv2.resize(affected_mask, (self.image_size, self.image_size))

        # 归一化掩码到0-1
        mask = (mask > 127).astype(np.float32)
        affected_mask = (affected_mask > 127).astype(np.float32)

        # === 6. 构建4通道输入 (RGB + Target Mask) ===
        rgb_normalized = rgb.astype(np.float32) / 255.0
        combined = np.concatenate([rgb_normalized, mask[:, :, np.newaxis]], axis=2)  # (H, W, 4)

        # 转换为tensor并归一化
        combined = torch.from_numpy(combined).permute(2, 0, 1)  # (4, H, W)
        mean = torch.tensor([0.485, 0.456, 0.406, 0.0]).view(4, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225, 1.0]).view(4, 1, 1)
        combined = (combined - mean) / std

        # 分割掩码转为tensor
        affected_mask_tensor = torch.from_numpy(affected_mask).unsqueeze(0)  # (1, H, W)

        # 标签
        stability_label = torch.tensor(stability_label, dtype=torch.float32)

        return combined, stability_label, affected_mask_tensor


class TwoStreamCollapseDataset(Dataset):
    """双流 CollapseDataset (全局 + 局部)

    适用于需要局部裁剪视图的场景
    """

    def __init__(
        self,
        data_dir: str,
        transform=None,
        image_size: int = 224,
        crop_size: int = 160,
        padding_ratio: float = 1.5,
        use_segmentation: bool = True,
    ):
        """
        Args:
            data_dir: 数据集根目录
            image_size: 全局图像大小
            crop_size: 局部裁剪目标大小
            padding_ratio: 裁剪区域扩展比例
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.crop_size = crop_size
        self.padding_ratio = padding_ratio
        self.use_segmentation = use_segmentation
        self.samples = []

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406, 0.0],
                    std=[0.229, 0.224, 0.225, 1.0]
                )
            ])
        else:
            self.transform = transform

        self._scan_samples()

    def _scan_samples(self):
        """扫描数据目录"""
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

                if not os.path.exists(mask_path) or not os.path.exists(result_path):
                    continue

                affected_path = os.path.join(sample_dir, "affected_mask.png")
                if self.use_segmentation and not os.path.exists(affected_path):
                    continue

                self.samples.append({
                    "rgb_path": rgb_path,
                    "mask_path": mask_path,
                    "affected_path": affected_path if self.use_segmentation else None,
                    "result_path": result_path
                })

        print(f"Found {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def _extract_crop(self, rgb, mask):
        """从目标掩码提取裁剪区域"""
        # 找到掩码边界
        contours, _ = cv2.findContours(
            (mask > 127).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            # 如果没找到轮廓，使用中心区域
            h, w = mask.shape
            cx, cy = w // 2, h // 2
            crop_size = min(w, h) // 2
            x1, y1 = cx - crop_size, cy - crop_size
            x2, y2 = cx + crop_size, cy + crop_size
        else:
            # 使用最大轮廓的边界框
            x, y, cw, ch = cv2.boundingRect(max(contours, key=cv2.contourArea))
            cx, cy = x + cw // 2, y + ch // 2

            # 扩展边界
            expand = int(max(cw, ch) * (self.padding_ratio - 1) / 2)
            half_size = max(cw, ch) // 2 + expand

            x1 = max(0, cx - half_size)
            y1 = max(0, cy - half_size)
            x2 = min(rgb.shape[1], cx + half_size)
            y2 = min(rgb.shape[0], cy + half_size)

        # 裁剪
        crop_rgb = rgb[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]

        # 填充到目标大小
        h, w = crop_rgb.shape[:2]
        target_h, target_w = self.crop_size, self.crop_size

        # 创建画布
        padded_rgb = np.zeros((target_h, target_w, 3), dtype=np.float32)
        padded_mask = np.zeros((target_h, target_w), dtype=np.float32)

        # 计算填充位置
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        top = pad_h // 2
        left = pad_w // 2

        # 缩放以适应
        if h > target_h or w > target_w:
            scale = min(target_h / h, target_w / w)
            new_h, new_w = int(h * scale), int(w * scale)
            crop_rgb = cv2.resize(crop_rgb, (new_w, new_h))
            crop_mask = cv2.resize(crop_mask, (new_w, new_h))
            h, w = new_h, new_w
            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)
            top = pad_h // 2
            left = pad_w // 2

        # 放置到画布中心
        padded_rgb[top:top+h, left:left+w] = crop_rgb
        padded_mask[top:top+h, left:left+w] = crop_mask

        return padded_rgb, padded_mask

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # === 全局视图 ===
        rgb = cv2.imread(sample["rgb_path"])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)

        with open(sample["result_path"], "r") as f:
            result = json.load(f)
        stability_label = result["stability_label"]

        if self.use_segmentation and sample["affected_path"] is not None:
            affected_mask = cv2.imread(sample["affected_path"], cv2.IMREAD_GRAYSCALE)
        else:
            affected_mask = np.zeros_like(mask)

        # Resize 全局图像
        rgb = cv2.resize(rgb, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        affected_mask = cv2.resize(affected_mask, (self.image_size, self.image_size))

        mask = (mask > 127).astype(np.float32)
        affected_mask = (affected_mask > 127).astype(np.float32)

        # 构建4通道全局输入
        rgb_normalized = rgb.astype(np.float32) / 255.0
        global_input = np.concatenate([rgb_normalized, mask[:, :, np.newaxis]], axis=2)

        global_input = torch.from_numpy(global_input).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406, 0.0]).view(4, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225, 1.0]).view(4, 1, 1)
        global_input = (global_input - mean) / std

        # === 局部视图 (仅RGB) ===
        crop_rgb, _ = self._extract_crop(rgb, mask)
        crop_rgb = crop_rgb.astype(np.float32) / 255.0

        # 局部视图使用ImageNet归一化 (3通道)
        crop_tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1)
        mean3 = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std3 = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        local_input = (crop_tensor - mean3) / std3

        # === 分割掩码 ===
        affected_mask_tensor = torch.from_numpy(affected_mask).unsqueeze(0)

        stability_label = torch.tensor(stability_label, dtype=torch.float32)

        return global_input, local_input, stability_label, affected_mask_tensor
