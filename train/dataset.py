"""
BoxWorld 数据集加载器

数据格式:
- 输入: RGB 图像 (640x480) + 被移除箱子掩码 (Target Mask)
- 输出: 
  - 稳定性标签 (0=不稳定, 1=稳定)
  - 受影响区域掩码 (只在不稳定时有意义)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import torchvision.transforms.functional as TF
import random


class BoxWorldDataset(Dataset):
    """BoxWorld 物理稳定性预测数据集"""
    
    def __init__(
        self,
        data_dirs: List[str],
        transform: Optional[Callable] = None,
        img_size: Tuple[int, int] = (480, 640),  # (H, W)
        include_depth: bool = False,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            data_dirs: 数据集目录列表 (如 ['dataset3', 'dataset_messy_small'])
            transform: 数据增强变换
            img_size: 图像尺寸 (H, W)
            include_depth: 是否包含深度信息
            max_samples: 最大样本数 (用于调试)
        """
        self.data_dirs = [Path(d) for d in data_dirs]
        self.transform = transform
        self.img_size = img_size
        self.include_depth = include_depth
        
        # 收集所有样本
        print(f"开始收集样本，数据目录: {self.data_dirs}")
        self.samples = self._collect_samples()
        
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
            
        print(f"加载了 {len(self.samples)} 个样本")
        
        # 统计类别分布
        self._print_class_distribution()
        
    def _collect_samples(self) -> List[Dict]:
        """收集所有有效样本"""
        samples = []
        
        for data_dir in self.data_dirs:
            if not data_dir.exists():
                print(f"警告: 目录不存在 {data_dir}")
                continue
            
            print(f"  扫描目录: {data_dir}")
            round_dirs = [d for d in sorted(data_dir.iterdir()) if d.is_dir() and d.name.startswith('round_')]
            print(f"  找到 {len(round_dirs)} 个 round 目录")
                
            # 遍历所有 round 目录
            for idx, view_dir in enumerate(round_dirs):
                if idx > 0 and idx % 100 == 0:
                    print(f"    处理进度: {idx}/{len(round_dirs)} ({idx*100//len(round_dirs)}%), 已收集 {len(samples)} 个样本")
                if not view_dir.is_dir():
                    continue
                if not view_dir.name.startswith('round_'):
                    continue
                    
                # 检查必要文件
                rgb_path = view_dir / 'rgb.png'
                removals_dir = view_dir / 'removals'
                
                if not rgb_path.exists() or not removals_dir.exists():
                    continue
                    
                # 遍历每个移除测试
                for removal_dir in removals_dir.iterdir():
                    if not removal_dir.is_dir():
                        continue
                        
                    result_path = removal_dir / 'result.json'
                    removed_mask_path = removal_dir / 'removed_mask.png'
                    affected_mask_path = removal_dir / 'affected_mask.png'
                    
                    if not all(p.exists() for p in [result_path, removed_mask_path, affected_mask_path]):
                        continue
                        
                    # 读取结果
                    try:
                        with open(result_path, 'r') as f:
                            result = json.load(f)
                    except:
                        continue
                        
                    sample = {
                        'rgb_path': str(rgb_path),
                        'removed_mask_path': str(removed_mask_path),
                        'affected_mask_path': str(affected_mask_path),
                        'is_stable': result.get('is_stable', result.get('stability_label', 1) == 1),
                        'stability_label': 1 if result.get('is_stable', result.get('stability_label', 1) == 1) else 0,
                        'view_dir': str(view_dir),
                        'removal_id': removal_dir.name,
                    }
                    
                    # 可选：深度图
                    if self.include_depth:
                        depth_path = view_dir / 'depth.npy'
                        if depth_path.exists():
                            sample['depth_path'] = str(depth_path)
                        else:
                            continue
                            
                    samples.append(sample)
                    
        return samples
    
    def _print_class_distribution(self):
        """打印类别分布"""
        if len(self.samples) == 0:
            print("警告: 数据集为空!")
            return
            
        stable_count = sum(1 for s in self.samples if s['stability_label'] == 1)
        unstable_count = len(self.samples) - stable_count
        
        print(f"类别分布:")
        print(f"  稳定 (label=1): {stable_count} ({100*stable_count/len(self.samples):.1f}%)")
        print(f"  不稳定 (label=0): {unstable_count} ({100*unstable_count/len(self.samples):.1f}%)")
        
    def get_class_weights(self) -> torch.Tensor:
        """计算类别权重 (用于处理类别不平衡)"""
        stable_count = sum(1 for s in self.samples if s['stability_label'] == 1)
        unstable_count = len(self.samples) - stable_count
        
        # 避免除零
        if stable_count == 0 or unstable_count == 0 or len(self.samples) == 0:
            return torch.tensor([1.0, 1.0])
        
        # 权重与类别频率成反比
        total = len(self.samples)
        weights = torch.tensor([
            total / (2 * unstable_count),  # 不稳定类权重
            total / (2 * stable_count),    # 稳定类权重
        ])
        
        return weights
    
    def get_sample_weights(self) -> List[float]:
        """获取每个样本的权重 (用于 WeightedRandomSampler)"""
        class_weights = self.get_class_weights()
        sample_weights = []
        
        for sample in self.samples:
            label = sample['stability_label']
            sample_weights.append(class_weights[label].item())
            
        return sample_weights
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # 加载 RGB 图像（直接加载为 PIL Image）
        rgb_pil = Image.open(sample['rgb_path']).convert('RGB')
        
        # 加载被移除箱子掩码 (Target Mask)
        removed_mask_pil = Image.open(sample['removed_mask_path']).convert('L')
        
        # 加载受影响区域掩码 (Ground Truth for segmentation)
        affected_mask_pil = Image.open(sample['affected_mask_path']).convert('L')
        
        # 稳定性标签
        stability_label = sample['stability_label']
        
        # 数据增强/预处理
        if self.transform is not None:
            rgb, [removed_mask, affected_mask] = self.transform(
                rgb_pil, [removed_mask_pil, affected_mask_pil]
            )
        else:
            # 默认转换（无增强）
            rgb = TF.to_tensor(rgb_pil)
            rgb = TF.normalize(rgb, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
            # 转换掩码
            removed_mask = torch.from_numpy(np.array(removed_mask_pil)).float() / 255.0
            affected_mask = torch.from_numpy(np.array(affected_mask_pil)).float() / 255.0
            
            # 确保掩码维度正确
            if removed_mask.dim() == 2:
                removed_mask = removed_mask.unsqueeze(0)  # (1, H, W)
            if affected_mask.dim() == 2:
                affected_mask = affected_mask.unsqueeze(0)  # (1, H, W)
            
        # 组合输入: RGB (3) + Target Mask (1) = 4 channels
        input_tensor = torch.cat([rgb, removed_mask], dim=0)  # (4, H, W)
        
        return {
            'input': input_tensor,                                    # (4, H, W)
            'stability_label': torch.tensor(stability_label).float(), # scalar
            'affected_mask': affected_mask,                           # (1, H, W)
            'rgb_path': sample['rgb_path'],
            'removal_id': sample['removal_id'],
        }


class TrainTransform:
    """训练数据增强（使用 PyTorch transforms）"""
    def __init__(self, img_size: Tuple[int, int] = (480, 640)):
        self.img_size = img_size
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        )
        
    def __call__(self, image, masks):
        """
        Args:
            image: PIL Image (RGB)
            masks: list of PIL Images (L mode)
        Returns:
            transformed image and masks as tensors
        """
        # 随机水平翻转
        if random.random() < 0.5:
            image = TF.hflip(image)
            masks = [TF.hflip(m) for m in masks]
        
        # 随机旋转和缩放
        if random.random() < 0.5:
            angle = random.uniform(-15, 15)
            scale = random.uniform(0.85, 1.15)
            translate = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))
            
            image = TF.affine(image, angle=angle, translate=translate, scale=scale, shear=0)
            masks = [TF.affine(m, angle=angle, translate=translate, scale=scale, shear=0) for m in masks]
        
        # 颜色抖动
        if random.random() < 0.5:
            image = self.color_jitter(image)
        
        # 调整尺寸
        image = TF.resize(image, self.img_size, Image.BILINEAR)
        masks = [TF.resize(m, self.img_size, Image.NEAREST) for m in masks]
        
        # 转换为张量并归一化
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # 掩码转换为张量
        masks = [torch.from_numpy(np.array(m)).float() / 255.0 for m in masks]
        masks = [m.unsqueeze(0) if m.dim() == 2 else m for m in masks]
        
        return image, masks


def get_train_transform(img_size: Tuple[int, int] = (480, 640)):
    """训练数据增强"""
    return TrainTransform(img_size)


class ValTransform:
    """验证/测试数据变换 (无增强，使用 PyTorch transforms)"""
    def __init__(self, img_size: Tuple[int, int] = (480, 640)):
        self.img_size = img_size
        
    def __call__(self, image, masks):
        """
        Args:
            image: PIL Image (RGB)
            masks: list of PIL Images (L mode)
        Returns:
            transformed image and masks as tensors
        """
        # 调整尺寸
        image = TF.resize(image, self.img_size, Image.BILINEAR)
        masks = [TF.resize(m, self.img_size, Image.NEAREST) for m in masks]
        
        # 转换为张量并归一化
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # 掩码转换为张量
        masks = [torch.from_numpy(np.array(m)).float() / 255.0 for m in masks]
        masks = [m.unsqueeze(0) if m.dim() == 2 else m for m in masks]
        
        return image, masks


def get_val_transform(img_size: Tuple[int, int] = (480, 640)):
    """验证/测试数据变换 (无增强)"""
    return ValTransform(img_size)


def create_dataloaders(
    train_dirs: List[str],
    val_dirs: Optional[List[str]] = None,
    test_dirs: Optional[List[str]] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    img_size: Tuple[int, int] = (480, 640),
    val_split: float = 0.2,
    use_weighted_sampler: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    创建训练、验证和测试数据加载器
    
    Args:
        train_dirs: 训练数据目录
        val_dirs: 验证数据目录 (如果为 None，则从训练集划分)
        test_dirs: 测试数据目录 (可选)
        batch_size: 批次大小
        num_workers: 数据加载线程数
        img_size: 图像尺寸 (H, W)
        val_split: 验证集比例 (仅当 val_dirs 为 None 时使用)
        use_weighted_sampler: 是否使用加权采样处理类别不平衡
        seed: 随机种子
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # 创建训练集
    train_transform = get_train_transform(img_size)
    val_transform = get_val_transform(img_size)
    
    if val_dirs is not None:
        # 使用独立的验证集目录
        train_dataset = BoxWorldDataset(train_dirs, transform=train_transform, img_size=img_size)
        val_dataset = BoxWorldDataset(val_dirs, transform=val_transform, img_size=img_size)
    else:
        # 从训练集划分验证集
        full_dataset = BoxWorldDataset(train_dirs, transform=None, img_size=img_size)
        
        # 随机划分
        torch.manual_seed(seed)
        n_samples = len(full_dataset)
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_val
        
        indices = torch.randperm(n_samples).tolist()
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # 创建子集
        train_samples = [full_dataset.samples[i] for i in train_indices]
        val_samples = [full_dataset.samples[i] for i in val_indices]
        
        train_dataset = BoxWorldDataset.__new__(BoxWorldDataset)
        train_dataset.samples = train_samples
        train_dataset.transform = train_transform
        train_dataset.img_size = img_size
        train_dataset.include_depth = False
        
        val_dataset = BoxWorldDataset.__new__(BoxWorldDataset)
        val_dataset.samples = val_samples
        val_dataset.transform = val_transform
        val_dataset.img_size = img_size
        val_dataset.include_depth = False
        
        print(f"\n划分数据集: 训练 {len(train_dataset)}, 验证 {len(val_dataset)}")
        
    # 创建采样器
    train_sampler = None
    if use_weighted_sampler:
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        
    # 创建 DataLoader
    # 优化设置：
    # - persistent_workers: 保持工作进程存活，避免每个 epoch 重建进程
    # - prefetch_factor: 预取更多批次，减少 GPU 等待时间
    # - num_workers: 根据系统 CPU 核心数调整（建议不超过 CPU 核心数）
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else 2,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else 2,
    )
    
    # 创建测试集
    test_loader = None
    if test_dirs is not None and len(test_dirs) > 0:
        test_dataset = BoxWorldDataset(test_dirs, transform=val_transform, img_size=img_size)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=4 if num_workers > 0 else 2,
        )
        print(f"测试集: {len(test_dataset)} 个样本")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试数据集
    import sys
    
    # 假设在项目根目录运行
    data_dirs = ['dataset_messy_small']
    
    dataset = BoxWorldDataset(data_dirs, transform=get_train_transform())
    
    print(f"\n测试数据加载:")
    sample = dataset[0]
    print(f"  输入尺寸: {sample['input'].shape}")
    print(f"  稳定性标签: {sample['stability_label']}")
    print(f"  受影响掩码尺寸: {sample['affected_mask'].shape}")
    
    # 测试 DataLoader
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dirs=data_dirs,
        batch_size=4,
        num_workers=0,
    )
    
    print(f"\nDataLoader 测试:")
    for batch in train_loader:
        print(f"  批次输入尺寸: {batch['input'].shape}")
        print(f"  批次标签: {batch['stability_label']}")
        break
