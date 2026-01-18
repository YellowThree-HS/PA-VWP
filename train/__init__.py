"""
BoxWorld TransUNet 训练模块

用于物理稳定性预测的 Attention-Based TransUNet 实现
"""

from .config import Config, get_config
from .dataset import BoxWorldDataset, create_dataloaders
from .models import TransUNet, TransUNetConfig

__all__ = [
    'Config',
    'get_config',
    'BoxWorldDataset',
    'create_dataloaders',
    'TransUNet',
    'TransUNetConfig',
]
