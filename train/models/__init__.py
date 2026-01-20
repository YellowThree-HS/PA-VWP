"""
BoxWorld-MVP 模型模块
Attention-Based TransUNet for Physical Stability Prediction

支持的 Backbone:
- ResNet-50 (原版)

支持的输入:
- RGB + Mask (4 通道)
- RGBD + Mask (5 通道)

消融实验模型:
- *_cls_only: 仅分类 (无分割头)
- *_seg_only: 仅分割 (无分类头)
"""

# ResNet-50 版本
from .transunet import TransUNet, TransUNetConfig
from .transunet_rgbd import TransUNetRGBD, TransUNetRGBDConfig

__all__ = [
    # ResNet-50 版本
    'TransUNet', 'TransUNetConfig',
    'TransUNetRGBD', 'TransUNetRGBDConfig',
]
