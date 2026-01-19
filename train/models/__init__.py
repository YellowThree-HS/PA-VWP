"""
BoxWorld-MVP 模型模块
Attention-Based TransUNet for Physical Stability Prediction

支持的 Backbone:
- ResNet-50 (原版)
- ConvNeXt-Base (现代化卷积网络)

支持的输入:
- RGB + Mask (4 通道)
- RGBD + Mask (5 通道)
"""

# ResNet-50 版本
from .transunet import TransUNet, TransUNetConfig
from .transunet_rgbd import TransUNetRGBD, TransUNetRGBDConfig

# ConvNeXt 版本
from .transunet_convnext import TransUNetConvNeXt, TransUNetConvNeXtConfig
from .transunet_convnext_rgbd import TransUNetConvNeXtRGBD, TransUNetConvNeXtRGBDConfig

__all__ = [
    # ResNet-50 版本
    'TransUNet', 'TransUNetConfig',
    'TransUNetRGBD', 'TransUNetRGBDConfig',
    # ConvNeXt 版本
    'TransUNetConvNeXt', 'TransUNetConvNeXtConfig',
    'TransUNetConvNeXtRGBD', 'TransUNetConvNeXtRGBDConfig',
]
