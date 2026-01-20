"""
BoxWorld-MVP 模型模块
Attention-Based TransUNet for Physical Stability Prediction

支持的 Backbone:
- ResNet-50 (原版)
- ConvNeXt-Base (现代化卷积网络)

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

# ConvNeXt 版本
from .transunet_convnext import TransUNetConvNeXt, TransUNetConvNeXtConfig
from .transunet_convnext_rgbd import TransUNetConvNeXtRGBD, TransUNetConvNeXtRGBDConfig

# ConvNeXt 消融实验模型 - 仅分类
from .transunet_convnext_cls_only import TransUNetConvNeXtClsOnly, TransUNetConvNeXtClsOnlyConfig
from .transunet_convnext_rgbd_cls_only import TransUNetConvNeXtRGBDClsOnly, TransUNetConvNeXtRGBDClsOnlyConfig

# ConvNeXt 消融实验模型 - 仅分割
from .transunet_convnext_seg_only import TransUNetConvNeXtSegOnly, TransUNetConvNeXtSegOnlyConfig
from .transunet_convnext_rgbd_seg_only import TransUNetConvNeXtRGBDSegOnly, TransUNetConvNeXtRGBDSegOnlyConfig

__all__ = [
    # ResNet-50 版本
    'TransUNet', 'TransUNetConfig',
    'TransUNetRGBD', 'TransUNetRGBDConfig',
    # ConvNeXt 版本
    'TransUNetConvNeXt', 'TransUNetConvNeXtConfig',
    'TransUNetConvNeXtRGBD', 'TransUNetConvNeXtRGBDConfig',
    # 消融实验 - 仅分类
    'TransUNetConvNeXtClsOnly', 'TransUNetConvNeXtClsOnlyConfig',
    'TransUNetConvNeXtRGBDClsOnly', 'TransUNetConvNeXtRGBDClsOnlyConfig',
    # 消融实验 - 仅分割
    'TransUNetConvNeXtSegOnly', 'TransUNetConvNeXtSegOnlyConfig',
    'TransUNetConvNeXtRGBDSegOnly', 'TransUNetConvNeXtRGBDSegOnlyConfig',
]
