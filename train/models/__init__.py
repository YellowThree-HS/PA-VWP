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
- *_fusion: 分割感知双路融合 (分割特征增强分类)
"""

# ResNet-50 版本
from .transunet import TransUNet, TransUNetConfig
from .transunet_rgbd import TransUNetRGBD, TransUNetRGBDConfig

# 融合版本 (分割感知双路融合)
from .transunet_fusion import TransUNetFusion, TransUNetFusionConfig
from .transunet_rgbd_fusion import TransUNetRGBDFusion, TransUNetRGBDFusionConfig

__all__ = [
    # ResNet-50 版本
    'TransUNet', 'TransUNetConfig',
    'TransUNetRGBD', 'TransUNetRGBDConfig',
    # 融合版本
    'TransUNetFusion', 'TransUNetFusionConfig',
    'TransUNetRGBDFusion', 'TransUNetRGBDFusionConfig',
]
