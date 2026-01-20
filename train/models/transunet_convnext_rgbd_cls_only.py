"""
TransUNet ConvNeXt RGBD 仅分类模型
用于多任务学习消融实验 - 移除分割头 (带深度图输入)

架构概述:
- Encoder: ConvNeXt (前三个 Stage) + Transformer Encoder
- 输入: RGB (3) + Depth (1) + Target Mask (1) = 5 通道
- Decoder: 仅分类头 (Classification Head)
- 无分割头，无 skip connections

实验目的:
验证分割任务是否对分类任务起到正则化/特征增强作用 (RGBD 版本)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用现有模块
from .transunet import (
    PatchEmbedding,
    PositionalEncoding,
    TransformerEncoder,
    ClassificationHead,
)
from .transunet_convnext_rgbd import (
    TransUNetConvNeXtRGBDConfig,
    ConvNeXtEncoderRGBD,
)


@dataclass
class TransUNetConvNeXtRGBDClsOnlyConfig(TransUNetConvNeXtRGBDConfig):
    """TransUNet ConvNeXt RGBD 仅分类配置"""
    pass  # 继承所有配置，不需要额外参数


class TransUNetConvNeXtRGBDClsOnly(nn.Module):
    """
    TransUNet with ConvNeXt Encoder - RGBD 仅分类版本
    
    用于消融实验：验证分割任务的辅助作用 (RGBD)
    - 移除分割头 (SegmentationHead)
    - 移除 skip connections
    - 只训练分类任务
    - 输入: RGB (3) + Depth (1) + Target Mask (1) = 5 通道
    """
    
    def __init__(self, config: Optional[TransUNetConvNeXtRGBDClsOnlyConfig] = None):
        super().__init__()
        
        if config is None:
            config = TransUNetConvNeXtRGBDClsOnlyConfig()
        self.config = config
        
        # CNN Encoder (ConvNeXt RGBD) - 复用现有实现
        self.cnn_encoder = ConvNeXtEncoderRGBD(config)
        
        # Patch Embedding: CNN 特征 -> Transformer 序列
        self.patch_embed = PatchEmbedding(
            in_channels=config.cnn_out_channels,  # 512 for ConvNeXt-Base
            hidden_dim=config.hidden_dim
        )
        
        # 位置编码
        max_patches = (config.img_height // 16) * (config.img_width // 16)
        self.pos_encoding = PositionalEncoding(
            hidden_dim=config.hidden_dim,
            max_len=max_patches + 1,
            dropout=config.dropout
        )
        
        # CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Transformer Encoder
        self.transformer = TransformerEncoder(config)
        
        # Classification Head (仅保留分类头)
        self.cls_head = ClassificationHead(
            hidden_dim=config.hidden_dim,
            cls_hidden_dim=config.cls_hidden_dim,
            num_classes=config.num_classes,
            dropout=config.dropout
        )
        
        # ❌ 移除: self.seg_head (分割头)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 (B, 5, H, W) - RGB + Depth + Target Mask
            
        Returns:
            cls_logits: 分类 logits (B, 1)
        """
        B, C, H, W = x.shape
        
        # CNN Encoder (忽略 skip_features，不需要用于分割)
        cnn_features, _ = self.cnn_encoder(x)
        
        # Patch Embedding
        patch_embeds, feat_h, feat_w = self.patch_embed(cnn_features)
        
        # 添加 CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, patch_embeds], dim=1)
        
        # 位置编码
        tokens = self.pos_encoding(tokens)
        
        # Transformer Encoder
        encoded = self.transformer(tokens)
        
        # Classification (仅分类)
        cls_logits = self.cls_head(encoded)
        
        return cls_logits
    
    def get_num_params(self, trainable_only: bool = True) -> int:
        """获取模型参数数量"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_transunet_convnext_rgbd_cls_only(
    pretrained: bool = True,
    img_height: int = 480,
    img_width: int = 640,
    num_transformer_layers: int = 12,
    hidden_dim: int = 768,
    **kwargs
) -> TransUNetConvNeXtRGBDClsOnly:
    """工厂函数：创建 RGBD 仅分类模型"""
    config = TransUNetConvNeXtRGBDClsOnlyConfig(
        pretrained=pretrained,
        img_height=img_height,
        img_width=img_width,
        num_layers=num_transformer_layers,
        hidden_dim=hidden_dim,
        in_channels=5,
        **kwargs
    )
    return TransUNetConvNeXtRGBDClsOnly(config)


# 模型变体
def transunet_convnext_rgbd_cls_only_base(pretrained: bool = True, **kwargs) -> TransUNetConvNeXtRGBDClsOnly:
    """TransUNet-ConvNeXt-RGBD-ClsOnly-Base: 12 层 Transformer, 768 hidden dim"""
    return create_transunet_convnext_rgbd_cls_only(
        pretrained=pretrained,
        num_transformer_layers=12,
        hidden_dim=768,
        num_heads=12,
        mlp_dim=3072,
        **kwargs
    )


def transunet_convnext_rgbd_cls_only_small(pretrained: bool = True, **kwargs) -> TransUNetConvNeXtRGBDClsOnly:
    """TransUNet-ConvNeXt-RGBD-ClsOnly-Small: 6 层 Transformer, 512 hidden dim"""
    return create_transunet_convnext_rgbd_cls_only(
        pretrained=pretrained,
        num_transformer_layers=6,
        hidden_dim=512,
        num_heads=8,
        mlp_dim=2048,
        **kwargs
    )


def transunet_convnext_rgbd_cls_only_tiny(pretrained: bool = True, **kwargs) -> TransUNetConvNeXtRGBDClsOnly:
    """TransUNet-ConvNeXt-RGBD-ClsOnly-Tiny: 4 层 Transformer, 384 hidden dim"""
    return create_transunet_convnext_rgbd_cls_only(
        pretrained=pretrained,
        num_transformer_layers=4,
        hidden_dim=384,
        num_heads=6,
        mlp_dim=1536,
        **kwargs
    )


if __name__ == "__main__":
    # 测试模型
    config = TransUNetConvNeXtRGBDClsOnlyConfig()
    model = TransUNetConvNeXtRGBDClsOnly(config)
    
    # 打印模型信息
    print(f"TransUNet-ConvNeXt-RGBD-ClsOnly 参数量: {model.get_num_params() / 1e6:.2f}M (可训练)")
    print(f"TransUNet-ConvNeXt-RGBD-ClsOnly 参数量: {model.get_num_params(trainable_only=False) / 1e6:.2f}M (总计)")
    
    # 测试前向传播
    x = torch.randn(2, 5, 480, 640)  # 5 通道输入
    cls_logits = model(x)
    
    print(f"输入尺寸: {x.shape}")
    print(f"分类输出尺寸: {cls_logits.shape}")  # (2, 1)
