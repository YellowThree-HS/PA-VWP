"""
TransUNet ConvNeXt 仅分割模型
用于多任务学习消融实验 - 移除分类头

架构概述:
- Encoder: ConvNeXt (前三个 Stage) + Transformer Encoder
- 输入: RGB (3) + Target Mask (1) = 4 通道
- Decoder: 仅分割头 (Segmentation Head with Skip Connections)
- 无分类头，无 CLS Token

实验目的:
验证分割任务单独训练的性能，与多任务模型对比
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
)
from .transunet_convnext import (
    TransUNetConvNeXtConfig,
    ConvNeXtEncoder,
    SegmentationHeadConvNeXt,
)


@dataclass
class TransUNetConvNeXtSegOnlyConfig(TransUNetConvNeXtConfig):
    """TransUNet ConvNeXt 仅分割配置"""
    pass  # 继承所有配置，不需要额外参数


class TransUNetConvNeXtSegOnly(nn.Module):
    """
    TransUNet with ConvNeXt Encoder - 仅分割版本
    
    用于消融实验：验证单独训练分割任务的性能
    - 移除分类头 (ClassificationHead)
    - 移除 CLS Token
    - 只训练分割任务
    """
    
    def __init__(self, config: Optional[TransUNetConvNeXtSegOnlyConfig] = None):
        super().__init__()
        
        if config is None:
            config = TransUNetConvNeXtSegOnlyConfig()
        self.config = config
        
        # CNN Encoder (ConvNeXt) - 复用现有实现
        self.cnn_encoder = ConvNeXtEncoder(config)
        
        # Patch Embedding: CNN 特征 -> Transformer 序列
        self.patch_embed = PatchEmbedding(
            in_channels=config.cnn_out_channels,  # 512 for ConvNeXt-Base
            hidden_dim=config.hidden_dim
        )
        
        # 位置编码 (不需要 +1，因为没有 CLS Token)
        max_patches = (config.img_height // 16) * (config.img_width // 16)
        self.pos_encoding = PositionalEncoding(
            hidden_dim=config.hidden_dim,
            max_len=max_patches,
            dropout=config.dropout
        )
        
        # ❌ 移除: self.cls_token (CLS Token)
        
        # Transformer Encoder
        self.transformer = TransformerEncoder(config)
        
        # ❌ 移除: self.cls_head (分类头)
        
        # Segmentation Head (仅保留分割头)
        self.seg_head = SegmentationHeadConvNeXt(config)
        
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
            x: 输入张量 (B, 4, H, W) - RGB + Target Mask
            
        Returns:
            seg_mask: 分割掩码 (B, 1, H, W)
        """
        B, C, H, W = x.shape
        
        # CNN Encoder (需要 skip_features 用于分割)
        cnn_features, skip_features = self.cnn_encoder(x)
        
        # Patch Embedding
        patch_embeds, feat_h, feat_w = self.patch_embed(cnn_features)
        
        # 位置编码 (没有 CLS Token)
        tokens = self.pos_encoding(patch_embeds)
        
        # Transformer Encoder
        encoded = self.transformer(tokens)
        
        # Segmentation (仅分割)
        seg_mask = self.seg_head(encoded, feat_h, feat_w, skip_features)
        
        # 调整分割掩码尺寸
        if seg_mask.shape[2:] != (H, W):
            seg_mask = F.interpolate(seg_mask, size=(H, W), mode='bilinear', align_corners=False)
            
        return seg_mask
    
    def get_num_params(self, trainable_only: bool = True) -> int:
        """获取模型参数数量"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_transunet_convnext_seg_only(
    pretrained: bool = True,
    img_height: int = 480,
    img_width: int = 640,
    num_transformer_layers: int = 12,
    hidden_dim: int = 768,
    **kwargs
) -> TransUNetConvNeXtSegOnly:
    """工厂函数：创建仅分割模型"""
    config = TransUNetConvNeXtSegOnlyConfig(
        pretrained=pretrained,
        img_height=img_height,
        img_width=img_width,
        num_layers=num_transformer_layers,
        hidden_dim=hidden_dim,
        **kwargs
    )
    return TransUNetConvNeXtSegOnly(config)


# 模型变体
def transunet_convnext_seg_only_base(pretrained: bool = True, **kwargs) -> TransUNetConvNeXtSegOnly:
    """TransUNet-ConvNeXt-SegOnly-Base: 12 层 Transformer, 768 hidden dim"""
    return create_transunet_convnext_seg_only(
        pretrained=pretrained,
        num_transformer_layers=12,
        hidden_dim=768,
        num_heads=12,
        mlp_dim=3072,
        **kwargs
    )


def transunet_convnext_seg_only_small(pretrained: bool = True, **kwargs) -> TransUNetConvNeXtSegOnly:
    """TransUNet-ConvNeXt-SegOnly-Small: 6 层 Transformer, 512 hidden dim"""
    return create_transunet_convnext_seg_only(
        pretrained=pretrained,
        num_transformer_layers=6,
        hidden_dim=512,
        num_heads=8,
        mlp_dim=2048,
        **kwargs
    )


def transunet_convnext_seg_only_tiny(pretrained: bool = True, **kwargs) -> TransUNetConvNeXtSegOnly:
    """TransUNet-ConvNeXt-SegOnly-Tiny: 4 层 Transformer, 384 hidden dim"""
    return create_transunet_convnext_seg_only(
        pretrained=pretrained,
        num_transformer_layers=4,
        hidden_dim=384,
        num_heads=6,
        mlp_dim=1536,
        **kwargs
    )


if __name__ == "__main__":
    # 测试模型
    config = TransUNetConvNeXtSegOnlyConfig()
    model = TransUNetConvNeXtSegOnly(config)
    
    # 打印模型信息
    print(f"TransUNet-ConvNeXt-SegOnly 参数量: {model.get_num_params() / 1e6:.2f}M (可训练)")
    print(f"TransUNet-ConvNeXt-SegOnly 参数量: {model.get_num_params(trainable_only=False) / 1e6:.2f}M (总计)")
    
    # 测试前向传播
    x = torch.randn(2, 4, 480, 640)
    seg_mask = model(x)
    
    print(f"输入尺寸: {x.shape}")
    print(f"分割输出尺寸: {seg_mask.shape}")  # (2, 1, 480, 640)
