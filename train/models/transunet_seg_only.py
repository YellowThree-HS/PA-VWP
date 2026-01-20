"""
TransUNet ResNet-50 仅分割模型
用于多任务学习消融实验 - 移除分类头

架构概述:
- Encoder: ResNet-50 (前三个 Stage) + Transformer Encoder
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
    TransUNetConfig,
    ResNetEncoder,
    PatchEmbedding,
    PositionalEncoding,
    TransformerEncoder,
    SegmentationHead,
)


@dataclass
class TransUNetSegOnlyConfig(TransUNetConfig):
    """TransUNet ResNet-50 仅分割配置"""
    pass


class TransUNetSegOnly(nn.Module):
    """
    TransUNet with ResNet-50 Encoder - 仅分割版本
    
    用于消融实验：验证单独训练分割任务的性能
    - 移除分类头 (ClassificationHead)
    - 移除 CLS Token
    - 只训练分割任务
    """
    
    def __init__(self, config: Optional[TransUNetSegOnlyConfig] = None):
        super().__init__()
        
        if config is None:
            config = TransUNetSegOnlyConfig()
        self.config = config
        
        # CNN Encoder (ResNet-50)
        self.cnn_encoder = ResNetEncoder(config)
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            in_channels=1024,
            hidden_dim=config.hidden_dim
        )
        
        # 位置编码 (不需要 +1，没有 CLS Token)
        max_patches = (config.img_height // 16) * (config.img_width // 16)
        self.pos_encoding = PositionalEncoding(
            hidden_dim=config.hidden_dim,
            max_len=max_patches,
            dropout=config.dropout
        )
        
        # Transformer Encoder
        self.transformer = TransformerEncoder(config)
        
        # Segmentation Head (仅保留)
        self.seg_head = SegmentationHead(config)
        
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
            x: 输入张量 (B, 4, H, W)
        Returns:
            seg_mask: (B, 1, H, W)
        """
        B, C, H, W = x.shape
        
        # CNN Encoder
        cnn_features, skip_features = self.cnn_encoder(x)
        
        # Patch Embedding
        patch_embeds, feat_h, feat_w = self.patch_embed(cnn_features)
        
        # 位置编码 (没有 CLS Token)
        tokens = self.pos_encoding(patch_embeds)
        
        # Transformer
        encoded = self.transformer(tokens)
        
        # Segmentation
        seg_mask = self.seg_head(encoded, feat_h, feat_w, skip_features)
        
        if seg_mask.shape[2:] != (H, W):
            seg_mask = F.interpolate(seg_mask, size=(H, W), mode='bilinear', align_corners=False)
            
        return seg_mask
    
    def get_num_params(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_transunet_seg_only(
    pretrained: bool = True,
    img_height: int = 480,
    img_width: int = 640,
    num_transformer_layers: int = 12,
    hidden_dim: int = 768,
    **kwargs
) -> TransUNetSegOnly:
    config = TransUNetSegOnlyConfig(
        pretrained=pretrained,
        img_height=img_height,
        img_width=img_width,
        num_layers=num_transformer_layers,
        hidden_dim=hidden_dim,
        **kwargs
    )
    return TransUNetSegOnly(config)


def transunet_seg_only_base(pretrained: bool = True, **kwargs) -> TransUNetSegOnly:
    return create_transunet_seg_only(
        pretrained=pretrained,
        num_transformer_layers=12,
        hidden_dim=768,
        num_heads=12,
        mlp_dim=3072,
        **kwargs
    )


def transunet_seg_only_small(pretrained: bool = True, **kwargs) -> TransUNetSegOnly:
    return create_transunet_seg_only(
        pretrained=pretrained,
        num_transformer_layers=6,
        hidden_dim=512,
        num_heads=8,
        mlp_dim=2048,
        **kwargs
    )


def transunet_seg_only_tiny(pretrained: bool = True, **kwargs) -> TransUNetSegOnly:
    return create_transunet_seg_only(
        pretrained=pretrained,
        num_transformer_layers=4,
        hidden_dim=384,
        num_heads=6,
        mlp_dim=1536,
        **kwargs
    )


if __name__ == "__main__":
    model = transunet_seg_only_base()
    print(f"参数量: {model.get_num_params() / 1e6:.2f}M")
    
    x = torch.randn(2, 4, 480, 640)
    seg_mask = model(x)
    print(f"输入: {x.shape}, 输出: {seg_mask.shape}")
