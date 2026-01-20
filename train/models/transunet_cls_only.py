"""
TransUNet ResNet-50 仅分类模型
用于多任务学习消融实验 - 移除分割头

架构概述:
- Encoder: ResNet-50 (前三个 Stage) + Transformer Encoder
- 输入: RGB (3) + Target Mask (1) = 4 通道
- Decoder: 仅分类头 (Classification Head)
- 无分割头，无 skip connections

实验目的:
验证分割任务是否对分类任务起到正则化/特征增强作用
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
    ClassificationHead,
)


@dataclass
class TransUNetClsOnlyConfig(TransUNetConfig):
    """TransUNet ResNet-50 仅分类配置"""
    pass  # 继承所有配置


class TransUNetClsOnly(nn.Module):
    """
    TransUNet with ResNet-50 Encoder - 仅分类版本
    
    用于消融实验：验证分割任务的辅助作用
    - 移除分割头 (SegmentationHead)
    - 移除 skip connections
    - 只训练分类任务
    """
    
    def __init__(self, config: Optional[TransUNetClsOnlyConfig] = None):
        super().__init__()
        
        if config is None:
            config = TransUNetClsOnlyConfig()
        self.config = config
        
        # CNN Encoder (ResNet-50)
        self.cnn_encoder = ResNetEncoder(config)
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            in_channels=1024,  # ResNet layer3 输出
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
        
        # Classification Head (仅保留)
        self.cls_head = ClassificationHead(
            hidden_dim=config.hidden_dim,
            cls_hidden_dim=config.cls_hidden_dim,
            num_classes=config.num_classes,
            dropout=config.dropout
        )
        
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
            cls_logits: (B, 1)
        """
        B, C, H, W = x.shape
        
        # CNN Encoder (忽略 skip_features)
        cnn_features, _ = self.cnn_encoder(x)
        
        # Patch Embedding
        patch_embeds, feat_h, feat_w = self.patch_embed(cnn_features)
        
        # 添加 CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, patch_embeds], dim=1)
        
        # 位置编码
        tokens = self.pos_encoding(tokens)
        
        # Transformer
        encoded = self.transformer(tokens)
        
        # Classification
        cls_logits = self.cls_head(encoded)
        
        return cls_logits
    
    def get_num_params(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_transunet_cls_only(
    pretrained: bool = True,
    img_height: int = 480,
    img_width: int = 640,
    num_transformer_layers: int = 12,
    hidden_dim: int = 768,
    **kwargs
) -> TransUNetClsOnly:
    config = TransUNetClsOnlyConfig(
        pretrained=pretrained,
        img_height=img_height,
        img_width=img_width,
        num_layers=num_transformer_layers,
        hidden_dim=hidden_dim,
        **kwargs
    )
    return TransUNetClsOnly(config)


def transunet_cls_only_base(pretrained: bool = True, **kwargs) -> TransUNetClsOnly:
    return create_transunet_cls_only(
        pretrained=pretrained,
        num_transformer_layers=12,
        hidden_dim=768,
        num_heads=12,
        mlp_dim=3072,
        **kwargs
    )


def transunet_cls_only_small(pretrained: bool = True, **kwargs) -> TransUNetClsOnly:
    return create_transunet_cls_only(
        pretrained=pretrained,
        num_transformer_layers=6,
        hidden_dim=512,
        num_heads=8,
        mlp_dim=2048,
        **kwargs
    )


def transunet_cls_only_tiny(pretrained: bool = True, **kwargs) -> TransUNetClsOnly:
    return create_transunet_cls_only(
        pretrained=pretrained,
        num_transformer_layers=4,
        hidden_dim=384,
        num_heads=6,
        mlp_dim=1536,
        **kwargs
    )


if __name__ == "__main__":
    model = transunet_cls_only_base()
    print(f"参数量: {model.get_num_params() / 1e6:.2f}M")
    
    x = torch.randn(2, 4, 480, 640)
    cls_logits = model(x)
    print(f"输入: {x.shape}, 输出: {cls_logits.shape}")
