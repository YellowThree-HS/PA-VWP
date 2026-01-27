"""
TransUNet Seg-From-Encoded 模型架构
分割头直接接在 Transformer Encoded 层后面，不使用 CNN Encoder 跳跃连接

架构概述:
- Encoder: ResNet-50 (仅用于提取特征) + Transformer Encoder
- 分割头: 纯粹基于 Transformer encoded patch tokens 进行上采样，无 skip connections
- 分类头: 复用标准 TransUNet 分类头

实验目的:
验证移除 CNN Encoder 多尺度特征后，纯 Transformer 特征对分割任务的表达能力
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transunet import (
    TransUNetConfig,
    ResNetEncoder,
    PatchEmbedding,
    PositionalEncoding,
    TransformerEncoder,
    ClassificationHead,
)


@dataclass
class TransUNetSegFromEncodedConfig(TransUNetConfig):
    """TransUNet Seg-From-Encoded 配置"""
    # 分割解码器配置 (无 skip connections，通道数可调整)
    seg_decoder_channels: Tuple[int, ...] = (256, 128, 64, 32)
    seg_decoder_use_bn: bool = True  # 是否使用 BatchNorm


class UpBlockNoSkip(nn.Module):
    """
    上采样块，不使用跳跃连接
    纯粹进行双线性上采样 + 卷积精化
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        scale_factor: int = 2,
        use_bn: bool = True
    ):
        super().__init__()
        self.scale_factor = scale_factor
        
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        ]
        
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        
        layers.extend([
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        ])
        
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        
        layers.append(nn.ReLU(inplace=True))
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        return self.conv(x)


class SegmentationHeadFromEncoded(nn.Module):
    """
    分割头 - 直接从 Transformer Encoded 特征解码
    不使用 CNN Encoder 的跳跃连接
    
    结构:
    Transformer 输出 (B, num_patches, hidden_dim)
    -> Reshape 到 2D: (B, hidden_dim, H/16, W/16)
    -> 多级上采样 (4次 2x) 到原始尺寸
    """
    
    def __init__(
        self, 
        config: TransUNetSegFromEncodedConfig,
    ):
        super().__init__()
        self.config = config
        
        decoder_channels = config.seg_decoder_channels
        
        # 从 Transformer 特征恢复到 2D
        # Transformer 输出: (B, num_patches, hidden_dim)
        # 需要 reshape 回 (B, hidden_dim, H/16, W/16)，再投影到 decoder 通道
        self.reshape_proj = nn.Conv2d(config.hidden_dim, decoder_channels[0], 1)
        
        # 上采样路径 (无跳跃连接)
        self.up_blocks = nn.ModuleList()
        
        # UpBlock 1: H/16 -> H/8
        self.up_blocks.append(UpBlockNoSkip(
            decoder_channels[0], decoder_channels[1], 
            use_bn=config.seg_decoder_use_bn
        ))
        
        # UpBlock 2: H/8 -> H/4
        self.up_blocks.append(UpBlockNoSkip(
            decoder_channels[1], decoder_channels[2], 
            use_bn=config.seg_decoder_use_bn
        ))
        
        # UpBlock 3: H/4 -> H/2
        self.up_blocks.append(UpBlockNoSkip(
            decoder_channels[2], decoder_channels[3], 
            use_bn=config.seg_decoder_use_bn
        ))
        
        # UpBlock 4: H/2 -> H
        self.up_blocks.append(UpBlockNoSkip(
            decoder_channels[3], decoder_channels[3], 
            use_bn=config.seg_decoder_use_bn
        ))
        
        # 最终卷积
        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_channels[3], 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, config.seg_classes, 1),
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        feat_h: int, 
        feat_w: int,
        return_intermediate: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Args:
            x: Transformer 输出的 patch tokens (B, num_patches, hidden_dim)
            feat_h, feat_w: 特征图高度和宽度 (H/16, W/16)
            return_intermediate: 是否返回中间特征
            
        Returns:
            seg_mask: 分割掩码 (B, seg_classes, H, W)
            intermediate_features: 各 UpBlock 输出的特征列表 (如果 return_intermediate=True)
        """
        B = x.shape[0]
        
        # Reshape: (B, num_patches, hidden_dim) -> (B, hidden_dim, H/16, W/16)
        x = x.transpose(1, 2).reshape(B, -1, feat_h, feat_w)
        x = self.reshape_proj(x)  # (B, decoder_channels[0], H/16, W/16)
        
        intermediate_features = []
        
        # 上采样路径
        for i, up_block in enumerate(self.up_blocks):
            x = up_block(x)
            
            # 保存中间特征 (前3个 UpBlock 的输出)
            if return_intermediate and i < 3:
                intermediate_features.append(x)
            
        # 最终输出
        seg_mask = self.final_conv(x)
        
        if return_intermediate:
            return seg_mask, intermediate_features
        return seg_mask, None


class TransUNetSegFromEncoded(nn.Module):
    """
    TransUNet Seg-From-Encoded: 分割头直接接在 Transformer Encoded 层后面
    
    用于物理稳定性预测的双任务模型:
    - Task 1: 稳定性二分类 (stable / unstable) 
    - Task 2: 受影响区域分割 (纯 Transformer 特征，无 CNN skip connections)
    
    架构特点:
    1. CNN Encoder 只用于初始特征提取，不提供 skip connections
    2. 分割头直接从 Transformer encoded patch tokens 解码
    3. 分类头复用标准实现
    """
    
    def __init__(self, config: Optional[TransUNetSegFromEncodedConfig] = None):
        super().__init__()
        
        if config is None:
            config = TransUNetSegFromEncodedConfig()
        self.config = config
        
        # CNN Encoder (ResNet-50) - 仅用于初始特征提取
        self.cnn_encoder = ResNetEncoder(config)
        
        # Patch Embedding: CNN 特征 -> Transformer 序列
        self.patch_embed = PatchEmbedding(
            in_channels=1024,  # ResNet layer3 输出
            hidden_dim=config.hidden_dim
        )
        
        # 位置编码
        max_patches = (config.img_height // 16) * (config.img_width // 16)
        self.pos_encoding = PositionalEncoding(
            hidden_dim=config.hidden_dim,
            max_len=max_patches + 1,  # +1 for CLS token
            dropout=config.dropout
        )
        
        # CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Transformer Encoder
        self.transformer = TransformerEncoder(config)
        
        # Classification Head (复用标准实现)
        self.cls_head = ClassificationHead(
            hidden_dim=config.hidden_dim,
            cls_hidden_dim=config.cls_hidden_dim,
            num_classes=config.num_classes,
            dropout=config.dropout
        )
        
        # Segmentation Head (直接从 Encoded 特征解码)
        self.seg_head = SegmentationHeadFromEncoded(config)
        
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
                    
    def forward(
        self, 
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入张量 (B, 4, H, W) - RGB + Target Mask
            return_attention: 是否返回注意力权重 (用于可视化)
            
        Returns:
            cls_logits: 分类 logits (B, 1)
            seg_mask: 分割掩码 (B, 1, H, W)
        """
        B, C, H, W = x.shape
        
        # CNN Encoder (忽略 skip_features，不再使用)
        cnn_features, _ = self.cnn_encoder(x)
        # cnn_features: (B, 1024, H/16, W/16)
        
        # Patch Embedding
        patch_embeds, feat_h, feat_w = self.patch_embed(cnn_features)
        # patch_embeds: (B, H/16 * W/16, hidden_dim)
        
        # 添加 CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, patch_embeds], dim=1)
        
        # 位置编码
        tokens = self.pos_encoding(tokens)
        
        # Transformer Encoder
        encoded = self.transformer(tokens)
        # encoded: (B, 1 + num_patches, hidden_dim)
        
        # 分离 CLS Token 和 patch tokens
        patch_tokens = encoded[:, 1:, :]  # (B, num_patches, hidden_dim)
        
        # Task 1: Classification (使用全部 tokens)
        cls_logits = self.cls_head(encoded)
        
        # Task 2: Segmentation (直接从 patch tokens 解码，无 skip connections)
        seg_mask, _ = self.seg_head(patch_tokens, feat_h, feat_w)
        
        # 调整分割掩码尺寸到原始输入尺寸
        if seg_mask.shape[2:] != (H, W):
            seg_mask = F.interpolate(seg_mask, size=(H, W), mode='bilinear', align_corners=False)
            
        return cls_logits, seg_mask
    
    def get_num_params(self, trainable_only: bool = True) -> int:
        """获取模型参数数量"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_transunet_seg_from_encoded(
    pretrained: bool = True,
    img_height: int = 480,
    img_width: int = 640,
    num_transformer_layers: int = 12,
    hidden_dim: int = 768,
    **kwargs
) -> TransUNetSegFromEncoded:
    """工厂函数：创建 TransUNet Seg-From-Encoded 模型"""
    config = TransUNetSegFromEncodedConfig(
        pretrained=pretrained,
        img_height=img_height,
        img_width=img_width,
        num_layers=num_transformer_layers,
        hidden_dim=hidden_dim,
        **kwargs
    )
    return TransUNetSegFromEncoded(config)


# 模型变体
def transunet_seg_from_encoded_base(pretrained: bool = True, **kwargs) -> TransUNetSegFromEncoded:
    """TransUNet-SegFromEncoded-Base: 12 层 Transformer, 768 hidden dim"""
    return create_transunet_seg_from_encoded(
        pretrained=pretrained,
        num_transformer_layers=12,
        hidden_dim=768,
        num_heads=12,
        mlp_dim=3072,
        **kwargs
    )


def transunet_seg_from_encoded_small(pretrained: bool = True, **kwargs) -> TransUNetSegFromEncoded:
    """TransUNet-SegFromEncoded-Small: 6 层 Transformer, 512 hidden dim"""
    return create_transunet_seg_from_encoded(
        pretrained=pretrained,
        num_transformer_layers=6,
        hidden_dim=512,
        num_heads=8,
        mlp_dim=2048,
        **kwargs
    )


def transunet_seg_from_encoded_tiny(pretrained: bool = True, **kwargs) -> TransUNetSegFromEncoded:
    """TransUNet-SegFromEncoded-Tiny: 4 层 Transformer, 384 hidden dim (用于快速实验)"""
    return create_transunet_seg_from_encoded(
        pretrained=pretrained,
        num_transformer_layers=4,
        hidden_dim=384,
        num_heads=6,
        mlp_dim=1536,
        **kwargs
    )


def transunet_seg_from_encoded_micro(pretrained: bool = True, **kwargs) -> TransUNetSegFromEncoded:
    """TransUNet-SegFromEncoded-Micro: 2 层 Transformer, 192 hidden dim (超轻量级)"""
    return create_transunet_seg_from_encoded(
        pretrained=pretrained,
        num_transformer_layers=2,
        hidden_dim=192,
        num_heads=3,
        mlp_dim=768,
        **kwargs
    )


if __name__ == "__main__":
    # 测试模型
    config = TransUNetSegFromEncodedConfig()
    model = TransUNetSegFromEncoded(config)
    
    # 打印模型信息
    print(f"TransUNet Seg-From-Encoded 参数量: {model.get_num_params() / 1e6:.2f}M (可训练)")
    print(f"TransUNet Seg-From-Encoded 参数量: {model.get_num_params(trainable_only=False) / 1e6:.2f}M (总计)")
    
    # 测试前向传播
    x = torch.randn(2, 4, 480, 640)
    cls_logits, seg_mask = model(x)
    
    print(f"输入尺寸: {x.shape}")
    print(f"分类输出尺寸: {cls_logits.shape}")  # (2, 1)
    print(f"分割输出尺寸: {seg_mask.shape}")    # (2, 1, 480, 640)
