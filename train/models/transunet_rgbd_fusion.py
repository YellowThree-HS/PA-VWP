"""
TransUNet RGBD Fusion 模型架构
分割感知双路融合 - 用于物理稳定性预测 (带深度图输入)

架构概述:
- 基于 TransUNet RGBD，增加分割-分类特征融合
- 输入: RGB (3) + Depth (1) + Target Mask (1) = 5 通道
- 融合策略:
  1. 分割掩码引导的注意力池化 (Mask-Guided Attention Pooling)
  2. 分割解码器多尺度特征融合 (Multi-Scale Decoder Features)
  3. CLS Token 全局语义
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transunet_rgbd import (
    TransUNetRGBDConfig,
    ResNetEncoderRGBD,
)

from .transunet import (
    PatchEmbedding,
    PositionalEncoding,
    TransformerEncoder,
)

from .transunet_fusion import (
    SegmentationHeadWithFeatures,
    MaskGuidedAttentionPooling,
    MultiScaleFeaturePooling,
    FusedClassificationHead,
)


@dataclass
class TransUNetRGBDFusionConfig(TransUNetRGBDConfig):
    """TransUNet RGBD Fusion 配置"""
    # 融合模块配置
    fusion_hidden_dim: int = 256  # 融合 MLP 隐藏层维度
    use_mask_guided: bool = True  # 是否使用分割掩码引导的注意力
    use_decoder_features: bool = True  # 是否使用分割解码器中间特征
    decoder_feature_channels: Tuple[int, ...] = (128, 64, 32)  # 从哪些 UpBlock 提取特征
    fusion_dropout: float = 0.1


class TransUNetRGBDFusion(nn.Module):
    """
    TransUNet RGBD Fusion: 分割感知双路融合模型 (带深度图)
    
    用于物理稳定性预测的双任务模型，增加分割-分类特征融合:
    - Task 1: 稳定性二分类 (stable / unstable) - 融合分割特征
    - Task 2: 受影响区域分割
    - 输入: RGB (3) + Depth (1) + Target Mask (1) = 5 通道
    
    融合策略:
    1. CLS Token: 全局语义理解
    2. Mask-Guided Attention: 分割掩码引导的空间注意力
    3. Decoder Features: 分割解码器多尺度特征
    """
    
    def __init__(self, config: Optional[TransUNetRGBDFusionConfig] = None):
        super().__init__()
        
        if config is None:
            config = TransUNetRGBDFusionConfig()
        self.config = config
        
        # CNN Encoder (ResNet-50 RGBD)
        self.cnn_encoder = ResNetEncoderRGBD(config)
        
        # Patch Embedding: CNN 特征 -> Transformer 序列
        self.patch_embed = PatchEmbedding(
            in_channels=1024,
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
        
        # Segmentation Head (带中间特征输出)
        # 注意：复用 TransUNetFusionConfig 的格式
        seg_config = TransUNetRGBDFusionConfig(
            hidden_dim=config.hidden_dim,
            decoder_channels=config.decoder_channels,
            skip_channels=config.skip_channels,
            seg_classes=config.seg_classes,
        )
        self.seg_head = SegmentationHeadWithFeatures(seg_config)
        
        # 融合模块
        if config.use_mask_guided:
            self.mask_guided_pooling = MaskGuidedAttentionPooling(
                hidden_dim=config.hidden_dim
            )
        else:
            self.mask_guided_pooling = None
            
        if config.use_decoder_features:
            self.decoder_feature_pooling = MultiScaleFeaturePooling(
                feature_channels=config.decoder_feature_channels
            )
            decoder_feat_dim = sum(config.decoder_feature_channels)
        else:
            self.decoder_feature_pooling = None
            decoder_feat_dim = 0
            
        # 融合分类头
        self.cls_head = FusedClassificationHead(
            cls_dim=config.hidden_dim,
            seg_guided_dim=config.hidden_dim if config.use_mask_guided else 0,
            decoder_feat_dim=decoder_feat_dim,
            hidden_dim=config.fusion_hidden_dim,
            num_classes=config.num_classes,
            dropout=config.fusion_dropout,
            use_mask_guided=config.use_mask_guided,
            use_decoder_features=config.use_decoder_features,
        )
        
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
            x: 输入张量 (B, 5, H, W) - RGB + Depth + Target Mask
            return_attention: 是否返回注意力权重 (用于可视化)
            
        Returns:
            cls_logits: 分类 logits (B, 1)
            seg_mask: 分割掩码 (B, 1, H, W)
        """
        B, C, H, W = x.shape
        
        # CNN Encoder
        cnn_features, skip_features = self.cnn_encoder(x)
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
        cls_token_out = encoded[:, 0, :]    # (B, hidden_dim)
        patch_tokens = encoded[:, 1:, :]     # (B, num_patches, hidden_dim)
        
        # Task 2: Segmentation (先执行，因为分类需要分割特征)
        seg_mask, intermediate_features = self.seg_head(
            patch_tokens, feat_h, feat_w, skip_features, 
            return_intermediate=True
        )
        
        # 调整分割掩码尺寸
        if seg_mask.shape[2:] != (H, W):
            seg_mask = F.interpolate(seg_mask, size=(H, W), mode='bilinear', align_corners=False)
        
        # === 特征融合 ===
        
        # 1. CLS Token 特征
        cls_feat = cls_token_out  # (B, hidden_dim)
        
        # 2. 分割掩码引导的注意力池化
        seg_guided_feat = None
        if self.mask_guided_pooling is not None:
            seg_guided_feat = self.mask_guided_pooling(
                patch_tokens, seg_mask, feat_h, feat_w
            )  # (B, hidden_dim)
        
        # 3. 分割解码器多尺度特征
        decoder_feat = None
        if self.decoder_feature_pooling is not None and intermediate_features:
            decoder_feat = self.decoder_feature_pooling(intermediate_features)
            # (B, 224)
        
        # Task 1: Classification (融合后)
        cls_logits = self.cls_head(cls_feat, seg_guided_feat, decoder_feat)
            
        return cls_logits, seg_mask
    
    def get_num_params(self, trainable_only: bool = True) -> int:
        """获取模型参数数量"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_transunet_rgbd_fusion(
    pretrained: bool = True,
    img_height: int = 480,
    img_width: int = 640,
    num_transformer_layers: int = 12,
    hidden_dim: int = 768,
    use_mask_guided: bool = True,
    use_decoder_features: bool = True,
    **kwargs
) -> TransUNetRGBDFusion:
    """工厂函数：创建 TransUNet RGBD Fusion 模型"""
    config = TransUNetRGBDFusionConfig(
        pretrained=pretrained,
        img_height=img_height,
        img_width=img_width,
        num_layers=num_transformer_layers,
        hidden_dim=hidden_dim,
        use_mask_guided=use_mask_guided,
        use_decoder_features=use_decoder_features,
        **kwargs
    )
    return TransUNetRGBDFusion(config)


# 模型变体
def transunet_rgbd_fusion_base(pretrained: bool = True, **kwargs) -> TransUNetRGBDFusion:
    """TransUNet-RGBD-Fusion-Base: 12 层 Transformer, 768 hidden dim, 完整融合"""
    return create_transunet_rgbd_fusion(
        pretrained=pretrained,
        num_transformer_layers=12,
        hidden_dim=768,
        num_heads=12,
        mlp_dim=3072,
        use_mask_guided=True,
        use_decoder_features=True,
        **kwargs
    )


def transunet_rgbd_fusion_small(pretrained: bool = True, **kwargs) -> TransUNetRGBDFusion:
    """TransUNet-RGBD-Fusion-Small: 6 层 Transformer, 512 hidden dim"""
    return create_transunet_rgbd_fusion(
        pretrained=pretrained,
        num_transformer_layers=6,
        hidden_dim=512,
        num_heads=8,
        mlp_dim=2048,
        use_mask_guided=True,
        use_decoder_features=True,
        **kwargs
    )


def transunet_rgbd_fusion_tiny(pretrained: bool = True, **kwargs) -> TransUNetRGBDFusion:
    """TransUNet-RGBD-Fusion-Tiny: 4 层 Transformer, 384 hidden dim (用于快速实验)"""
    return create_transunet_rgbd_fusion(
        pretrained=pretrained,
        num_transformer_layers=4,
        hidden_dim=384,
        num_heads=6,
        mlp_dim=1536,
        use_mask_guided=True,
        use_decoder_features=True,
        **kwargs
    )


if __name__ == "__main__":
    # 测试模型
    config = TransUNetRGBDFusionConfig()
    model = TransUNetRGBDFusion(config)
    
    # 打印模型信息
    print(f"TransUNet RGBD Fusion 参数量: {model.get_num_params() / 1e6:.2f}M (可训练)")
    print(f"TransUNet RGBD Fusion 参数量: {model.get_num_params(trainable_only=False) / 1e6:.2f}M (总计)")
    
    # 测试前向传播
    x = torch.randn(2, 5, 480, 640)  # 5 通道: RGB + Depth + Mask
    cls_logits, seg_mask = model(x)
    
    print(f"输入尺寸: {x.shape}")
    print(f"分类输出尺寸: {cls_logits.shape}")  # (2, 1)
    print(f"分割输出尺寸: {seg_mask.shape}")    # (2, 1, 480, 640)
