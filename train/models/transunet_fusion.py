"""
TransUNet Fusion 模型架构
分割感知双路融合 - 用于物理稳定性预测

架构概述:
- 基于 TransUNet，增加分割-分类特征融合
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

from .transunet import (
    TransUNetConfig,
    ResNetEncoder,
    PatchEmbedding,
    PositionalEncoding,
    TransformerEncoder,
    UpBlock,
)


@dataclass
class TransUNetFusionConfig(TransUNetConfig):
    """TransUNet Fusion 配置"""
    # 融合模块配置
    fusion_hidden_dim: int = 256  # 融合 MLP 隐藏层维度
    use_mask_guided: bool = True  # 是否使用分割掩码引导的注意力
    use_decoder_features: bool = True  # 是否使用分割解码器中间特征
    decoder_feature_channels: Tuple[int, ...] = (128, 64, 32)  # 从哪些 UpBlock 提取特征
    fusion_dropout: float = 0.1


class SegmentationHeadWithFeatures(nn.Module):
    """
    带中间特征输出的分割头
    除了输出分割掩码，还返回解码器各阶段的特征用于融合
    """
    
    def __init__(
        self, 
        config: TransUNetFusionConfig,
    ):
        super().__init__()
        self.config = config
        
        # 从 Transformer 特征恢复到 2D
        self.reshape_proj = nn.Conv2d(config.hidden_dim, config.decoder_channels[0], 1)
        
        # 上采样路径
        decoder_channels = config.decoder_channels
        skip_channels = config.skip_channels
        
        self.up_blocks = nn.ModuleList()
        
        # UpBlock 1: H/16 -> H/8, 与 layer3 (1024 channels) 跳跃连接
        self.up_blocks.append(UpBlock(
            decoder_channels[0], decoder_channels[1], skip_channels[0]
        ))
        
        # UpBlock 2: H/8 -> H/4, 与 layer2 (512 channels) 跳跃连接
        self.up_blocks.append(UpBlock(
            decoder_channels[1], decoder_channels[2], skip_channels[1]
        ))
        
        # UpBlock 3: H/4 -> H/2, 与 layer1 (256 channels) 跳跃连接
        self.up_blocks.append(UpBlock(
            decoder_channels[2], decoder_channels[3], skip_channels[2]
        ))
        
        # UpBlock 4: H/2 -> H, 与 conv1 (64 channels) 跳跃连接
        self.up_blocks.append(UpBlock(
            decoder_channels[3], decoder_channels[3], skip_channels[3]
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
        skip_features: list,
        return_intermediate: bool = True
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Args:
            x: Transformer 输出 (B, seq_len, hidden_dim)
            feat_h, feat_w: 特征图高度和宽度 (H/16, W/16)
            skip_features: [layer3_feat, layer2_feat, layer1_feat, conv1_feat]
            return_intermediate: 是否返回中间特征
            
        Returns:
            seg_mask: 分割掩码 (B, 1, H, W)
            intermediate_features: 各 UpBlock 输出的特征列表 (如果 return_intermediate=True)
        """
        B = x.shape[0]
        
        # Reshape: (B, seq_len, hidden_dim) -> (B, hidden_dim, H, W)
        x = x.transpose(1, 2).reshape(B, -1, feat_h, feat_w)
        x = self.reshape_proj(x)  # (B, decoder_channels[0], H/16, W/16)
        
        intermediate_features = []
        
        # 上采样路径，从深到浅
        for i, up_block in enumerate(self.up_blocks):
            skip = skip_features[i] if i < len(skip_features) else None
            x = up_block(x, skip)
            
            # 保存中间特征 (前3个 UpBlock 的输出)
            if return_intermediate and i < 3:
                intermediate_features.append(x)
            
        # 最终输出
        seg_mask = self.final_conv(x)
        
        if return_intermediate:
            return seg_mask, intermediate_features
        return seg_mask, None


class MaskGuidedAttentionPooling(nn.Module):
    """
    分割掩码引导的注意力池化
    使用分割头预测的 mask 作为空间注意力权重，引导 patch tokens 的聚合
    """
    
    def __init__(self, hidden_dim: int, temperature: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        
    def forward(
        self, 
        patch_tokens: torch.Tensor,  # (B, num_patches, hidden_dim)
        seg_mask: torch.Tensor,       # (B, 1, H, W)
        feat_h: int,
        feat_w: int
    ) -> torch.Tensor:
        """
        Args:
            patch_tokens: Transformer patch 输出 (B, num_patches, hidden_dim)
            seg_mask: 分割掩码 (B, 1, H, W)
            feat_h, feat_w: patch 特征图尺寸
            
        Returns:
            seg_guided_feat: 分割引导的特征 (B, hidden_dim)
        """
        B = patch_tokens.shape[0]
        
        # 将分割掩码下采样到 patch 尺寸
        # seg_mask: (B, 1, H, W) -> (B, 1, feat_h, feat_w)
        attention_map = F.interpolate(
            seg_mask, 
            size=(feat_h, feat_w), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Sigmoid 确保值在 [0, 1] 范围 (如果 seg_mask 是 logits)
        attention_map = torch.sigmoid(attention_map / self.temperature)
        
        # Flatten: (B, 1, feat_h, feat_w) -> (B, num_patches)
        attention_weights = attention_map.flatten(2).squeeze(1)  # (B, num_patches)
        
        # 归一化 (softmax-like, 但保持原始权重的相对大小)
        # 添加小量防止除零
        attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # 加权聚合 patch tokens
        # (B, num_patches, 1) * (B, num_patches, hidden_dim) -> sum -> (B, hidden_dim)
        seg_guided_feat = (attention_weights.unsqueeze(-1) * patch_tokens).sum(dim=1)
        
        return seg_guided_feat


class MultiScaleFeaturePooling(nn.Module):
    """
    多尺度分割解码器特征池化
    从分割解码器的多个阶段提取特征，全局池化后拼接
    """
    
    def __init__(self, feature_channels: Tuple[int, ...] = (128, 64, 32)):
        super().__init__()
        self.feature_channels = feature_channels
        self.total_channels = sum(feature_channels)
        
    def forward(self, intermediate_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            intermediate_features: 分割解码器各阶段的特征
                - feat1: (B, 128, H/8, W/8)
                - feat2: (B, 64, H/4, W/4)  
                - feat3: (B, 32, H/2, W/2)
                
        Returns:
            pooled_features: (B, total_channels)
        """
        pooled = []
        for feat in intermediate_features:
            # 全局平均池化
            pooled_feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)  # (B, C)
            pooled.append(pooled_feat)
            
        # 拼接所有尺度的特征
        return torch.cat(pooled, dim=1)  # (B, 128+64+32=224)


class FusedClassificationHead(nn.Module):
    """
    融合分类头
    整合 CLS Token、分割引导特征、解码器多尺度特征
    """
    
    def __init__(
        self,
        cls_dim: int = 768,           # CLS Token 维度
        seg_guided_dim: int = 768,     # 分割引导特征维度
        decoder_feat_dim: int = 224,   # 解码器多尺度特征维度 (128+64+32)
        hidden_dim: int = 256,
        num_classes: int = 1,
        dropout: float = 0.1,
        use_mask_guided: bool = True,
        use_decoder_features: bool = True,
    ):
        super().__init__()
        
        self.use_mask_guided = use_mask_guided
        self.use_decoder_features = use_decoder_features
        
        # 计算输入维度
        input_dim = cls_dim
        if use_mask_guided:
            input_dim += seg_guided_dim
        if use_decoder_features:
            input_dim += decoder_feat_dim
            
        # 融合 MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
    def forward(
        self,
        cls_feat: torch.Tensor,                    # (B, cls_dim)
        seg_guided_feat: Optional[torch.Tensor],   # (B, seg_guided_dim)
        decoder_feat: Optional[torch.Tensor],      # (B, decoder_feat_dim)
    ) -> torch.Tensor:
        """
        Args:
            cls_feat: CLS Token 全局特征 (B, 768)
            seg_guided_feat: 分割掩码引导的特征 (B, 768)
            decoder_feat: 解码器多尺度特征 (B, 224)
            
        Returns:
            cls_logits: 分类 logits (B, num_classes)
        """
        features = [cls_feat]
        
        if self.use_mask_guided and seg_guided_feat is not None:
            features.append(seg_guided_feat)
            
        if self.use_decoder_features and decoder_feat is not None:
            features.append(decoder_feat)
            
        # 拼接所有特征
        fused = torch.cat(features, dim=1)
        
        return self.fusion_mlp(fused)


class TransUNetFusion(nn.Module):
    """
    TransUNet Fusion: 分割感知双路融合模型
    
    用于物理稳定性预测的双任务模型，增加分割-分类特征融合:
    - Task 1: 稳定性二分类 (stable / unstable) - 融合分割特征
    - Task 2: 受影响区域分割
    
    融合策略:
    1. CLS Token: 全局语义理解
    2. Mask-Guided Attention: 分割掩码引导的空间注意力
    3. Decoder Features: 分割解码器多尺度特征
    """
    
    def __init__(self, config: Optional[TransUNetFusionConfig] = None):
        super().__init__()
        
        if config is None:
            config = TransUNetFusionConfig()
        self.config = config
        
        # CNN Encoder (ResNet-50)
        self.cnn_encoder = ResNetEncoder(config)
        
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
        self.seg_head = SegmentationHeadWithFeatures(config)
        
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
            x: 输入张量 (B, 4, H, W) - RGB + Target Mask
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


def create_transunet_fusion(
    pretrained: bool = True,
    img_height: int = 480,
    img_width: int = 640,
    num_transformer_layers: int = 12,
    hidden_dim: int = 768,
    use_mask_guided: bool = True,
    use_decoder_features: bool = True,
    **kwargs
) -> TransUNetFusion:
    """工厂函数：创建 TransUNet Fusion 模型"""
    config = TransUNetFusionConfig(
        pretrained=pretrained,
        img_height=img_height,
        img_width=img_width,
        num_layers=num_transformer_layers,
        hidden_dim=hidden_dim,
        use_mask_guided=use_mask_guided,
        use_decoder_features=use_decoder_features,
        **kwargs
    )
    return TransUNetFusion(config)


# 模型变体
def transunet_fusion_base(pretrained: bool = True, **kwargs) -> TransUNetFusion:
    """TransUNet-Fusion-Base: 12 层 Transformer, 768 hidden dim, 完整融合"""
    return create_transunet_fusion(
        pretrained=pretrained,
        num_transformer_layers=12,
        hidden_dim=768,
        num_heads=12,
        mlp_dim=3072,
        use_mask_guided=True,
        use_decoder_features=True,
        **kwargs
    )


def transunet_fusion_small(pretrained: bool = True, **kwargs) -> TransUNetFusion:
    """TransUNet-Fusion-Small: 6 层 Transformer, 512 hidden dim"""
    return create_transunet_fusion(
        pretrained=pretrained,
        num_transformer_layers=6,
        hidden_dim=512,
        num_heads=8,
        mlp_dim=2048,
        use_mask_guided=True,
        use_decoder_features=True,
        **kwargs
    )


def transunet_fusion_tiny(pretrained: bool = True, **kwargs) -> TransUNetFusion:
    """TransUNet-Fusion-Tiny: 4 层 Transformer, 384 hidden dim (用于快速实验)"""
    return create_transunet_fusion(
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
    config = TransUNetFusionConfig()
    model = TransUNetFusion(config)
    
    # 打印模型信息
    print(f"TransUNet Fusion 参数量: {model.get_num_params() / 1e6:.2f}M (可训练)")
    print(f"TransUNet Fusion 参数量: {model.get_num_params(trainable_only=False) / 1e6:.2f}M (总计)")
    
    # 测试前向传播
    x = torch.randn(2, 4, 480, 640)
    cls_logits, seg_mask = model(x)
    
    print(f"输入尺寸: {x.shape}")
    print(f"分类输出尺寸: {cls_logits.shape}")  # (2, 1)
    print(f"分割输出尺寸: {seg_mask.shape}")    # (2, 1, 480, 640)
