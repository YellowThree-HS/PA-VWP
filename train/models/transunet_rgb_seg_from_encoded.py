"""
TransUNet RGB-Only Seg-From-Encoded 模型架构
仅使用 RGB 输入，无需深度信息，通过分割任务增强分类性能

目标：
1. 纯 RGB 输入（3通道）
2. 通过 SegFromEncoded 架构，让分割任务作为辅助任务增强分类特征
3. 性能应超越 cls_only 模型

架构设计要点：
- 更强的特征提取：多尺度特征融合
- 注意力机制增强关键区域
- 分割和分类任务的特征解耦与融合
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

from .transunet import (
    TransUNetConfig,
    PatchEmbedding,
    PositionalEncoding,
    TransformerEncoder,
    ClassificationHead,
)
from .transunet_seg_from_encoded import (
    TransUNetSegFromEncodedConfig,
    SegmentationHeadFromEncoded,
)


@dataclass
class TransUNetRGBSegFromEncodedConfig(TransUNetSegFromEncodedConfig):
    """RGB-Only Seg-From-Encoded 配置"""
    # 多尺度特征融合配置
    use_multiscale_fusion: bool = True
    multiscale_fusion_dim: int = 512
    
    # 注意力增强配置
    use_spatial_attention: bool = True
    use_channel_attention: bool = True
    
    # 任务解耦配置
    use_task_specific_tokens: bool = True  # 使用分类和分割专用token
    task_token_dim: int = 64
    
    # 特征增强配置
    use_feature_enhancement: bool = True
    enhancement_layers: int = 2


class SpatialAttentionModule(nn.Module):
    """空间注意力模块 - 关注目标区域"""
    
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        self.in_channels = in_channels
        
        # 空间注意力：使用卷积生成注意力图
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            attention: (B, 1, H, W)
        """
        attention = self.spatial_conv(x)
        return attention


class ChannelAttentionModule(nn.Module):
    """通道注意力模块 - 强调重要特征通道"""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            attention: (B, C, 1, 1)
        """
        b, c, _, _ = x.size()
        
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        attention = (avg_out + max_out).view(b, c, 1, 1)
        return attention


class MultiscaleFeatureFusion(nn.Module):
    """多尺度特征融合模块"""
    
    def __init__(self, channels: List[int], out_channels: int):
        super().__init__()
        self.channels = channels
        
        # 将不同尺度特征投影到相同维度
        self.projections = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in channels
        ])
        
        # 融合后的卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * len(channels), out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: 不同尺度特征列表 [(B, C1, H1, W1), (B, C2, H2, W2), ...]
        Returns:
            fused: 融合后的特征 (B, out_channels, H_min, W_min)
        """
        # 投影到相同维度
        projected = []
        for feat, proj in zip(features, self.projections):
            # 上采样到最小尺度的尺寸（最后一个特征）
            target_size = features[-1].shape[2:]
            proj_feat = proj(feat)
            if proj_feat.shape[2:] != target_size:
                proj_feat = F.interpolate(proj_feat, size=target_size, mode='bilinear', align_corners=False)
            projected.append(proj_feat)
        
        # 拼接并融合
        concat = torch.cat(projected, dim=1)
        fused = self.fusion_conv(concat)
        
        return fused


class FeatureEnhancement(nn.Module):
    """特征增强模块 - 使用轻量级卷积增强局部特征"""
    
    def __init__(self, in_channels: int, num_layers: int = 2):
        super().__init__()
        
        layers = []
        current_channels = in_channels
        
        for i in range(num_layers):
            layers.extend([
                nn.Conv2d(current_channels, current_channels, 3, padding=1),
                nn.BatchNorm2d(current_channels),
                nn.ReLU(inplace=True)
            ])
        
        self.enhancement = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enhancement(x) + x  # 残差连接


class EnhancedRGBEncoder(nn.Module):
    """
    增强的 RGB 编码器
    - 使用 ResNet-50 作为骨干
    - 添加多尺度特征融合
    - 添加注意力机制
    """
    
    def __init__(self, config: TransUNetRGBSegFromEncodedConfig):
        super().__init__()
        
        # 加载预训练 ResNet-50
        if config.pretrained:
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            resnet = resnet50(weights=None)
        
        # 修改第一层卷积：3 channels（纯RGB）
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # ResNet 层
        self.layer1 = resnet.layer1  # /4,  256 channels
        self.layer2 = resnet.layer2  # /8,  512 channels
        self.layer3 = resnet.layer3  # /16, 1024 channels
        
        # 删除 layer4 节省内存
        del resnet.layer4
        
        # 多尺度特征融合
        self.use_multiscale = config.use_multiscale_fusion
        if self.use_multiscale:
            self.multiscale_fusion = MultiscaleFeatureFusion(
                channels=[256, 512, 1024],
                out_channels=config.multiscale_fusion_dim
            )
            final_channels = config.multiscale_fusion_dim
        else:
            final_channels = 1024
        
        # 注意力模块
        self.use_spatial_attn = config.use_spatial_attention
        self.use_channel_attn = config.use_channel_attention
        
        if self.use_spatial_attn:
            self.spatial_attention = SpatialAttentionModule(final_channels)
        
        if self.use_channel_attn:
            self.channel_attention = ChannelAttentionModule(final_channels)
        
        # 特征增强
        self.use_enhancement = config.use_feature_enhancement
        if self.use_enhancement:
            self.enhancement = FeatureEnhancement(
                final_channels, 
                config.enhancement_layers
            )
        
        self.out_channels = final_channels
        
        # 是否冻结 BatchNorm
        if config.freeze_bn:
            self._freeze_bn()
    
    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: RGB (B, 3, H, W)
        Returns:
            features: 最终特征图 (B, out_channels, H/16, W/16)
            skip_features: 用于分割头的跳跃连接特征
        """
        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        conv1_feat = x
        
        x = self.maxpool(x)
        
        # ResNet 层
        x1 = self.layer1(x)    # (B, 256, H/4, W/4)
        x2 = self.layer2(x1)   # (B, 512, H/8, W/8)
        x3 = self.layer3(x2)   # (B, 1024, H/16, W/16)
        
        skip_features = [x3, x2, x1, conv1_feat]
        
        # 多尺度特征融合
        if self.use_multiscale:
            features = self.multiscale_fusion([x1, x2, x3])
        else:
            features = x3
        
        # 注意力增强
        if self.use_spatial_attn:
            spatial_attn = self.spatial_attention(features)
            features = features * spatial_attn
        
        if self.use_channel_attn:
            channel_attn = self.channel_attention(features)
            features = features * channel_attn
        
        # 特征增强
        if self.use_enhancement:
            features = self.enhancement(features)
        
        return features, skip_features


class TransUNetRGBSegFromEncoded(nn.Module):
    """
    TransUNet RGB-Only Seg-From-Encoded 模型
    
    设计目标：
    1. 仅使用 RGB 输入（3通道）
    2. 通过分割任务作为辅助，增强分类特征学习
    3. 性能超越 cls_only 模型
    
    关键创新：
    - 增强的 RGB 编码器（多尺度融合 + 注意力）
    - 任务专用 token（分类和分割各自学习）
    - 更强的特征表示能力
    """
    
    def __init__(self, config: Optional[TransUNetRGBSegFromEncodedConfig] = None):
        super().__init__()
        
        if config is None:
            config = TransUNetRGBSegFromEncodedConfig()
        self.config = config
        
        # 增强的 RGB 编码器
        self.rgb_encoder = EnhancedRGBEncoder(config)
        
        # Patch Embedding
        encoder_out_channels = self.rgb_encoder.out_channels
        self.patch_embed = PatchEmbedding(
            in_channels=encoder_out_channels,
            hidden_dim=config.hidden_dim
        )
        
        # 位置编码
        max_patches = (config.img_height // 16) * (config.img_width // 16)
        self.pos_encoding = PositionalEncoding(
            hidden_dim=config.hidden_dim,
            max_len=max_patches + 3,  # +1 for CLS, +2 for task tokens
            dropout=config.dropout
        )
        
        # CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # 任务专用 Token
        if config.use_task_specific_tokens:
            self.cls_task_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
            self.seg_task_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
            nn.init.trunc_normal_(self.cls_task_token, std=0.02)
            nn.init.trunc_normal_(self.seg_task_token, std=0.02)
            self.num_task_tokens = 2
        else:
            self.num_task_tokens = 0
        
        # Transformer Encoder
        self.transformer = TransformerEncoder(config)
        
        # Classification Head
        if config.use_task_specific_tokens:
            # 使用分类任务 token
            cls_input_dim = config.hidden_dim
        else:
            cls_input_dim = config.hidden_dim
            
        self.cls_head = ClassificationHead(
            hidden_dim=config.hidden_dim,
            cls_hidden_dim=config.cls_hidden_dim,
            num_classes=config.num_classes,
            dropout=config.dropout
        )
        
        # Segmentation Head
        self.seg_head = SegmentationHeadFromEncoded(config)
        
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
            x: 输入张量 (B, 3, H, W) - 纯 RGB
            return_attention: 是否返回注意力权重
            
        Returns:
            cls_logits: 分类 logits (B, 1)
            seg_mask: 分割掩码 (B, 1, H, W)
        """
        B, C, H, W = x.shape
        
        # RGB 编码
        cnn_features, skip_features = self.rgb_encoder(x)
        
        # Patch Embedding
        patch_embeds, feat_h, feat_w = self.patch_embed(cnn_features)
        
        # 构建 tokens
        tokens_list = [self.cls_token.expand(B, -1, -1), patch_embeds]
        
        if self.config.use_task_specific_tokens:
            tokens_list.extend([
                self.cls_task_token.expand(B, -1, -1),
                self.seg_task_token.expand(B, -1, -1)
            ])
        
        tokens = torch.cat(tokens_list, dim=1)
        
        # 位置编码
        tokens = self.pos_encoding(tokens)
        
        # Transformer Encoder
        encoded = self.transformer(tokens)
        
        # 分离 tokens
        if self.config.use_task_specific_tokens:
            # tokens: [CLS, patch_tokens, CLS_task, SEG_task]
            cls_tokens_for_head = torch.cat([
                encoded[:, 0:1, :],      # CLS
                encoded[:, -2:-1, :]     # CLS_task
            ], dim=1)
            patch_tokens = encoded[:, 1:-2, :]
        else:
            cls_tokens_for_head = encoded
            patch_tokens = encoded[:, 1:, :]
        
        # Classification
        cls_logits = self.cls_head(cls_tokens_for_head)
        
        # Segmentation
        seg_mask, _ = self.seg_head(patch_tokens, feat_h, feat_w)
        
        # 调整分割掩码尺寸
        if seg_mask.shape[2:] != (H, W):
            seg_mask = F.interpolate(seg_mask, size=(H, W), mode='bilinear', align_corners=False)
        
        return cls_logits, seg_mask
    
    def get_num_params(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# ============= 工厂函数 =============

def create_rgb_seg_from_encoded(
    pretrained: bool = True,
    img_height: int = 480,
    img_width: int = 640,
    num_transformer_layers: int = 4,
    hidden_dim: int = 384,
    use_multiscale_fusion: bool = True,
    use_spatial_attention: bool = True,
    use_channel_attention: bool = True,
    use_task_tokens: bool = True,
    **kwargs
) -> TransUNetRGBSegFromEncoded:
    """创建 RGB-Only Seg-From-Encoded 模型"""
    config = TransUNetRGBSegFromEncodedConfig(
        pretrained=pretrained,
        img_height=img_height,
        img_width=img_width,
        num_layers=num_transformer_layers,
        hidden_dim=hidden_dim,
        use_multiscale_fusion=use_multiscale_fusion,
        use_spatial_attention=use_spatial_attention,
        use_channel_attention=use_channel_attention,
        use_task_specific_tokens=use_task_tokens,
        **kwargs
    )
    return TransUNetRGBSegFromEncoded(config)


# ============= 模型变体 =============

def rgb_seg_from_encoded_tiny(pretrained: bool = True, **kwargs) -> TransUNetRGBSegFromEncoded:
    """RGB-SegFromEncoded-Tiny: 4 层 Transformer, 384 hidden dim"""
    return create_rgb_seg_from_encoded(
        pretrained=pretrained,
        num_transformer_layers=4,
        hidden_dim=384,
        num_heads=6,
        mlp_dim=1536,
        **kwargs
    )


def rgb_seg_from_encoded_small(pretrained: bool = True, **kwargs) -> TransUNetRGBSegFromEncoded:
    """RGB-SegFromEncoded-Small: 6 层 Transformer, 512 hidden dim"""
    return create_rgb_seg_from_encoded(
        pretrained=pretrained,
        num_transformer_layers=6,
        hidden_dim=512,
        num_heads=8,
        mlp_dim=2048,
        **kwargs
    )


def rgb_seg_from_encoded_base(pretrained: bool = True, **kwargs) -> TransUNetRGBSegFromEncoded:
    """RGB-SegFromEncoded-Base: 12 层 Transformer, 768 hidden dim"""
    return create_rgb_seg_from_encoded(
        pretrained=pretrained,
        num_transformer_layers=12,
        hidden_dim=768,
        num_heads=12,
        mlp_dim=3072,
        **kwargs
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Testing RGB-SegFromEncoded-Tiny")
    print("=" * 60)
    
    model = rgb_seg_from_encoded_tiny()
    print(f"参数量: {model.get_num_params() / 1e6:.2f}M")
    
    x = torch.randn(2, 3, 480, 640)
    cls_logits, seg_mask = model(x)
    print(f"输入: {x.shape}")
    print(f"分类输出: {cls_logits.shape}")
    print(f"分割输出: {seg_mask.shape}")
