"""
VIPFormer Seg-Guided 模型架构 (RGB 单流版本)
分割引导分类 - 通过 Transformer 内部交互实现分割特征对分类的引导

架构创新:
1. SEG Query Tokens: 可学习的分割查询，在 Transformer 中学习分割语义
2. Cross-Attention Fusion: CLS token 通过交叉注意力聚合分割信息
3. Uncertainty-Aware Weighting: 根据分割置信度动态调整融合权重
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
)


@dataclass
class VIPFormerSegGuidedConfig(TransUNetConfig):
    """VIPFormer Seg-Guided 配置"""
    # 分割查询配置
    num_seg_queries: int = 16

    # 分割解码器配置
    seg_decoder_channels: Tuple[int, ...] = (256, 128, 64, 32)

    # 融合配置
    fusion_type: str = 'cross_attention'  # 'cross_attention', 'concat', 'gated'
    use_uncertainty_weighting: bool = True
    fusion_dropout: float = 0.1


# ============= 基础模块 =============

class UpBlockNoSkip(nn.Module):
    """上采样块 - 无 skip connection"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.conv(x)


class SegmentationDecoder(nn.Module):
    """分割解码器 - 从 Transformer 特征直接解码"""

    def __init__(self, config: VIPFormerSegGuidedConfig):
        super().__init__()
        hidden_dim = config.hidden_dim
        decoder_channels = config.seg_decoder_channels

        self.reshape_proj = nn.Conv2d(hidden_dim, decoder_channels[0], 1)

        self.up_blocks = nn.ModuleList()
        in_ch = decoder_channels[0]
        for out_ch in decoder_channels[1:]:
            self.up_blocks.append(UpBlockNoSkip(in_ch, out_ch))
            in_ch = out_ch
        self.up_blocks.append(UpBlockNoSkip(in_ch, in_ch))

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, config.seg_classes, 1),
        )

    def forward(self, x: torch.Tensor, feat_h: int, feat_w: int):
        B = x.shape[0]
        x = x.transpose(1, 2).reshape(B, -1, feat_h, feat_w)
        x = self.reshape_proj(x)
        for up_block in self.up_blocks:
            x = up_block(x)
        return self.final_conv(x)


# ============= 融合模块 =============

class CrossAttentionFusion(nn.Module):
    """交叉注意力融合: CLS token 聚合 SEG queries 和 patch tokens"""

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, cls_token, seg_queries, patch_tokens):
        B = cls_token.shape[0]
        kv_tokens = torch.cat([seg_queries, patch_tokens], dim=1)

        q = self.q_proj(cls_token)
        k = self.k_proj(kv_tokens)
        v = self.v_proj(kv_tokens)

        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, 1, self.hidden_dim)
        out = self.out_proj(out)

        return self.norm(cls_token + out).squeeze(1)


class UncertaintyWeighting(nn.Module):
    """根据分割置信度动态调整融合权重"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, seg_logits):
        probs = torch.sigmoid(seg_logits)
        entropy = -probs * torch.log(probs + 1e-8) - (1 - probs) * torch.log(1 - probs + 1e-8)
        mean_entropy = entropy.mean(dim=(1, 2, 3)).unsqueeze(-1)
        return self.net(mean_entropy)


class GatedFusion(nn.Module):
    """门控融合模块"""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cls_feat, seg_feat):
        concat = torch.cat([cls_feat, seg_feat], dim=-1)
        weights = self.gate(concat)
        fused = weights[:, 0:1] * cls_feat + weights[:, 1:2] * seg_feat
        return self.norm(self.dropout(fused))


# ============= 分类头 =============

class SegGuidedClassificationHead(nn.Module):
    """分割引导分类头"""

    def __init__(self, config: VIPFormerSegGuidedConfig):
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim

        if config.fusion_type == 'cross_attention':
            self.fusion = CrossAttentionFusion(
                hidden_dim, config.num_heads, config.fusion_dropout
            )
        elif config.fusion_type == 'gated':
            self.seg_pool = nn.Linear(hidden_dim, hidden_dim)
            self.fusion = GatedFusion(hidden_dim, config.fusion_dropout)
        else:
            self.fusion = None

        if config.use_uncertainty_weighting:
            self.uncertainty = UncertaintyWeighting()
        else:
            self.uncertainty = None

        input_dim = hidden_dim * 2 if config.fusion_type == 'concat' else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(hidden_dim // 2, config.num_classes),
        )

    def forward(self, cls_token, seg_queries, patch_tokens, seg_logits):
        config = self.config

        if config.fusion_type == 'cross_attention':
            fused = self.fusion(cls_token.unsqueeze(1), seg_queries, patch_tokens)
        elif config.fusion_type == 'gated':
            seg_pooled = F.relu(self.seg_pool(seg_queries.mean(dim=1)))
            fused = self.fusion(cls_token, seg_pooled)
        else:
            seg_pooled = seg_queries.mean(dim=1)
            fused = torch.cat([cls_token, seg_pooled], dim=-1)

        if self.uncertainty is not None and config.fusion_type != 'concat':
            # concat 模式下 fused 维度与 cls_token 不同，跳过 uncertainty weighting
            confidence = self.uncertainty(seg_logits)
            fused = confidence * fused + (1 - confidence) * cls_token

        return self.classifier(fused)


# ============= 主模型 =============

class VIPFormerSegGuided(nn.Module):
    """VIPFormer Seg-Guided: RGB 单流分割引导分类模型"""

    def __init__(self, config: Optional[VIPFormerSegGuidedConfig] = None):
        super().__init__()
        if config is None:
            config = VIPFormerSegGuidedConfig()
        self.config = config

        # CNN Encoder
        self.cnn_encoder = ResNetEncoder(config)

        # Patch Embedding
        self.patch_embed = PatchEmbedding(in_channels=1024, hidden_dim=config.hidden_dim)

        # 位置编码
        max_patches = (config.img_height // 16) * (config.img_width // 16)
        self.pos_encoding = PositionalEncoding(
            hidden_dim=config.hidden_dim,
            max_len=max_patches + 1 + config.num_seg_queries,
            dropout=config.dropout,
        )

        # CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # SEG Query Tokens
        self.seg_queries = nn.Parameter(torch.zeros(1, config.num_seg_queries, config.hidden_dim))
        nn.init.trunc_normal_(self.seg_queries, std=0.02)

        # Transformer Encoder
        self.transformer = TransformerEncoder(config)

        # Segmentation Decoder
        self.seg_decoder = SegmentationDecoder(config)

        # Seg-Guided Classification Head
        self.cls_head = SegGuidedClassificationHead(config)

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入张量 (B, 4, H, W) - RGB + Target Mask

        Returns:
            cls_logits: 分类 logits (B, num_classes)
            seg_mask: 分割 logits (B, seg_classes, H, W)
        """
        B, C, H, W = x.shape
        config = self.config

        # CNN Encoder
        cnn_features, _ = self.cnn_encoder(x)

        # Patch Embedding
        patch_embeds, feat_h, feat_w = self.patch_embed(cnn_features)

        # 拼接 CLS Token + SEG Queries + Patch Tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        seg_queries = self.seg_queries.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, seg_queries, patch_embeds], dim=1)

        # 位置编码
        tokens = self.pos_encoding(tokens)

        # Transformer Encoder
        encoded = self.transformer(tokens)

        # 分离各部分
        cls_token_out = encoded[:, 0, :]
        seg_queries_out = encoded[:, 1:1 + config.num_seg_queries, :]
        patch_tokens_out = encoded[:, 1 + config.num_seg_queries:, :]

        # Segmentation (logits)
        seg_mask = self.seg_decoder(patch_tokens_out, feat_h, feat_w)
        if seg_mask.shape[2:] != (H, W):
            seg_mask = F.interpolate(seg_mask, size=(H, W), mode='bilinear', align_corners=False)

        # Classification (分割引导)
        cls_logits = self.cls_head(cls_token_out, seg_queries_out, patch_tokens_out, seg_mask)

        return cls_logits, seg_mask

    def get_num_params(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# ============= 工厂函数 (RGB 单流) =============

def create_vipformer_seg_guided(
    pretrained: bool = True,
    img_height: int = 480,
    img_width: int = 640,
    num_transformer_layers: int = 12,
    hidden_dim: int = 768,
    num_seg_queries: int = 16,
    fusion_type: str = 'cross_attention',
    **kwargs,
) -> VIPFormerSegGuided:
    """创建 VIPFormer RGB 单流 Seg-Guided 模型"""
    config = VIPFormerSegGuidedConfig(
        pretrained=pretrained,
        img_height=img_height,
        img_width=img_width,
        num_layers=num_transformer_layers,
        hidden_dim=hidden_dim,
        num_seg_queries=num_seg_queries,
        fusion_type=fusion_type,
        **kwargs,
    )
    return VIPFormerSegGuided(config)


def vipformer_seg_guided_base(pretrained: bool = True, **kwargs) -> VIPFormerSegGuided:
    """VIPFormer-SegGuided-Base: 12 层 Transformer, 768 hidden dim"""
    return create_vipformer_seg_guided(
        pretrained=pretrained,
        num_transformer_layers=12,
        hidden_dim=768,
        num_heads=12,
        mlp_dim=3072,
        num_seg_queries=16,
        **kwargs,
    )


def vipformer_seg_guided_small(pretrained: bool = True, **kwargs) -> VIPFormerSegGuided:
    """VIPFormer-SegGuided-Small: 6 层 Transformer, 512 hidden dim"""
    return create_vipformer_seg_guided(
        pretrained=pretrained,
        num_transformer_layers=6,
        hidden_dim=512,
        num_heads=8,
        mlp_dim=2048,
        num_seg_queries=12,
        **kwargs,
    )


def vipformer_seg_guided_tiny(pretrained: bool = True, **kwargs) -> VIPFormerSegGuided:
    """VIPFormer-SegGuided-Tiny: 4 层 Transformer, 384 hidden dim"""
    return create_vipformer_seg_guided(
        pretrained=pretrained,
        num_transformer_layers=4,
        hidden_dim=384,
        num_heads=6,
        mlp_dim=1536,
        num_seg_queries=8,
        **kwargs,
    )