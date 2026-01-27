"""
VIPFormer RGBD Seg-Guided 模型架构 (双流版本)

将 RGB(3)+Mask(1) 与 Depth(1) 分开编码，在特征级融合后送入 Transformer；
分割头从 Transformer patch tokens 解码，并用于引导分类头融合。
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transunet import (
    TransUNetConfig,
    PatchEmbedding,
    PositionalEncoding,
    TransformerEncoder,
)
from .transunet_twostream import (
    RGBEncoder,
    DepthEncoder,
    FeatureFusion,
)
from .vipformer_seg_guided import (
    SegmentationDecoder,
    SegGuidedClassificationHead,
)


@dataclass
class VIPFormerRGBDSegGuidedConfig(TransUNetConfig):
    """VIPFormer RGBD Seg-Guided 配置"""

    # 分割查询配置
    num_seg_queries: int = 16

    # 分割解码器配置
    seg_decoder_channels: Tuple[int, ...] = (256, 128, 64, 32)

    # 融合配置（分类头内部融合）
    fusion_type: str = 'cross_attention'  # 'cross_attention', 'concat', 'gated'
    use_uncertainty_weighting: bool = True
    fusion_dropout: float = 0.1

    # 双流配置
    rgb_channels: int = 4   # RGB + Target Mask
    depth_channels: int = 1
    depth_branch_type: str = 'resnet50'
    depth_fusion_type: str = 'concat'
    depth_fusion_dim: int = 1024


class VIPFormerRGBDSegGuided(nn.Module):
    """VIPFormer RGBD 双流 Seg-Guided 模型"""

    def __init__(self, config: Optional[VIPFormerRGBDSegGuidedConfig] = None):
        super().__init__()
        if config is None:
            config = VIPFormerRGBDSegGuidedConfig()
        self.config = config

        # 双流编码器
        self.rgb_encoder = RGBEncoder(config)
        self.depth_encoder = DepthEncoder(config)

        # 特征融合
        rgb_out_channels = 1024
        depth_out_channels = self.depth_encoder.out_channels
        self.fusion = FeatureFusion(
            rgb_channels=rgb_out_channels,
            depth_channels=depth_out_channels,
            out_channels=config.depth_fusion_dim,
            fusion_type=config.depth_fusion_type,
        )

        # Patch Embedding
        self.patch_embed = PatchEmbedding(in_channels=config.depth_fusion_dim, hidden_dim=config.hidden_dim)

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
        self.seg_decoder = SegmentationDecoder(config)  # type: ignore[arg-type]

        # Seg-Guided Classification Head
        self.cls_head = SegGuidedClassificationHead(config)  # type: ignore[arg-type]

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

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: 输入张量 (B, 5, H, W) - RGB(3) + Depth(1) + TargetMask(1)

        Returns:
            cls_logits: 分类 logits (B, num_classes)
            seg_mask: 分割 logits (B, seg_classes, H, W)
        """
        B, C, H, W = x.shape
        config = self.config

        # 分离 RGB+Mask 和 Depth
        rgb_mask = torch.cat([x[:, :3], x[:, 4:5]], dim=1)
        depth = x[:, 3:4]

        # 双流编码
        rgb_feat, _ = self.rgb_encoder(rgb_mask)
        depth_feat = self.depth_encoder(depth)

        # 特征融合
        fused_feat = self.fusion(rgb_feat, depth_feat)

        # Patch Embedding
        patch_embeds, feat_h, feat_w = self.patch_embed(fused_feat)

        # 拼接 tokens
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

        # Segmentation
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


# ============= 工厂函数 (RGBD 双流) =============

def create_vipformer_rgbd_seg_guided(
    pretrained: bool = True,
    img_height: int = 480,
    img_width: int = 640,
    num_transformer_layers: int = 12,
    hidden_dim: int = 768,
    num_seg_queries: int = 16,
    fusion_type: str = 'cross_attention',
    depth_fusion_type: str = 'concat',
    **kwargs,
) -> VIPFormerRGBDSegGuided:
    """创建 VIPFormer RGBD 双流 Seg-Guided 模型"""
    config = VIPFormerRGBDSegGuidedConfig(
        pretrained=pretrained,
        img_height=img_height,
        img_width=img_width,
        num_layers=num_transformer_layers,
        hidden_dim=hidden_dim,
        num_seg_queries=num_seg_queries,
        fusion_type=fusion_type,
        depth_fusion_type=depth_fusion_type,
        **kwargs,
    )
    return VIPFormerRGBDSegGuided(config)


def vipformer_rgbd_seg_guided_base(pretrained: bool = True, **kwargs) -> VIPFormerRGBDSegGuided:
    """VIPFormer-RGBD-SegGuided-Base: 12 层 Transformer, 768 hidden dim"""
    return create_vipformer_rgbd_seg_guided(
        pretrained=pretrained,
        num_transformer_layers=12,
        hidden_dim=768,
        num_heads=12,
        mlp_dim=3072,
        num_seg_queries=16,
        **kwargs,
    )


def vipformer_rgbd_seg_guided_tiny(pretrained: bool = True, **kwargs) -> VIPFormerRGBDSegGuided:
    """VIPFormer-RGBD-SegGuided-Tiny: 4 层 Transformer, 384 hidden dim"""
    return create_vipformer_rgbd_seg_guided(
        pretrained=pretrained,
        num_transformer_layers=4,
        hidden_dim=384,
        num_heads=6,
        mlp_dim=1536,
        num_seg_queries=8,
        **kwargs,
    )

