"""
VIPFormer RGBD Seg-Aware Attention 模型架构 (双流版本)

核心创新: 两阶段 Transformer，分割预测生成注意力偏置引导分类
双流架构: RGB分支 + Depth分支，特征级融合后送入 Transformer

架构流程:
1. RGB Encoder + Depth Encoder → 特征融合
2. Stage 1 Transformer (浅层): 生成初步分割预测
3. 分割预测 → Attention Bias Map
4. Stage 2 Transformer (深层): CLS token 的注意力被 Bias Map 调制
5. 最终分类 + 最终分割
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
)
from .transunet_twostream import (
    RGBEncoder,
    DepthEncoder,
    FeatureFusion,
)
from .vipformer_seg_aware import (
    VIPFormerSegAwareConfig,
    MidSegPredictor,
    SegmentationDecoder,
    Stage1TransformerLayer,
    Stage2TransformerLayer,
)


@dataclass
class VIPFormerRGBDSegAwareConfig(VIPFormerSegAwareConfig):
    """VIPFormer RGBD Seg-Aware 配置"""
    # 双流配置
    rgb_channels: int = 4   # RGB + Target Mask
    depth_channels: int = 1
    depth_branch_type: str = 'resnet50'
    depth_fusion_type: str = 'concat'
    depth_fusion_dim: int = 1024


class VIPFormerRGBDSegAware(nn.Module):
    """VIPFormer RGBD 双流 Seg-Aware 模型"""

    def __init__(self, config: Optional[VIPFormerRGBDSegAwareConfig] = None):
        super().__init__()
        if config is None:
            config = VIPFormerRGBDSegAwareConfig()
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
        self.patch_embed = PatchEmbedding(
            in_channels=config.depth_fusion_dim,
            hidden_dim=config.hidden_dim
        )

        # 计算特征图尺寸
        self.feat_h = config.img_height // 16
        self.feat_w = config.img_width // 16
        num_patches = self.feat_h * self.feat_w

        # 位置编码
        self.pos_encoding = PositionalEncoding(
            hidden_dim=config.hidden_dim,
            max_len=num_patches + 1,
            dropout=config.dropout,
        )

        # CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Stage 1 Transformer (标准注意力)
        self.stage1_layers = nn.ModuleList([
            Stage1TransformerLayer(config) for _ in range(config.stage1_layers)
        ])

        # 中间分割预测器 (生成 Attention Bias)
        self.mid_seg_predictor = MidSegPredictor(
            config.hidden_dim, self.feat_h, self.feat_w
        )

        # Stage 2 Transformer (分割感知注意力)
        self.stage2_layers = nn.ModuleList([
            Stage2TransformerLayer(config) for _ in range(config.stage2_layers)
        ])

        # 最终 LayerNorm
        self.norm = nn.LayerNorm(config.hidden_dim)

        # 分割解码器
        self.seg_decoder = SegmentationDecoder(config)

        # 分类头
        self.cls_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.cls_hidden_dim),
            nn.LayerNorm(config.cls_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.cls_hidden_dim, config.num_classes),
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入张量 (B, 5, H, W) - RGB(3) + Depth(1) + TargetMask(1)
        Returns:
            cls_logits: 分类 logits (B, num_classes)
            seg_mask: 分割 logits (B, seg_classes, H, W)
        """
        B, C, H, W = x.shape

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

        # 添加 CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, patch_embeds], dim=1)

        # 位置编码
        tokens = self.pos_encoding(tokens)

        # ========== Stage 1: 标准 Transformer ==========
        for layer in self.stage1_layers:
            tokens = layer(tokens)

        # 中间分割预测 (用于生成 Attention Bias)
        patch_tokens_mid = tokens[:, 1:, :]
        mid_seg_logits = self.mid_seg_predictor(patch_tokens_mid)

        # 生成 Seg Attention Bias
        seg_prob = torch.sigmoid(mid_seg_logits)
        seg_bias = seg_prob.flatten(2).squeeze(1)

        # ========== Stage 2: 分割感知 Transformer ==========
        for layer in self.stage2_layers:
            tokens = layer(tokens, seg_bias)

        # 最终 LayerNorm
        tokens = self.norm(tokens)

        # 分离 CLS token 和 patch tokens
        cls_token_out = tokens[:, 0, :]
        patch_tokens_out = tokens[:, 1:, :]

        # 分类
        cls_logits = self.cls_head(cls_token_out)

        # 分割
        seg_mask = self.seg_decoder(patch_tokens_out, feat_h, feat_w)
        if seg_mask.shape[2:] != (H, W):
            seg_mask = F.interpolate(seg_mask, size=(H, W), mode='bilinear', align_corners=False)

        return cls_logits, seg_mask

    def get_num_params(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# ============= 工厂函数 (RGBD 双流) =============

def create_vipformer_rgbd_seg_aware(
    pretrained: bool = True,
    img_height: int = 480,
    img_width: int = 640,
    stage1_layers: int = 2,
    stage2_layers: int = 2,
    hidden_dim: int = 768,
    depth_fusion_type: str = 'concat',
    **kwargs,
) -> VIPFormerRGBDSegAware:
    """创建 VIPFormer RGBD 双流 Seg-Aware 模型"""
    config = VIPFormerRGBDSegAwareConfig(
        pretrained=pretrained,
        img_height=img_height,
        img_width=img_width,
        stage1_layers=stage1_layers,
        stage2_layers=stage2_layers,
        hidden_dim=hidden_dim,
        depth_fusion_type=depth_fusion_type,
        **kwargs,
    )
    return VIPFormerRGBDSegAware(config)


def vipformer_rgbd_seg_aware_base(pretrained: bool = True, **kwargs) -> VIPFormerRGBDSegAware:
    """VIPFormer-RGBD-SegAware-Base: 6+6 层 Transformer, 768 hidden dim"""
    return create_vipformer_rgbd_seg_aware(
        pretrained=pretrained,
        stage1_layers=6,
        stage2_layers=6,
        hidden_dim=768,
        num_heads=12,
        mlp_dim=3072,
        **kwargs,
    )


def vipformer_rgbd_seg_aware_small(pretrained: bool = True, **kwargs) -> VIPFormerRGBDSegAware:
    """VIPFormer-RGBD-SegAware-Small: 3+3 层 Transformer, 512 hidden dim"""
    return create_vipformer_rgbd_seg_aware(
        pretrained=pretrained,
        stage1_layers=3,
        stage2_layers=3,
        hidden_dim=512,
        num_heads=8,
        mlp_dim=2048,
        **kwargs,
    )


def vipformer_rgbd_seg_aware_tiny(pretrained: bool = True, **kwargs) -> VIPFormerRGBDSegAware:
    """VIPFormer-RGBD-SegAware-Tiny: 2+2 层 Transformer, 384 hidden dim"""
    return create_vipformer_rgbd_seg_aware(
        pretrained=pretrained,
        stage1_layers=2,
        stage2_layers=2,
        hidden_dim=384,
        num_heads=6,
        mlp_dim=1536,
        **kwargs,
    )


if __name__ == "__main__":
    # 测试模型
    model = vipformer_rgbd_seg_aware_tiny()
    print(f"VIPFormer-RGBD-SegAware-Tiny 参数量: {model.get_num_params() / 1e6:.2f}M")

    x = torch.randn(2, 5, 480, 640)
    cls_logits, seg_mask = model(x)
    print(f"输入: {x.shape}")
    print(f"分类输出: {cls_logits.shape}")
    print(f"分割输出: {seg_mask.shape}")
