"""
VIPFormer Seg-Aware Attention 模型架构 (RGB 单流版本)

核心创新: 两阶段 Transformer，分割预测生成注意力偏置引导分类

架构流程:
1. CNN Encoder → Patch Tokens
2. Stage 1 Transformer (浅层): 生成初步分割预测
3. 分割预测 → Attention Bias Map (哪些区域是受影响区域)
4. Stage 2 Transformer (深层): CLS token 的注意力被 Bias Map 调制
5. 最终分类 + 最终分割

优势:
- 分割信息直接影响 CLS token 的注意力分布
- 让模型"看向"受影响区域来做分类决策
- 类似人类判断：先看哪里会倒，再判断稳不稳定
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transunet import (
    TransUNetConfig,
    ResNetEncoder,
    PatchEmbedding,
    PositionalEncoding,
)


@dataclass
class VIPFormerSegAwareConfig(TransUNetConfig):
    """VIPFormer Seg-Aware 配置"""
    # 两阶段 Transformer 配置
    stage1_layers: int = 2      # Stage 1 层数 (浅层，用于初步分割)
    stage2_layers: int = 2      # Stage 2 层数 (深层，带 Seg Bias)

    # 分割解码器配置
    seg_decoder_channels: Tuple[int, ...] = (256, 128, 64, 32)

    # Seg Attention Bias 配置
    seg_bias_scale: float = 1.0         # 初始 bias 缩放因子
    seg_bias_learnable: bool = True     # 是否学习 bias 缩放
    seg_bias_mode: str = 'additive'     # 'additive' 或 'multiplicative'


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


class MidSegPredictor(nn.Module):
    """中间分割预测器 - 轻量级，用于生成 Attention Bias"""

    def __init__(self, hidden_dim: int, feat_h: int, feat_w: int):
        super().__init__()
        self.feat_h = feat_h
        self.feat_w = feat_w

        # 轻量级预测头: hidden_dim -> 1
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, num_patches, hidden_dim)
        Returns:
            seg_logits: (B, 1, feat_h, feat_w)
        """
        B = patch_tokens.shape[0]
        # (B, num_patches, 1)
        logits = self.predictor(patch_tokens)
        # Reshape to 2D
        logits = logits.transpose(1, 2).reshape(B, 1, self.feat_h, self.feat_w)
        return logits


class SegmentationDecoder(nn.Module):
    """分割解码器 - 从 Transformer 特征直接解码"""

    def __init__(self, config: VIPFormerSegAwareConfig):
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


# ============= Seg-Aware Transformer 模块 =============

class SegAwareMultiHeadAttention(nn.Module):
    """分割感知多头注意力 - CLS token 的注意力受分割 bias 引导"""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        seg_bias_scale: float = 1.0,
        seg_bias_learnable: bool = True,
        seg_bias_mode: str = 'additive',
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.seg_bias_mode = seg_bias_mode

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # 可学习的 seg bias 缩放因子
        if seg_bias_learnable:
            self.seg_bias_scale = nn.Parameter(torch.tensor(seg_bias_scale))
        else:
            self.register_buffer('seg_bias_scale', torch.tensor(seg_bias_scale))

    def forward(
        self,
        x: torch.Tensor,
        seg_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, hidden_dim) 其中 N = 1 + num_patches (CLS + patches)
            seg_bias: (B, num_patches) 分割概率作为注意力偏置
        Returns:
            out: (B, N, hidden_dim)
        """
        B, N, _ = x.shape
        num_patches = N - 1

        # QKV 投影
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)

        # 应用分割偏置 (只对 CLS token 的 query)
        if seg_bias is not None:
            # seg_bias: (B, num_patches) -> (B, 1, 1, num_patches)
            # 只影响 CLS token (第0行) 对 patch tokens (第1列开始) 的注意力
            bias = seg_bias.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, num_patches)
            bias = bias * self.seg_bias_scale

            if self.seg_bias_mode == 'additive':
                # 加性偏置: 受影响区域获得更高注意力
                attn[:, :, 0:1, 1:] = attn[:, :, 0:1, 1:] + bias
            else:
                # 乘性偏置
                attn[:, :, 0:1, 1:] = attn[:, :, 0:1, 1:] * (1 + bias)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 输出
        out = (attn @ v).transpose(1, 2).reshape(B, N, self.hidden_dim)
        out = self.proj(out)
        return out


class StandardMultiHeadAttention(nn.Module):
    """标准多头注意力 (用于 Stage 1)"""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, self.hidden_dim)
        return self.proj(out)


class MLP(nn.Module):
    """Transformer MLP 块"""

    def __init__(self, hidden_dim: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Stage1TransformerLayer(nn.Module):
    """Stage 1 Transformer 层 - 标准注意力"""

    def __init__(self, config: VIPFormerSegAwareConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.attn = StandardMultiHeadAttention(
            config.hidden_dim, config.num_heads, config.attention_dropout
        )
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = MLP(config.hidden_dim, config.mlp_dim, config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Stage2TransformerLayer(nn.Module):
    """Stage 2 Transformer 层 - 分割感知注意力"""

    def __init__(self, config: VIPFormerSegAwareConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.attn = SegAwareMultiHeadAttention(
            config.hidden_dim,
            config.num_heads,
            config.attention_dropout,
            config.seg_bias_scale,
            config.seg_bias_learnable,
            config.seg_bias_mode,
        )
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = MLP(config.hidden_dim, config.mlp_dim, config.dropout)

    def forward(
        self, x: torch.Tensor, seg_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), seg_bias)
        x = x + self.mlp(self.norm2(x))
        return x


# ============= 主模型 =============

class VIPFormerSegAware(nn.Module):
    """
    VIPFormer Seg-Aware: RGB 单流分割感知注意力模型

    两阶段 Transformer:
    - Stage 1: 标准注意力，生成初步分割预测
    - Stage 2: 分割感知注意力，CLS token 被引导关注受影响区域
    """

    def __init__(self, config: Optional[VIPFormerSegAwareConfig] = None):
        super().__init__()
        if config is None:
            config = VIPFormerSegAwareConfig()
        self.config = config

        # CNN Encoder
        self.cnn_encoder = ResNetEncoder(config)

        # Patch Embedding
        self.patch_embed = PatchEmbedding(in_channels=1024, hidden_dim=config.hidden_dim)

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
            x: 输入张量 (B, 4, H, W) - RGB + Target Mask

        Returns:
            cls_logits: 分类 logits (B, num_classes)
            seg_mask: 分割 logits (B, seg_classes, H, W)
        """
        B, C, H, W = x.shape

        # CNN Encoder
        cnn_features, _ = self.cnn_encoder(x)

        # Patch Embedding
        patch_embeds, feat_h, feat_w = self.patch_embed(cnn_features)

        # 添加 CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, patch_embeds], dim=1)

        # 位置编码
        tokens = self.pos_encoding(tokens)

        # ========== Stage 1: 标准 Transformer ==========
        for layer in self.stage1_layers:
            tokens = layer(tokens)

        # 中间分割预测 (用于生成 Attention Bias)
        patch_tokens_mid = tokens[:, 1:, :]  # 排除 CLS token
        mid_seg_logits = self.mid_seg_predictor(patch_tokens_mid)

        # 生成 Seg Attention Bias
        # mid_seg_logits: (B, 1, feat_h, feat_w) -> (B, num_patches)
        seg_prob = torch.sigmoid(mid_seg_logits)
        seg_bias = seg_prob.flatten(2).squeeze(1)  # (B, num_patches)

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

        # 分割 (使用最终的 patch tokens)
        seg_mask = self.seg_decoder(patch_tokens_out, feat_h, feat_w)
        if seg_mask.shape[2:] != (H, W):
            seg_mask = F.interpolate(seg_mask, size=(H, W), mode='bilinear', align_corners=False)

        return cls_logits, seg_mask

    def get_num_params(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# ============= 工厂函数 (RGB 单流) =============

def create_vipformer_seg_aware(
    pretrained: bool = True,
    img_height: int = 480,
    img_width: int = 640,
    stage1_layers: int = 2,
    stage2_layers: int = 2,
    hidden_dim: int = 768,
    **kwargs,
) -> VIPFormerSegAware:
    """创建 VIPFormer RGB 单流 Seg-Aware 模型"""
    config = VIPFormerSegAwareConfig(
        pretrained=pretrained,
        img_height=img_height,
        img_width=img_width,
        stage1_layers=stage1_layers,
        stage2_layers=stage2_layers,
        hidden_dim=hidden_dim,
        **kwargs,
    )
    return VIPFormerSegAware(config)


def vipformer_seg_aware_base(pretrained: bool = True, **kwargs) -> VIPFormerSegAware:
    """VIPFormer-SegAware-Base: 6+6 层 Transformer, 768 hidden dim"""
    return create_vipformer_seg_aware(
        pretrained=pretrained,
        stage1_layers=6,
        stage2_layers=6,
        hidden_dim=768,
        num_heads=12,
        mlp_dim=3072,
        **kwargs,
    )


def vipformer_seg_aware_small(pretrained: bool = True, **kwargs) -> VIPFormerSegAware:
    """VIPFormer-SegAware-Small: 3+3 层 Transformer, 512 hidden dim"""
    return create_vipformer_seg_aware(
        pretrained=pretrained,
        stage1_layers=3,
        stage2_layers=3,
        hidden_dim=512,
        num_heads=8,
        mlp_dim=2048,
        **kwargs,
    )


def vipformer_seg_aware_tiny(pretrained: bool = True, **kwargs) -> VIPFormerSegAware:
    """VIPFormer-SegAware-Tiny: 2+2 层 Transformer, 384 hidden dim"""
    return create_vipformer_seg_aware(
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
    model = vipformer_seg_aware_tiny()
    print(f"VIPFormer-SegAware-Tiny 参数量: {model.get_num_params() / 1e6:.2f}M")

    x = torch.randn(2, 4, 480, 640)
    cls_logits, seg_mask = model(x)
    print(f"输入: {x.shape}")
    print(f"分类输出: {cls_logits.shape}")
    print(f"分割输出: {seg_mask.shape}")
