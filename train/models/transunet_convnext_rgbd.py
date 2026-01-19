"""
TransUNet ConvNeXt RGBD 模型架构
使用 ConvNeXt 替代 ResNet-50 作为 CNN Encoder - 带深度图输入

架构概述:
- Encoder: ConvNeXt (前三个 Stage) + Transformer Encoder
- 输入: RGB (3) + Depth (1) + Target Mask (1) = 5 通道
- Decoder A: 稳定性分类头 (Classification Head)
- Decoder B: 受影响区域分割头 (Segmentation Head with Skip Connections)

ConvNeXt 优势:
- 借鉴 Transformer 的设计 (大 Kernel size, Layer Norm)
- 通常比 ResNet-50 性能更强
- 作为 Feature Extractor 非常流行
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

# 复用原有模块
from .transunet import (
    PatchEmbedding,
    PositionalEncoding,
    TransformerEncoder,
    ClassificationHead,
)
from .transunet_convnext import (
    TransUNetConvNeXtConfig,
    UpBlockConvNeXt,
    SegmentationHeadConvNeXt,
)


@dataclass
class TransUNetConvNeXtRGBDConfig(TransUNetConvNeXtConfig):
    """TransUNet ConvNeXt RGBD 配置 - 5 通道输入"""
    # 输入配置
    img_height: int = 480
    img_width: int = 640
    in_channels: int = 5  # RGB (3) + Depth (1) + Target Mask (1)
    
    # 其他配置继承自 TransUNetConvNeXtConfig


class ConvNeXtEncoderRGBD(nn.Module):
    """
    ConvNeXt 编码器 - RGBD 版本
    使用 ConvNeXt-Base 的前三个 Stage 作为特征提取器
    修改第一层以接受 5 通道输入 (RGB + Depth + Target Mask)
    """
    
    def __init__(self, config: TransUNetConvNeXtRGBDConfig):
        super().__init__()
        
        # 加载预训练 ConvNeXt-Base
        if config.pretrained:
            convnext = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        else:
            convnext = convnext_base(weights=None)
        
        # ConvNeXt 结构:
        # features[0]: Stem (Conv2d + LayerNorm) -> 128 channels, /4
        # features[1]: Stage 1 -> 128 channels
        # features[2]: Downsample 1 -> 256 channels
        # features[3]: Stage 2 -> 256 channels
        # features[4]: Downsample 2 -> 512 channels
        # features[5]: Stage 3 -> 512 channels
        
        # 修改 Stem 的第一个卷积层以接受 5 通道
        original_stem = convnext.features[0]
        original_conv = original_stem[0]  # Conv2d(3, 128, kernel_size=4, stride=4)
        
        # 创建新的 5 通道卷积
        self.stem_conv = nn.Conv2d(
            config.in_channels, 128,
            kernel_size=4, stride=4, padding=0
        )
        
        # 初始化权重
        with torch.no_grad():
            # 复制 RGB 权重
            self.stem_conv.weight[:, :3, :, :] = original_conv.weight
            self.stem_conv.bias[:] = original_conv.bias
            
            # Depth 通道 (通道4): 使用 RGB 通道的平均值初始化
            # 这种初始化方式假设深度图与亮度信息相关
            self.stem_conv.weight[:, 3:4, :, :] = original_conv.weight.mean(dim=1, keepdim=True)
            
            # Target mask 通道 (通道5): 初始化为 0
            self.stem_conv.weight[:, 4:, :, :] = 0
            
        # Stem LayerNorm
        self.stem_norm = original_stem[1]
        
        # Stage 1: 128 channels, /4
        self.stage1 = convnext.features[1]
        
        # Downsample 1 + Stage 2: 256 channels, /8
        self.downsample1 = convnext.features[2]
        self.stage2 = convnext.features[3]
        
        # Downsample 2 + Stage 3: 512 channels, /16
        self.downsample2 = convnext.features[4]
        self.stage3 = convnext.features[5]
        
        # 不使用 Stage 4，节省内存
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Returns:
            features: 最终特征图 (B, 512, H/16, W/16) - Stage 3 输出
            skip_features: 用于分割头的跳跃连接特征
        """
        # Stem: /4, 128 channels
        x = self.stem_conv(x)
        # stem_norm 是 LayerNorm2d，内部会自动处理 (B,C,H,W) -> (B,H,W,C) -> (B,C,H,W)
        x = self.stem_norm(x)
        stem_feat = x
        
        # Stage 1: /4, 128 channels
        x = self.stage1(x)
        stage1_feat = x
        
        # Downsample 1 + Stage 2: /8, 256 channels
        x = self.downsample1(x)
        x = self.stage2(x)
        stage2_feat = x
        
        # Downsample 2 + Stage 3: /16, 512 channels
        x = self.downsample2(x)
        x = self.stage3(x)
        stage3_feat = x
        
        # 跳跃连接特征 (从深到浅)
        skip_features = [stage3_feat, stage2_feat, stage1_feat, stem_feat]
        
        return x, skip_features


class TransUNetConvNeXtRGBD(nn.Module):
    """
    TransUNet with ConvNeXt Encoder - RGBD 版本
    
    使用 ConvNeXt 替代 ResNet-50 的双任务模型 (带深度图):
    - Task 1: 稳定性二分类 (stable / unstable)
    - Task 2: 受影响区域分割
    - 输入: RGB (3) + Depth (1) + Target Mask (1) = 5 通道
    """
    
    def __init__(self, config: Optional[TransUNetConvNeXtRGBDConfig] = None):
        super().__init__()
        
        if config is None:
            config = TransUNetConvNeXtRGBDConfig()
        self.config = config
        
        # CNN Encoder (ConvNeXt RGBD)
        self.cnn_encoder = ConvNeXtEncoderRGBD(config)
        
        # Patch Embedding: CNN 特征 -> Transformer 序列
        self.patch_embed = PatchEmbedding(
            in_channels=config.cnn_out_channels,  # 512 for ConvNeXt-Base
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
        
        # Classification Head
        self.cls_head = ClassificationHead(
            hidden_dim=config.hidden_dim,
            cls_hidden_dim=config.cls_hidden_dim,
            num_classes=config.num_classes,
            dropout=config.dropout
        )
        
        # Segmentation Head
        self.seg_head = SegmentationHeadConvNeXt(config)
        
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
            return_attention: 是否返回注意力权重
            
        Returns:
            cls_logits: 分类 logits (B, 1)
            seg_mask: 分割掩码 (B, 1, H, W)
        """
        B, C, H, W = x.shape
        
        # CNN Encoder
        cnn_features, skip_features = self.cnn_encoder(x)
        
        # Patch Embedding
        patch_embeds, feat_h, feat_w = self.patch_embed(cnn_features)
        
        # 添加 CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, patch_embeds], dim=1)
        
        # 位置编码
        tokens = self.pos_encoding(tokens)
        
        # Transformer Encoder
        encoded = self.transformer(tokens)
        
        # 分离 CLS Token 和 patch tokens
        patch_tokens = encoded[:, 1:, :]
        
        # Task 1: Classification
        cls_logits = self.cls_head(encoded)
        
        # Task 2: Segmentation
        seg_mask = self.seg_head(patch_tokens, feat_h, feat_w, skip_features)
        
        # 调整分割掩码尺寸
        if seg_mask.shape[2:] != (H, W):
            seg_mask = F.interpolate(seg_mask, size=(H, W), mode='bilinear', align_corners=False)
            
        return cls_logits, seg_mask
    
    def get_num_params(self, trainable_only: bool = True) -> int:
        """获取模型参数数量"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_transunet_convnext_rgbd(
    pretrained: bool = True,
    img_height: int = 480,
    img_width: int = 640,
    num_transformer_layers: int = 12,
    hidden_dim: int = 768,
    **kwargs
) -> TransUNetConvNeXtRGBD:
    """工厂函数：创建 TransUNet-ConvNeXt-RGBD 模型"""
    config = TransUNetConvNeXtRGBDConfig(
        pretrained=pretrained,
        img_height=img_height,
        img_width=img_width,
        num_layers=num_transformer_layers,
        hidden_dim=hidden_dim,
        in_channels=5,
        **kwargs
    )
    return TransUNetConvNeXtRGBD(config)


# 模型变体
def transunet_convnext_rgbd_base(pretrained: bool = True, **kwargs) -> TransUNetConvNeXtRGBD:
    """TransUNet-ConvNeXt-RGBD-Base: 12 层 Transformer, 768 hidden dim"""
    return create_transunet_convnext_rgbd(
        pretrained=pretrained,
        num_transformer_layers=12,
        hidden_dim=768,
        num_heads=12,
        mlp_dim=3072,
        **kwargs
    )


def transunet_convnext_rgbd_small(pretrained: bool = True, **kwargs) -> TransUNetConvNeXtRGBD:
    """TransUNet-ConvNeXt-RGBD-Small: 6 层 Transformer, 512 hidden dim"""
    return create_transunet_convnext_rgbd(
        pretrained=pretrained,
        num_transformer_layers=6,
        hidden_dim=512,
        num_heads=8,
        mlp_dim=2048,
        **kwargs
    )


def transunet_convnext_rgbd_tiny(pretrained: bool = True, **kwargs) -> TransUNetConvNeXtRGBD:
    """TransUNet-ConvNeXt-RGBD-Tiny: 4 层 Transformer, 384 hidden dim"""
    return create_transunet_convnext_rgbd(
        pretrained=pretrained,
        num_transformer_layers=4,
        hidden_dim=384,
        num_heads=6,
        mlp_dim=1536,
        **kwargs
    )


if __name__ == "__main__":
    # 测试模型
    config = TransUNetConvNeXtRGBDConfig()
    model = TransUNetConvNeXtRGBD(config)
    
    # 打印模型信息
    print(f"TransUNet-ConvNeXt-RGBD 参数量: {model.get_num_params() / 1e6:.2f}M (可训练)")
    print(f"TransUNet-ConvNeXt-RGBD 参数量: {model.get_num_params(trainable_only=False) / 1e6:.2f}M (总计)")
    
    # 测试前向传播
    x = torch.randn(2, 5, 480, 640)  # 5 通道输入
    cls_logits, seg_mask = model(x)
    
    print(f"输入尺寸: {x.shape}")
    print(f"分类输出尺寸: {cls_logits.shape}")  # (2, 1)
    print(f"分割输出尺寸: {seg_mask.shape}")    # (2, 1, 480, 640)
