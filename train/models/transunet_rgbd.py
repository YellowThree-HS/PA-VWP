"""
TransUNet RGBD 模型架构
Hybrid-Encoder + Dual-Decoder 用于物理稳定性预测 - 带深度图输入

架构概述:
- Encoder: ResNet-50 (修改输入通道为5) + Transformer Encoder
- 输入: RGB (3) + Depth (1) + Target Mask (1) = 5 channels
- Decoder A: 稳定性分类头 (Classification Head)
- Decoder B: 受影响区域分割头 (Segmentation Head with Skip Connections)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

# 复用原有模块
from .transunet import (
    PatchEmbedding,
    PositionalEncoding,
    TransformerEncoder,
    ClassificationHead,
    SegmentationHead,
    TransUNetConfig,
)


@dataclass
class TransUNetRGBDConfig(TransUNetConfig):
    """TransUNet RGBD 配置 - 5 通道输入"""
    # 输入配置
    img_height: int = 480
    img_width: int = 640
    in_channels: int = 5  # RGB (3) + Depth (1) + Target Mask (1)
    
    # ResNet 配置
    pretrained: bool = True
    freeze_bn: bool = False
    
    # Transformer 配置 (继承自父类)
    # hidden_dim, num_heads, num_layers, mlp_dim, dropout, attention_dropout
    
    # 解码器配置 (继承自父类)
    # decoder_channels, skip_channels
    
    # 分类头配置 (继承自父类)
    # num_classes, cls_hidden_dim
    
    # 分割头配置 (继承自父类)
    # seg_classes


class ResNetEncoderRGBD(nn.Module):
    """
    ResNet-50 编码器 - RGBD 版本
    修改第一层卷积以接受 5 通道输入 (RGB + Depth + Target Mask)
    """
    
    def __init__(self, config: TransUNetRGBDConfig):
        super().__init__()
        
        # 加载预训练 ResNet-50
        if config.pretrained:
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            resnet = resnet50(weights=None)
            
        # 修改第一层卷积：3 channels -> 5 channels
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(
            config.in_channels, 64, 
            kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # 初始化：复制 RGB 权重，额外通道初始化
        with torch.no_grad():
            # RGB 通道：复制预训练权重
            self.conv1.weight[:, :3, :, :] = original_conv1.weight
            
            # Depth 通道 (通道4)：使用 RGB 通道的平均值初始化
            # 这种初始化方式假设深度图与亮度信息相关
            self.conv1.weight[:, 3:4, :, :] = original_conv1.weight.mean(dim=1, keepdim=True)
            
            # Target Mask 通道 (通道5)：初始化为 0
            self.conv1.weight[:, 4:, :, :] = 0
            
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # ResNet 层 (不使用 layer4，节省内存)
        self.layer1 = resnet.layer1  # /4,  256 channels
        self.layer2 = resnet.layer2  # /8,  512 channels
        self.layer3 = resnet.layer3  # /16, 1024 channels
        # layer4 未使用，不加载以节省内存
        del resnet.layer4
        
        # 是否冻结 BatchNorm
        if config.freeze_bn:
            self._freeze_bn()
            
    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
                    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Returns:
            features: 最终特征图 (B, 1024, H/16, W/16) - 使用 layer3 输出
            skip_features: 用于分割头的跳跃连接特征
        """
        # Stem
        x = self.conv1(x)      # /2, 64 channels
        x = self.bn1(x)
        x = self.relu(x)
        conv1_feat = x         # 保存用于跳跃连接
        
        x = self.maxpool(x)    # /4
        
        # ResNet 块
        x1 = self.layer1(x)    # /4, 256 channels
        x2 = self.layer2(x1)   # /8, 512 channels
        x3 = self.layer3(x2)   # /16, 1024 channels
        
        # 跳跃连接特征 (从深到浅)
        skip_features = [x3, x2, x1, conv1_feat]
        
        return x3, skip_features


class TransUNetRGBD(nn.Module):
    """
    TransUNet RGBD: Hybrid-Encoder + Dual-Decoder - 带深度图输入
    
    用于物理稳定性预测的双任务模型:
    - Task 1: 稳定性二分类 (stable / unstable)
    - Task 2: 受影响区域分割
    - 输入: RGB (3) + Depth (1) + Target Mask (1) = 5 通道
    """
    
    def __init__(self, config: Optional[TransUNetRGBDConfig] = None):
        super().__init__()
        
        if config is None:
            config = TransUNetRGBDConfig()
        self.config = config
        
        # CNN Encoder (ResNet-50 RGBD)
        self.cnn_encoder = ResNetEncoderRGBD(config)
        
        # Patch Embedding: CNN 特征 -> Transformer 序列
        self.patch_embed = PatchEmbedding(
            in_channels=1024,  # layer3 输出通道数
            hidden_dim=config.hidden_dim
        )
        
        # 位置编码
        # 特征图尺寸: (H/16, W/16) = (30, 40) for 480x640 input
        max_patches = (config.img_height // 16) * (config.img_width // 16)
        self.pos_encoding = PositionalEncoding(
            hidden_dim=config.hidden_dim,
            max_len=max_patches + 1,  # +1 for CLS token
            dropout=config.dropout
        )
        
        # CLS Token (可选，用于分类)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Transformer Encoder (物理推理核心)
        self.transformer = TransformerEncoder(config)
        
        # Classification Head (Task 1)
        self.cls_head = ClassificationHead(
            hidden_dim=config.hidden_dim,
            cls_hidden_dim=config.cls_hidden_dim,
            num_classes=config.num_classes,
            dropout=config.dropout
        )
        
        # Segmentation Head (Task 2)
        self.seg_head = SegmentationHead(config)
        
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
            elif isinstance(m, nn.Conv2d):
                # 不重新初始化 ResNet 卷积权重
                if hasattr(m, '_initialized'):
                    continue
                    
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
        # encoded: (B, 1 + H/16 * W/16, hidden_dim)
        
        # 分离 CLS Token 和 patch tokens
        cls_token_out = encoded[:, 0:1, :]  # (B, 1, hidden_dim)
        patch_tokens = encoded[:, 1:, :]     # (B, H/16 * W/16, hidden_dim)
        
        # Task 1: Classification
        cls_logits = self.cls_head(encoded)  # 使用全部 tokens
        
        # Task 2: Segmentation
        seg_mask = self.seg_head(patch_tokens, feat_h, feat_w, skip_features)
        
        # 调整分割掩码尺寸到原始输入尺寸
        if seg_mask.shape[2:] != (H, W):
            seg_mask = F.interpolate(seg_mask, size=(H, W), mode='bilinear', align_corners=False)
            
        return cls_logits, seg_mask
    
    def get_num_params(self, trainable_only: bool = True) -> int:
        """获取模型参数数量"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_transunet_rgbd(
    pretrained: bool = True,
    img_height: int = 480,
    img_width: int = 640,
    num_transformer_layers: int = 12,
    hidden_dim: int = 768,
    **kwargs
) -> TransUNetRGBD:
    """工厂函数：创建 TransUNet RGBD 模型"""
    config = TransUNetRGBDConfig(
        pretrained=pretrained,
        img_height=img_height,
        img_width=img_width,
        num_layers=num_transformer_layers,
        hidden_dim=hidden_dim,
        in_channels=5,  # 确保是 5 通道
        **kwargs
    )
    return TransUNetRGBD(config)


# 模型变体
def transunet_rgbd_base(pretrained: bool = True, **kwargs) -> TransUNetRGBD:
    """TransUNet-RGBD-Base: 12 层 Transformer, 768 hidden dim"""
    return create_transunet_rgbd(
        pretrained=pretrained,
        num_transformer_layers=12,
        hidden_dim=768,
        num_heads=12,
        mlp_dim=3072,
        **kwargs
    )


def transunet_rgbd_small(pretrained: bool = True, **kwargs) -> TransUNetRGBD:
    """TransUNet-RGBD-Small: 6 层 Transformer, 512 hidden dim"""
    return create_transunet_rgbd(
        pretrained=pretrained,
        num_transformer_layers=6,
        hidden_dim=512,
        num_heads=8,
        mlp_dim=2048,
        **kwargs
    )


def transunet_rgbd_tiny(pretrained: bool = True, **kwargs) -> TransUNetRGBD:
    """TransUNet-RGBD-Tiny: 4 层 Transformer, 384 hidden dim (用于快速实验)"""
    return create_transunet_rgbd(
        pretrained=pretrained,
        num_transformer_layers=4,
        hidden_dim=384,
        num_heads=6,
        mlp_dim=1536,
        **kwargs
    )


if __name__ == "__main__":
    # 测试模型
    config = TransUNetRGBDConfig()
    model = TransUNetRGBD(config)
    
    # 打印模型信息
    print(f"TransUNet-RGBD 参数量: {model.get_num_params() / 1e6:.2f}M (可训练)")
    print(f"TransUNet-RGBD 参数量: {model.get_num_params(trainable_only=False) / 1e6:.2f}M (总计)")
    
    # 测试前向传播
    x = torch.randn(2, 5, 480, 640)  # 5 通道输入
    cls_logits, seg_mask = model(x)
    
    print(f"输入尺寸: {x.shape}")
    print(f"分类输出尺寸: {cls_logits.shape}")  # (2, 1)
    print(f"分割输出尺寸: {seg_mask.shape}")    # (2, 1, 480, 640)
