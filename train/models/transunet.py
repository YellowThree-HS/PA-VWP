"""
TransUNet 模型架构
Hybrid-Encoder + Dual-Decoder 用于物理稳定性预测

架构概述:
- Encoder: ResNet-50 (修改输入通道为4) + Transformer Encoder
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


@dataclass
class TransUNetConfig:
    """TransUNet 配置"""
    # 输入配置
    img_height: int = 480
    img_width: int = 640
    in_channels: int = 4  # RGB (3) + Target Mask (1)
    
    # ResNet 配置
    pretrained: bool = True
    freeze_bn: bool = False
    
    # Transformer 配置
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    mlp_dim: int = 3072
    dropout: float = 0.1
    attention_dropout: float = 0.0
    
    # 解码器配置
    decoder_channels: Tuple[int, ...] = (256, 128, 64, 32)
    skip_channels: Tuple[int, ...] = (1024, 512, 256, 64)
    
    # 分类头配置
    num_classes: int = 1  # 二分类 (stable/unstable)
    cls_hidden_dim: int = 256
    
    # 分割头配置
    seg_classes: int = 1  # 受影响区域掩码


class PatchEmbedding(nn.Module):
    """
    将 CNN 特征图转换为 Transformer 序列
    输入: (B, C, H, W) -> 输出: (B, num_patches, hidden_dim)
    """
    
    def __init__(self, in_channels: int, hidden_dim: int, patch_size: int = 1):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, hidden_dim, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H'*W', hidden_dim)
        x = self.norm(x)
        return x, H, W


class PositionalEncoding(nn.Module):
    """可学习的位置编码"""
    
    def __init__(self, hidden_dim: int, max_len: int = 10000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, hidden_dim)
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer 编码器块"""
    
    def __init__(
        self, 
        hidden_dim: int, 
        num_heads: int, 
        mlp_dim: int, 
        dropout: float = 0.1,
        attention_dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadAttention(hidden_dim, num_heads, attention_dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Transformer 编码器 - 物理推理核心"""
    
    def __init__(self, config: TransUNetConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                mlp_dim=config.mlp_dim,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout
            )
            for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class UpBlock(nn.Module):
    """上采样块，支持跳跃连接"""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        skip_channels: int = 0,
        scale_factor: int = 2
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        
        if skip is not None:
            # 处理尺寸不匹配的情况
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            
        return self.conv(x)


class ClassificationHead(nn.Module):
    """稳定性分类头 (Task 1)"""
    
    def __init__(self, hidden_dim: int, cls_hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, cls_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(cls_hidden_dim, cls_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(cls_hidden_dim // 2, num_classes),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, hidden_dim)
        # 全局平均池化
        x = x.mean(dim=1)  # (B, hidden_dim)
        return self.mlp(x)  # (B, num_classes)


class SegmentationHead(nn.Module):
    """受影响区域分割头 (Task 2)"""
    
    def __init__(
        self, 
        config: TransUNetConfig,
        encoder_channels: Tuple[int, ...] = (2048,)
    ):
        super().__init__()
        
        # 从 Transformer 特征恢复到 2D
        # Transformer 输出: (B, H/16 * W/16, hidden_dim)
        # 需要 reshape 回 (B, hidden_dim, H/16, W/16)
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
        skip_features: list
    ) -> torch.Tensor:
        """
        Args:
            x: Transformer 输出 (B, seq_len, hidden_dim)
            feat_h, feat_w: 特征图高度和宽度 (H/16, W/16)
            skip_features: [layer1_feat, layer2_feat, layer3_feat, conv1_feat]
        """
        B = x.shape[0]
        
        # Reshape: (B, seq_len, hidden_dim) -> (B, hidden_dim, H, W)
        x = x.transpose(1, 2).reshape(B, -1, feat_h, feat_w)
        x = self.reshape_proj(x)  # (B, decoder_channels[0], H/16, W/16)
        
        # 上采样路径，从深到浅
        # skip_features: [layer3, layer2, layer1, conv1]
        for i, up_block in enumerate(self.up_blocks):
            skip = skip_features[i] if i < len(skip_features) else None
            x = up_block(x, skip)
            
        # 最终输出
        x = self.final_conv(x)
        return x


class ResNetEncoder(nn.Module):
    """
    ResNet-50 编码器
    修改第一层卷积以接受 4 通道输入 (RGB + Target Mask)
    """
    
    def __init__(self, config: TransUNetConfig):
        super().__init__()
        
        # 加载预训练 ResNet-50
        if config.pretrained:
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            resnet = resnet50(weights=None)
            
        # 修改第一层卷积：3 channels -> 4 channels
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(
            config.in_channels, 64, 
            kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # 初始化：复制 RGB 权重，第 4 通道 (mask) 初始化为 0
        with torch.no_grad():
            self.conv1.weight[:, :3, :, :] = original_conv1.weight
            self.conv1.weight[:, 3:, :, :] = 0  # Target mask 通道初始化为 0
            
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


class TransUNet(nn.Module):
    """
    TransUNet: Hybrid-Encoder + Dual-Decoder
    
    用于物理稳定性预测的双任务模型:
    - Task 1: 稳定性二分类 (stable / unstable)
    - Task 2: 受影响区域分割
    """
    
    def __init__(self, config: Optional[TransUNetConfig] = None):
        super().__init__()
        
        if config is None:
            config = TransUNetConfig()
        self.config = config
        
        # CNN Encoder (ResNet-50)
        self.cnn_encoder = ResNetEncoder(config)
        
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


def create_transunet(
    pretrained: bool = True,
    img_height: int = 480,
    img_width: int = 640,
    num_transformer_layers: int = 12,
    hidden_dim: int = 768,
    **kwargs
) -> TransUNet:
    """工厂函数：创建 TransUNet 模型"""
    config = TransUNetConfig(
        pretrained=pretrained,
        img_height=img_height,
        img_width=img_width,
        num_layers=num_transformer_layers,
        hidden_dim=hidden_dim,
        **kwargs
    )
    return TransUNet(config)


# 模型变体
def transunet_base(pretrained: bool = True, **kwargs) -> TransUNet:
    """TransUNet-Base: 12 层 Transformer, 768 hidden dim"""
    return create_transunet(
        pretrained=pretrained,
        num_transformer_layers=12,
        hidden_dim=768,
        num_heads=12,
        mlp_dim=3072,
        **kwargs
    )


def transunet_small(pretrained: bool = True, **kwargs) -> TransUNet:
    """TransUNet-Small: 6 层 Transformer, 512 hidden dim"""
    return create_transunet(
        pretrained=pretrained,
        num_transformer_layers=6,
        hidden_dim=512,
        num_heads=8,
        mlp_dim=2048,
        **kwargs
    )


def transunet_tiny(pretrained: bool = True, **kwargs) -> TransUNet:
    """TransUNet-Tiny: 4 层 Transformer, 384 hidden dim (用于快速实验)"""
    return create_transunet(
        pretrained=pretrained,
        num_transformer_layers=4,
        hidden_dim=384,
        num_heads=6,
        mlp_dim=1536,
        **kwargs
    )


if __name__ == "__main__":
    # 测试模型
    config = TransUNetConfig()
    model = TransUNet(config)
    
    # 打印模型信息
    print(f"TransUNet 参数量: {model.get_num_params() / 1e6:.2f}M (可训练)")
    print(f"TransUNet 参数量: {model.get_num_params(trainable_only=False) / 1e6:.2f}M (总计)")
    
    # 测试前向传播
    x = torch.randn(2, 4, 480, 640)
    cls_logits, seg_mask = model(x)
    
    print(f"输入尺寸: {x.shape}")
    print(f"分类输出尺寸: {cls_logits.shape}")  # (2, 1)
    print(f"分割输出尺寸: {seg_mask.shape}")    # (2, 1, 480, 640)
