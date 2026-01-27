"""
TransUNet 双流网络模型架构
Two-Stream Network: RGB分支 + 深度分支，特征级融合

架构概述:
- RGB分支: ResNet-50 编码 RGB + Target Mask (4通道)
- 深度分支: ResNet-50 (轻量级) 编码 Depth (1通道)
- 特征融合: 在 CNN 特征级别进行融合，然后送入 Transformer
- Decoder: 分类头 + 分割头

实验目的:
验证双流网络对深度信息的独立建模能力，对比单流5通道输入
"""

import math
from dataclasses import dataclass, field
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
    SegmentationHead,
)
from .transunet_seg_from_encoded import (
    TransUNetSegFromEncodedConfig,
    SegmentationHeadFromEncoded,
)


@dataclass
class TransUNetTwoStreamConfig(TransUNetConfig):
    """TransUNet 双流网络配置"""
    # 输入配置
    img_height: int = 480
    img_width: int = 640
    rgb_channels: int = 4   # RGB (3) + Target Mask (1)
    depth_channels: int = 1  # Depth only
    
    # ResNet 配置
    pretrained: bool = True
    freeze_bn: bool = False
    
    # 深度分支配置 (可使用更轻量级的结构)
    depth_branch_type: str = 'resnet50'  # 'resnet50', 'resnet34', 'lite'
    depth_branch_hidden_dim: int = 256   # 深度分支输出维度
    
    # 融合配置
    fusion_type: str = 'concat'  # 'concat', 'add', 'attention'
    fusion_dim: int = 1024       # 融合后的特征维度
    
    # Transformer 配置 (继承自父类)
    # hidden_dim, num_heads, num_layers, mlp_dim, dropout, attention_dropout


@dataclass 
class TransUNetTwoStreamSegFromEncodedConfig(TransUNetTwoStreamConfig, TransUNetSegFromEncodedConfig):
    """双流网络 + SegFromEncoded 配置"""
    # 分割解码器配置 (无 skip connections)
    seg_decoder_channels: Tuple[int, ...] = (256, 128, 64, 32)
    seg_decoder_use_bn: bool = True


class RGBEncoder(nn.Module):
    """
    RGB分支编码器 (ResNet-50)
    处理 RGB + Target Mask (4通道输入)
    """
    
    def __init__(self, config: TransUNetTwoStreamConfig):
        super().__init__()
        
        # 加载预训练 ResNet-50
        if config.pretrained:
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            resnet = resnet50(weights=None)
            
        # 修改第一层卷积：3 channels -> 4 channels (RGB + TargetMask)
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(
            config.rgb_channels, 64, 
            kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # 初始化：复制 RGB 权重，额外通道初始化为 0
        with torch.no_grad():
            self.conv1.weight[:, :3, :, :] = original_conv1.weight
            self.conv1.weight[:, 3:, :, :] = 0  # Target Mask 通道
            
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        # ResNet 层
        self.layer1 = resnet.layer1  # /4,  256 channels
        self.layer2 = resnet.layer2  # /8,  512 channels
        self.layer3 = resnet.layer3  # /16, 1024 channels
        
        # 删除 layer4 节省内存
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
                    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: RGB + TargetMask (B, 4, H, W)
        Returns:
            features: 最终特征图 (B, 1024, H/16, W/16)
            skip_features: 用于分割头的跳跃连接特征
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        conv1_feat = x
        
        x = self.maxpool(x)
        
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        
        skip_features = [x3, x2, x1, conv1_feat]
        
        return x3, skip_features


class DepthEncoder(nn.Module):
    """
    深度分支编码器
    专门处理深度图 (1通道输入)
    """
    
    def __init__(self, config: TransUNetTwoStreamConfig):
        super().__init__()
        
        self.branch_type = config.depth_branch_type
        
        if config.depth_branch_type == 'resnet50':
            # 使用完整 ResNet-50
            if config.pretrained:
                resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            else:
                resnet = resnet50(weights=None)
                
            # 修改第一层卷积：1 channel
            original_conv1 = resnet.conv1
            self.conv1 = nn.Conv2d(
                config.depth_channels, 64, 
                kernel_size=7, stride=2, padding=3, bias=False
            )
            
            # 初始化：使用 RGB 通道的平均值
            with torch.no_grad():
                self.conv1.weight[:, 0:1, :, :] = original_conv1.weight.mean(dim=1, keepdim=True)
                
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool
            
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            
            del resnet.layer4
            
            self.out_channels = 1024
            
        elif config.depth_branch_type == 'lite':
            # 轻量级深度编码器
            self.encoder = nn.Sequential(
                # /2
                nn.Conv2d(config.depth_channels, 32, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                # /4
                nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                # /8
                nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                # /16
                nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
            self.out_channels = 256
        else:
            raise ValueError(f"Unknown depth branch type: {config.depth_branch_type}")
            
        # 是否冻结 BatchNorm
        if config.freeze_bn:
            self._freeze_bn()
            
    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Depth (B, 1, H, W)
        Returns:
            features: 深度特征图 (B, out_channels, H/16, W/16)
        """
        if self.branch_type == 'resnet50':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            return x
        else:
            return self.encoder(x)


class FeatureFusion(nn.Module):
    """
    特征融合模块
    融合 RGB 分支和深度分支的特征
    """
    
    def __init__(
        self, 
        rgb_channels: int, 
        depth_channels: int, 
        out_channels: int,
        fusion_type: str = 'concat'
    ):
        super().__init__()
        
        self.fusion_type = fusion_type
        
        if fusion_type == 'concat':
            # 拼接后投影
            self.proj = nn.Sequential(
                nn.Conv2d(rgb_channels + depth_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        elif fusion_type == 'add':
            # 先投影到相同维度，再相加
            self.rgb_proj = nn.Conv2d(rgb_channels, out_channels, 1, bias=False)
            self.depth_proj = nn.Conv2d(depth_channels, out_channels, 1, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
        elif fusion_type == 'attention':
            # 注意力融合
            self.rgb_proj = nn.Conv2d(rgb_channels, out_channels, 1, bias=False)
            self.depth_proj = nn.Conv2d(depth_channels, out_channels, 1, bias=False)
            
            # 通道注意力
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(out_channels * 2, out_channels),
                nn.ReLU(inplace=True),
                nn.Linear(out_channels, out_channels * 2),
                nn.Sigmoid(),
            )
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
            
    def forward(self, rgb_feat: torch.Tensor, depth_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb_feat: RGB 特征 (B, rgb_channels, H, W)
            depth_feat: 深度特征 (B, depth_channels, H, W)
        Returns:
            fused_feat: 融合特征 (B, out_channels, H, W)
        """
        if self.fusion_type == 'concat':
            concat_feat = torch.cat([rgb_feat, depth_feat], dim=1)
            return self.proj(concat_feat)
            
        elif self.fusion_type == 'add':
            rgb_proj = self.rgb_proj(rgb_feat)
            depth_proj = self.depth_proj(depth_feat)
            fused = rgb_proj + depth_proj
            return self.relu(self.bn(fused))
            
        elif self.fusion_type == 'attention':
            B = rgb_feat.shape[0]
            rgb_proj = self.rgb_proj(rgb_feat)
            depth_proj = self.depth_proj(depth_feat)
            
            # 通道注意力权重
            concat_pool = torch.cat([
                rgb_proj.mean(dim=(2, 3)),
                depth_proj.mean(dim=(2, 3))
            ], dim=1)
            attn = self.channel_attention(concat_pool)
            attn = attn.view(B, 2, -1)
            rgb_attn = attn[:, 0:1, :].unsqueeze(-1)
            depth_attn = attn[:, 1:2, :].unsqueeze(-1)
            
            fused = rgb_proj * rgb_attn + depth_proj * depth_attn
            return self.relu(self.bn(fused))


class TransUNetTwoStreamClsOnly(nn.Module):
    """
    TransUNet 双流网络 - 仅分类版本
    
    架构:
    - RGB分支: ResNet-50 处理 RGB + TargetMask
    - 深度分支: ResNet-50 处理 Depth
    - 特征融合: 在 CNN 特征级别融合
    - Transformer: 处理融合后的特征
    - 输出: 仅分类
    """
    
    def __init__(self, config: Optional[TransUNetTwoStreamConfig] = None):
        super().__init__()
        
        if config is None:
            config = TransUNetTwoStreamConfig()
        self.config = config
        
        # 双流编码器
        self.rgb_encoder = RGBEncoder(config)
        self.depth_encoder = DepthEncoder(config)
        
        # 特征融合
        rgb_out_channels = 1024  # ResNet-50 layer3 输出
        depth_out_channels = self.depth_encoder.out_channels
        
        self.fusion = FeatureFusion(
            rgb_channels=rgb_out_channels,
            depth_channels=depth_out_channels,
            out_channels=config.fusion_dim,
            fusion_type=config.fusion_type,
        )
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            in_channels=config.fusion_dim,
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
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 (B, 5, H, W) - RGB(3) + Depth(1) + TargetMask(1)
        Returns:
            cls_logits: 分类 logits (B, 1)
        """
        B, C, H, W = x.shape
        
        # 分离 RGB+Mask 和 Depth
        rgb_mask = torch.cat([x[:, :3], x[:, 4:5]], dim=1)  # (B, 4, H, W)
        depth = x[:, 3:4]  # (B, 1, H, W)
        
        # 双流编码
        rgb_feat, _ = self.rgb_encoder(rgb_mask)   # (B, 1024, H/16, W/16)
        depth_feat = self.depth_encoder(depth)     # (B, depth_channels, H/16, W/16)
        
        # 特征融合
        fused_feat = self.fusion(rgb_feat, depth_feat)  # (B, fusion_dim, H/16, W/16)
        
        # Patch Embedding
        patch_embeds, feat_h, feat_w = self.patch_embed(fused_feat)
        
        # 添加 CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, patch_embeds], dim=1)
        
        # 位置编码
        tokens = self.pos_encoding(tokens)
        
        # Transformer
        encoded = self.transformer(tokens)
        
        # Classification
        cls_logits = self.cls_head(encoded)
        
        return cls_logits
    
    def get_num_params(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class TransUNetTwoStreamSegFromEncoded(nn.Module):
    """
    TransUNet 双流网络 - SegFromEncoded 版本
    
    架构:
    - RGB分支: ResNet-50 处理 RGB + TargetMask
    - 深度分支: ResNet-50 处理 Depth
    - 特征融合: 在 CNN 特征级别融合
    - Transformer: 处理融合后的特征
    - 输出: 分类 + 分割 (分割头直接从 Transformer 解码，无 skip connections)
    """
    
    def __init__(self, config: Optional[TransUNetTwoStreamSegFromEncodedConfig] = None):
        super().__init__()
        
        if config is None:
            config = TransUNetTwoStreamSegFromEncodedConfig()
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
            out_channels=config.fusion_dim,
            fusion_type=config.fusion_type,
        )
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            in_channels=config.fusion_dim,
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
        
        # Segmentation Head (直接从 Encoded 特征解码)
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
                    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入张量 (B, 5, H, W) - RGB(3) + Depth(1) + TargetMask(1)
        Returns:
            cls_logits: 分类 logits (B, 1)
            seg_mask: 分割掩码 (B, 1, H, W)
        """
        B, C, H, W = x.shape
        
        # 分离 RGB+Mask 和 Depth
        rgb_mask = torch.cat([x[:, :3], x[:, 4:5]], dim=1)  # (B, 4, H, W)
        depth = x[:, 3:4]  # (B, 1, H, W)
        
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
        
        # Transformer Encoder
        encoded = self.transformer(tokens)
        
        # 分离 CLS Token 和 patch tokens
        patch_tokens = encoded[:, 1:, :]
        
        # Task 1: Classification
        cls_logits = self.cls_head(encoded)
        
        # Task 2: Segmentation
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

def create_twostream_cls_only(
    pretrained: bool = True,
    img_height: int = 480,
    img_width: int = 640,
    num_transformer_layers: int = 12,
    hidden_dim: int = 768,
    fusion_type: str = 'concat',
    depth_branch_type: str = 'resnet50',
    **kwargs
) -> TransUNetTwoStreamClsOnly:
    """创建双流网络 - 仅分类版本"""
    config = TransUNetTwoStreamConfig(
        pretrained=pretrained,
        img_height=img_height,
        img_width=img_width,
        num_layers=num_transformer_layers,
        hidden_dim=hidden_dim,
        fusion_type=fusion_type,
        depth_branch_type=depth_branch_type,
        **kwargs
    )
    return TransUNetTwoStreamClsOnly(config)


def create_twostream_seg_from_encoded(
    pretrained: bool = True,
    img_height: int = 480,
    img_width: int = 640,
    num_transformer_layers: int = 12,
    hidden_dim: int = 768,
    fusion_type: str = 'concat',
    depth_branch_type: str = 'resnet50',
    **kwargs
) -> TransUNetTwoStreamSegFromEncoded:
    """创建双流网络 - SegFromEncoded 版本"""
    config = TransUNetTwoStreamSegFromEncodedConfig(
        pretrained=pretrained,
        img_height=img_height,
        img_width=img_width,
        num_layers=num_transformer_layers,
        hidden_dim=hidden_dim,
        fusion_type=fusion_type,
        depth_branch_type=depth_branch_type,
        **kwargs
    )
    return TransUNetTwoStreamSegFromEncoded(config)


# ============= 模型变体 =============

# ClsOnly 变体
def twostream_cls_only_base(pretrained: bool = True, **kwargs) -> TransUNetTwoStreamClsOnly:
    """TwoStream-ClsOnly-Base: 12 层 Transformer, 768 hidden dim"""
    return create_twostream_cls_only(
        pretrained=pretrained,
        num_transformer_layers=12,
        hidden_dim=768,
        num_heads=12,
        mlp_dim=3072,
        **kwargs
    )


def twostream_cls_only_small(pretrained: bool = True, **kwargs) -> TransUNetTwoStreamClsOnly:
    """TwoStream-ClsOnly-Small: 6 层 Transformer, 512 hidden dim"""
    return create_twostream_cls_only(
        pretrained=pretrained,
        num_transformer_layers=6,
        hidden_dim=512,
        num_heads=8,
        mlp_dim=2048,
        **kwargs
    )


def twostream_cls_only_tiny(pretrained: bool = True, **kwargs) -> TransUNetTwoStreamClsOnly:
    """TwoStream-ClsOnly-Tiny: 4 层 Transformer, 384 hidden dim"""
    return create_twostream_cls_only(
        pretrained=pretrained,
        num_transformer_layers=4,
        hidden_dim=384,
        num_heads=6,
        mlp_dim=1536,
        **kwargs
    )


def twostream_cls_only_micro(pretrained: bool = True, **kwargs) -> TransUNetTwoStreamClsOnly:
    """TwoStream-ClsOnly-Micro: 2 层 Transformer, 192 hidden dim"""
    return create_twostream_cls_only(
        pretrained=pretrained,
        num_transformer_layers=2,
        hidden_dim=192,
        num_heads=3,
        mlp_dim=768,
        **kwargs
    )


# SegFromEncoded 变体
def twostream_seg_from_encoded_base(pretrained: bool = True, **kwargs) -> TransUNetTwoStreamSegFromEncoded:
    """TwoStream-SegFromEncoded-Base: 12 层 Transformer, 768 hidden dim"""
    return create_twostream_seg_from_encoded(
        pretrained=pretrained,
        num_transformer_layers=12,
        hidden_dim=768,
        num_heads=12,
        mlp_dim=3072,
        **kwargs
    )


def twostream_seg_from_encoded_small(pretrained: bool = True, **kwargs) -> TransUNetTwoStreamSegFromEncoded:
    """TwoStream-SegFromEncoded-Small: 6 层 Transformer, 512 hidden dim"""
    return create_twostream_seg_from_encoded(
        pretrained=pretrained,
        num_transformer_layers=6,
        hidden_dim=512,
        num_heads=8,
        mlp_dim=2048,
        **kwargs
    )


def twostream_seg_from_encoded_tiny(pretrained: bool = True, **kwargs) -> TransUNetTwoStreamSegFromEncoded:
    """TwoStream-SegFromEncoded-Tiny: 4 层 Transformer, 384 hidden dim"""
    return create_twostream_seg_from_encoded(
        pretrained=pretrained,
        num_transformer_layers=4,
        hidden_dim=384,
        num_heads=6,
        mlp_dim=1536,
        **kwargs
    )


def twostream_seg_from_encoded_micro(pretrained: bool = True, **kwargs) -> TransUNetTwoStreamSegFromEncoded:
    """TwoStream-SegFromEncoded-Micro: 2 层 Transformer, 192 hidden dim"""
    return create_twostream_seg_from_encoded(
        pretrained=pretrained,
        num_transformer_layers=2,
        hidden_dim=192,
        num_heads=3,
        mlp_dim=768,
        **kwargs
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Testing TwoStream-ClsOnly-Tiny")
    print("=" * 60)
    
    model_cls = twostream_cls_only_tiny()
    print(f"参数量: {model_cls.get_num_params() / 1e6:.2f}M")
    
    x = torch.randn(2, 5, 480, 640)
    cls_logits = model_cls(x)
    print(f"输入: {x.shape}, 输出: {cls_logits.shape}")
    
    print("\n" + "=" * 60)
    print("Testing TwoStream-SegFromEncoded-Tiny")
    print("=" * 60)
    
    model_seg = twostream_seg_from_encoded_tiny()
    print(f"参数量: {model_seg.get_num_params() / 1e6:.2f}M")
    
    cls_logits, seg_mask = model_seg(x)
    print(f"输入: {x.shape}")
    print(f"分类输出: {cls_logits.shape}")
    print(f"分割输出: {seg_mask.shape}")
