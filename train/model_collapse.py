"""
CollapseNet: 多任务学习网络用于坍塌预测

架构:
- 共享编码器 (ResNet-50 backbone) 提取特征
- 分类头 (Stability Head) 预测坍塌概率
- 分割头 (Segmentation Decoder) 预测受影响区域掩码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBlock(nn.Module):
    """卷积块: Conv -> BN -> ReLU"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    """U-Net 解码器块: 上采样 -> Concat -> Conv -> Conv"""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        # 上采样: 双线性插值
        self.up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        # 融合卷积: in_ch (上采样) + skip_ch (跳跃连接)
        self.conv = nn.Sequential(
            ConvBlock(in_ch + skip_ch, out_ch),
            ConvBlock(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # 处理尺寸不匹配 (可能需要中心裁剪)
        if x.size() != skip.size():
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=True
            )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ClassificationHead(nn.Module):
    """稳定性分类头: 全局特征 -> MLP -> 二分类"""

    def __init__(self, in_channels=2048, hidden_dim=512, dropout=0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # GAP: (B, C, 1, 1)
            nn.Flatten(),  # (B, C)
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # Logit
        )

    def forward(self, x):
        return self.head(x)


class SegmentationDecoder(nn.Module):
    """U-Net 风格分割解码器"""

    def __init__(self, encoder_channels, decoder_channels):
        """
        Args:
            encoder_channels: [64, 256, 512, 2048] - ResNet 各层输出通道
            decoder_channels: 解码器各层通道数
        """
        super().__init__()

        # 跳跃连接特征 (从编码器各层获取)
        self.skip1 = encoder_channels[0]  # 64
        self.skip2 = encoder_channels[1]  # 256
        self.skip3 = encoder_channels[2]  # 512
        # encoder_channels[3] 是瓶颈特征 (2048)

        # 解码器结构 (U-Net style)
        # 从 bottleneck (2048) 开始，逐步上采样
        self.dec4 = DecoderBlock(2048, self.skip3, decoder_channels[0])  # 1024
        self.dec3 = DecoderBlock(decoder_channels[0], self.skip2, decoder_channels[1])  # 512
        self.dec2 = DecoderBlock(decoder_channels[1], self.skip1, decoder_channels[2])  # 256
        self.dec1 = DecoderBlock(decoder_channels[2], self.skip1, decoder_channels[3])  # 64

        # 最终输出层
        self.out_conv = nn.Conv2d(decoder_channels[3], 1, kernel_size=1)

    def forward(self, bottleneck, skip_features):
        """
        Args:
            bottleneck: (B, 2048, H/32, W/32)
            skip_features: dict with keys '1', '2', '3', '4'
        """
        # Stage 4: 瓶颈 + skip3 (512)
        x = self.dec4(bottleneck, skip_features['3'])

        # Stage 3: + skip2 (256)
        x = self.dec3(x, skip_features['2'])

        # Stage 2: + skip1 (64)
        x = self.dec2(x, skip_features['1'])

        # Stage 1: + skip1 (64) again (用于恢复分辨率)
        x = self.dec1(x, skip_features['1'])

        # 输出: (B, 1, H, W)
        return self.out_conv(x)


class CollapseNet(nn.Module):
    """坍塌预测多任务网络

    输入: (B, 4, H, W) - RGB + Target Mask
    输出:
        - stability: (B, 1) - 稳定性概率
        - segmentation: (B, 1, H, W) - 受影响区域掩码
    """

    def __init__(
        self,
        backbone="resnet50",
        pretrained=True,
        num_classes=1,
        decoder_channels=[1024, 512, 256, 64],
        dropout=0.5,
    ):
        super().__init__()

        # === 1. 共享编码器 (ResNet) ===
        if backbone == "resnet50":
            resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            encoder_channels = [64, 256, 512, 2048]  # 各层输出通道
        elif backbone == "resnet101":
            resnet = models.resnet101(weights='IMAGENET1K_V1' if pretrained else None)
            encoder_channels = [64, 256, 512, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # 修改第一层: 3通道 -> 4通道 (RGB + Target Mask)
        old_conv = resnet.conv1
        self.conv1 = nn.Conv2d(
            4, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # 初始化: 前3通道用预训练权重，第4通道初始化为0
        if pretrained:
            with torch.no_grad():
                self.conv1.weight[:, :3, :, :] = old_conv.weight
                self.conv1.weight[:, 3:, :, :] = 0
        else:
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')

        # 编码器其余部分
        self.bn1 = resnet.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 64 -> 64 (H/4, W/4)
        self.layer2 = resnet.layer2  # 64 -> 256 (H/8, W/8)
        self.layer3 = resnet.layer3  # 256 -> 512 (H/16, W/16)
        self.layer4 = resnet.layer4  # 512 -> 2048 (H/32, W/32)

        # === 2. 分类头 ===
        self.cls_head = ClassificationHead(
            in_channels=encoder_channels[-1],
            hidden_dim=512,
            dropout=dropout
        )

        # === 3. 分割解码器 ===
        self.seg_decoder = SegmentationDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels
        )

    def forward(self, x):
        """
        Args:
            x: (B, 4, H, W) 输入图像 (RGB + Mask)

        Returns:
            stability: (B, 1) 稳定性 logits
            segmentation: (B, 1, H, W) 分割 logits
        """

        # === 编码器前向传播 ===
        # 保存跳跃连接特征
        skip_features = {}

        # Stage 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # (B, 64, H/4, W/4)
        skip_features['1'] = x

        # Stage 2
        x = self.layer1(x)  # (B, 256, H/4, W/4)
        skip_features['2'] = x

        # Stage 3
        x = self.layer2(x)  # (B, 512, H/8, W/8)
        skip_features['3'] = x

        # Stage 4 (Bottleneck)
        x = self.layer3(x)
        bottleneck = self.layer4(x)  # (B, 2048, H/32, W/32)

        # === 分支 1: 分类头 ===
        stability = self.cls_head(bottleneck)  # (B, 1)

        # === 分支 2: 分割解码器 ===
        segmentation = self.seg_decoder(bottleneck, skip_features)  # (B, 1, H, W)

        return stability, segmentation

    def get_features_for_late_fusion(self):
        """获取用于后期融合的特征提取器 (用于迁移学习)"""
        return nn.ModuleDict({
            'conv1': self.conv1,
            'bn1': self.bn1,
            'relu': self.relu,
            'maxpool': self.maxpool,
            'layer1': self.layer1,
            'layer2': self.layer2,
            'layer3': self.layer3,
            'layer4': self.layer4,
        })


class CollapseNetWithLateFusion(CollapseNet):
    """支持后期融合的 CollapseNet 变体"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 添加额外的局部特征处理分支
        self.local_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Sequential(*list(self.layer1.children())),  # reuse layer1
            nn.Sequential(*list(self.layer2.children())),  # reuse layer2
        )

        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048 + 256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )

    def forward(self, global_x, local_x=None):
        """
        Args:
            global_x: (B, 4, H, W) 全局视图
            local_x: (B, 3, H, W) 局部裁剪视图 (可选)
        """
        # 全局分支
        skip_features = {}
        gx = self.conv1(global_x)
        gx = self.bn1(gx)
        gx = self.relu(gx)
        gx = self.maxpool(gx)
        skip_features['1'] = gx

        gx = self.layer1(gx)
        skip_features['2'] = gx

        gx = self.layer2(gx)
        skip_features['3'] = gx

        gx = self.layer3(gx)
        bottleneck = self.layer4(gx)

        # 全局特征
        global_feat = self.cls_head.head[1](bottleneck)  # (B, 2048)

        if local_x is not None:
            # 局部分支
            lx = self.local_encoder(local_x)
            lx = lx.view(lx.size(0), -1)  # (B, 256)

            # 融合
            fused = torch.cat([global_feat, lx], dim=1)
            stability = self.fusion_layer(fused)
        else:
            stability = self.cls_head(bottleneck)

        segmentation = self.seg_decoder(bottleneck, skip_features)

        return stability, segmentation
