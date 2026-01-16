"""
BoxWorld 双流网络稳定性预测模型
Two-Stream "Focus" Network

- Global Stream: 全局视图，判断整体堆叠结构
- Local Stream: 局部裁剪，观察接触点细节
"""

import torch
import torch.nn as nn
from torchvision import models


class TwoStreamStabilityPredictor(nn.Module):
    """双流稳定性预测模型

    输入:
        global_input: (B, 4, 224, 224) - 全局视图 RGB + Mask
        local_input: (B, 4, 224, 224) - 局部裁剪 RGB + Mask

    输出: (B, 1) - 稳定概率
    """

    def __init__(self, pretrained: bool = True, backbone: str = "resnet50"):
        super().__init__()

        self.global_stream = self._create_stream(pretrained, backbone)
        self.local_stream = self._create_stream(pretrained, backbone)

        # 特征维度
        if backbone == "resnet50":
            feat_dim = 2048
        elif backbone == "resnet34":
            feat_dim = 512
        else:
            feat_dim = 2048

        # 融合分类头：两个流的特征concat后分类
        self.fusion_head = nn.Sequential(
            nn.Linear(feat_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def _create_stream(self, pretrained: bool, backbone: str):
        """创建单个流的backbone"""
        if backbone == "resnet50":
            resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        elif backbone == "resnet34":
            resnet = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        else:
            resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)

        # 修改第一层卷积：3通道 -> 4通道
        old_conv = resnet.conv1
        new_conv = nn.Conv2d(
            4, old_conv.out_channels,
            kernel_size=7, stride=2, padding=3, bias=False
        )

        # 初始化：前3通道用预训练权重，第4通道初始化为0
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            new_conv.weight[:, 3:, :, :] = 0

        # 构建stream（不包含fc层）
        stream = nn.Sequential(
            new_conv,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
            nn.Flatten()
        )

        return stream

    def forward(self, global_input, local_input):
        # 提取全局特征
        global_feat = self.global_stream(global_input)  # (B, feat_dim)

        # 提取局部特征
        local_feat = self.local_stream(local_input)  # (B, feat_dim)

        # 融合
        fused = torch.cat([global_feat, local_feat], dim=1)  # (B, feat_dim*2)

        # 分类
        out = self.fusion_head(fused)

        return out


if __name__ == "__main__":
    # 测试模型
    model = TwoStreamStabilityPredictor(pretrained=False)
    global_input = torch.randn(2, 4, 224, 224)
    local_input = torch.randn(2, 4, 224, 224)
    output = model(global_input, local_input)
    print(f"Output shape: {output.shape}")  # (2, 1)
    print(f"Output: {output}")
