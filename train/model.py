"""
BoxWorld 稳定性预测模型
基于ResNet-50，修改为4通道输入
"""

import torch
import torch.nn as nn
from torchvision import models


class StabilityPredictor(nn.Module):
    """稳定性预测模型

    输入: (B, 4, H, W) - RGB + Mask
    输出: (B, 1) - 稳定概率
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # 加载预训练ResNet-50
        resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)

        # 修改第一层卷积：3通道 -> 4通道
        old_conv = resnet.conv1
        self.conv1 = nn.Conv2d(
            4, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # 初始化：前3通道用预训练权重，第4通道初始化为0
        with torch.no_grad():
            self.conv1.weight[:, :3, :, :] = old_conv.weight
            self.conv1.weight[:, 3:, :, :] = 0

        # 复制其他层
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # 分类头
        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x
