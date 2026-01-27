# TransUNet RGBD + 双流网络模型结构说明

本文档详细介绍项目中 RGBD 输入相关的模型架构，包括**单流 RGBD 模型**和**双流 Two-Stream 模型**。

---

## 1. 概述

### 1.1 任务背景

本项目用于**物理稳定性预测**，包含两个子任务：
- **Task 1: 稳定性分类** - 二分类任务 (stable / unstable)
- **Task 2: 受影响区域分割** - 预测物体移除后的受影响区域

### 1.2 输入格式

所有模型接收 **5 通道输入**：
- **RGB** (3 通道): 场景彩色图像
- **Depth** (1 通道): 深度图
- **Target Mask** (1 通道): 目标物体掩码

输入尺寸: `(B, 5, 480, 640)`

### 1.3 模型变体对比

| 模型类型 | 深度处理方式 | 特征融合位置 | 特点 |
|---------|-------------|-------------|------|
| TransUNet RGBD | 5通道直接输入单个 ResNet | 输入层 | 简单高效，早期融合 |
| TwoStream-ClsOnly | 双分支分别处理 | CNN 特征级 | 独立建模深度信息 |
| TwoStream-SegFromEncoded | 双分支分别处理 | CNN 特征级 | 支持分割任务，无 skip connections |

---

## 2. TransUNet RGBD (单流模型)

### 2.1 架构图

```
输入: RGB(3) + Depth(1) + TargetMask(1) = 5通道
                    │
                    ▼
    ┌──────────────────────────────────┐
    │     ResNet-50 Encoder (5通道)     │
    │  ┌─────────────────────────────┐ │
    │  │ Conv1 (5→64, 7x7, stride=2) │ │  → /2, 保存 conv1_feat
    │  │ BN + ReLU + MaxPool         │ │  → /4
    │  │ Layer1 (256 channels)       │ │  → /4, 保存 x1
    │  │ Layer2 (512 channels)       │ │  → /8, 保存 x2
    │  │ Layer3 (1024 channels)      │ │  → /16, 保存 x3 (输出)
    │  └─────────────────────────────┘ │
    │          skip_features: [x3, x2, x1, conv1_feat]
    └──────────────────────────────────┘
                    │
                    ▼ (B, 1024, H/16, W/16)
    ┌──────────────────────────────────┐
    │         Patch Embedding           │
    │  Conv2d(1024→hidden_dim, 1x1)    │
    │  LayerNorm                        │
    └──────────────────────────────────┘
                    │
                    ▼ (B, 1200, hidden_dim)  # 30x40=1200 patches
    ┌──────────────────────────────────┐
    │      CLS Token + 位置编码          │
    │  [CLS] ⊕ patch_embeds            │
    │  + Positional Encoding           │
    └──────────────────────────────────┘
                    │
                    ▼ (B, 1201, hidden_dim)
    ┌──────────────────────────────────┐
    │      Transformer Encoder          │
    │  N × TransformerBlock:           │
    │    - LayerNorm                    │
    │    - Multi-Head Self-Attention   │
    │    - LayerNorm                    │
    │    - MLP (FFN)                    │
    └──────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐      ┌───────────────┐
│Classification │      │ Segmentation  │
│    Head       │      │    Head       │
│  全局池化      │      │ 使用 skip     │
│  MLP→(B,1)   │      │ connections   │
└───────────────┘      └───────────────┘
        │                       │
        ▼                       ▼
   cls_logits              seg_mask
    (B, 1)              (B, 1, 480, 640)
```

### 2.2 核心组件

#### 2.2.1 ResNet-50 RGBD Encoder

```python
class ResNetEncoderRGBD(nn.Module):
    """
    修改 ResNet-50 第一层卷积以接受 5 通道输入
    """
    def __init__(self, config):
        # 加载预训练 ResNet-50
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # 修改第一层: 3 channels → 5 channels
        self.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 权重初始化策略:
        # - RGB 通道 (0-2): 复制预训练权重
        # - Depth 通道 (3): 使用 RGB 权重均值初始化
        # - TargetMask 通道 (4): 初始化为 0
        
        self.layer1 = resnet.layer1  # /4,  256 channels
        self.layer2 = resnet.layer2  # /8,  512 channels  
        self.layer3 = resnet.layer3  # /16, 1024 channels
        # 不使用 layer4，节省内存
```

#### 2.2.2 分割头 (带 Skip Connections)

```python
class SegmentationHead(nn.Module):
    """
    使用 CNN Encoder 的多尺度特征进行上采样
    
    上采样路径:
    H/16 → H/8  (跳跃连接 layer3: 1024 channels)
    H/8  → H/4  (跳跃连接 layer2: 512 channels)
    H/4  → H/2  (跳跃连接 layer1: 256 channels)
    H/2  → H    (跳跃连接 conv1:  64 channels)
    """
    decoder_channels = (256, 128, 64, 32)
    skip_channels = (1024, 512, 256, 64)
```

### 2.3 模型配置

```python
@dataclass
class TransUNetRGBDConfig:
    # 输入配置
    img_height: int = 480
    img_width: int = 640
    in_channels: int = 5  # RGB(3) + Depth(1) + TargetMask(1)
    
    # Transformer 配置
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    mlp_dim: int = 3072
    dropout: float = 0.1
    
    # 解码器配置
    decoder_channels: Tuple[int, ...] = (256, 128, 64, 32)
    skip_channels: Tuple[int, ...] = (1024, 512, 256, 64)
```

### 2.4 模型变体

| 变体 | Transformer 层数 | Hidden Dim | Heads | MLP Dim | 参数量 |
|-----|-----------------|------------|-------|---------|-------|
| Base | 12 | 768 | 12 | 3072 | ~95M |
| Small | 6 | 512 | 8 | 2048 | ~50M |
| Tiny | 4 | 384 | 6 | 1536 | ~35M |

---

## 3. TransUNet TwoStream (双流模型)

### 3.1 架构图

```
输入: RGB(3) + Depth(1) + TargetMask(1) = 5通道
         │
         │ 分离通道
         ├──────────────────────┐
         ▼                      ▼
RGB + TargetMask (4通道)    Depth (1通道)
         │                      │
         ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│  RGB Encoder    │    │  Depth Encoder  │
│  (ResNet-50)    │    │  (ResNet-50)    │
│                 │    │                 │
│ Conv1: 4→64    │    │ Conv1: 1→64    │
│ Layer1: 256    │    │ Layer1: 256    │
│ Layer2: 512    │    │ Layer2: 512    │
│ Layer3: 1024   │    │ Layer3: 1024   │
└─────────────────┘    └─────────────────┘
         │                      │
         ▼                      ▼
   (B, 1024, H/16, W/16)  (B, 1024, H/16, W/16)
         │                      │
         └──────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │    Feature Fusion     │
        │                       │
        │ 融合方式:              │
        │ - concat (拼接投影)   │
        │ - add (相加)          │
        │ - attention (注意力)  │
        └───────────────────────┘
                    │
                    ▼ (B, fusion_dim, H/16, W/16)
        ┌───────────────────────┐
        │    Patch Embedding    │
        │  Conv2d → LayerNorm   │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  CLS Token + 位置编码  │
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Transformer Encoder  │
        │  N × TransformerBlock │
        └───────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
 Classification           Segmentation
    Head                     Head
    (B, 1)               (B, 1, H, W)
```

### 3.2 核心组件

#### 3.2.1 RGB 分支编码器

```python
class RGBEncoder(nn.Module):
    """
    处理 RGB + TargetMask (4通道输入)
    """
    def __init__(self, config):
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # 修改第一层: 3 → 4 通道
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3)
        
        # 权重初始化:
        # - RGB 通道 (0-2): 复制预训练权重
        # - TargetMask 通道 (3): 初始化为 0
        
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        
    def forward(self, x):
        # 返回: features (B, 1024, H/16, W/16), skip_features
        ...
```

#### 3.2.2 深度分支编码器

```python
class DepthEncoder(nn.Module):
    """
    专门处理深度图 (1通道输入)
    支持多种分支类型
    """
    def __init__(self, config):
        if config.depth_branch_type == 'resnet50':
            # 完整 ResNet-50，1 通道输入
            self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
            # 权重初始化: 使用 RGB 通道权重均值
            self.conv1.weight = original_conv1.weight.mean(dim=1, keepdim=True)
            self.out_channels = 1024
            
        elif config.depth_branch_type == 'lite':
            # 轻量级编码器
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, 7, stride=2, padding=3),   # /2
                nn.Conv2d(32, 64, 3, stride=2, padding=1),  # /4
                nn.Conv2d(64, 128, 3, stride=2, padding=1), # /8
                nn.Conv2d(128, 256, 3, stride=2, padding=1),# /16
            )
            self.out_channels = 256
```

#### 3.2.3 特征融合模块

```python
class FeatureFusion(nn.Module):
    """
    融合 RGB 分支和深度分支的特征
    
    支持三种融合方式:
    1. concat: 通道拼接 + 1x1 卷积投影
    2. add: 投影到相同维度后相加
    3. attention: 通道注意力加权融合
    """
    def __init__(self, rgb_channels, depth_channels, out_channels, fusion_type='concat'):
        if fusion_type == 'concat':
            # 拼接后投影
            self.proj = nn.Sequential(
                nn.Conv2d(rgb_channels + depth_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            
        elif fusion_type == 'add':
            # 投影 + 相加
            self.rgb_proj = nn.Conv2d(rgb_channels, out_channels, 1)
            self.depth_proj = nn.Conv2d(depth_channels, out_channels, 1)
            
        elif fusion_type == 'attention':
            # 通道注意力
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(out_channels * 2, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels * 2),
                nn.Sigmoid(),
            )
```

### 3.3 模型配置

```python
@dataclass
class TransUNetTwoStreamConfig(TransUNetConfig):
    # 输入配置
    rgb_channels: int = 4    # RGB(3) + TargetMask(1)
    depth_channels: int = 1  # Depth only
    
    # 深度分支配置
    depth_branch_type: str = 'resnet50'  # 'resnet50', 'lite'
    depth_branch_hidden_dim: int = 256
    
    # 融合配置
    fusion_type: str = 'concat'  # 'concat', 'add', 'attention'
    fusion_dim: int = 1024       # 融合后特征维度
    
    # Transformer 配置 (继承自父类)
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
```

### 3.4 双流模型变体

#### 3.4.1 TwoStream-ClsOnly (仅分类)

```python
class TransUNetTwoStreamClsOnly(nn.Module):
    """
    双流网络 - 仅分类版本
    - 无分割头，适合纯分类任务
    - 参数量较少
    """
    def forward(self, x):
        # 分离通道
        rgb_mask = torch.cat([x[:, :3], x[:, 4:5]], dim=1)  # (B, 4, H, W)
        depth = x[:, 3:4]  # (B, 1, H, W)
        
        # 双流编码
        rgb_feat, _ = self.rgb_encoder(rgb_mask)
        depth_feat = self.depth_encoder(depth)
        
        # 特征融合
        fused_feat = self.fusion(rgb_feat, depth_feat)
        
        # Transformer + 分类
        ...
        return cls_logits  # (B, 1)
```

#### 3.4.2 TwoStream-SegFromEncoded (分类+分割)

```python
class TransUNetTwoStreamSegFromEncoded(nn.Module):
    """
    双流网络 - SegFromEncoded 版本
    - 分割头直接从 Transformer 输出解码
    - 不使用 skip connections
    - 适合端到端物理推理
    """
    def forward(self, x):
        ...
        # Task 1: Classification
        cls_logits = self.cls_head(encoded)
        
        # Task 2: Segmentation (无 skip connections)
        seg_mask, _ = self.seg_head(patch_tokens, feat_h, feat_w)
        
        return cls_logits, seg_mask  # (B, 1), (B, 1, H, W)
```

### 3.5 模型变体规模

| 变体 | Transformer 层数 | Hidden Dim | Heads | 参数量 (估计) |
|-----|-----------------|------------|-------|-------------|
| Base | 12 | 768 | 12 | ~140M |
| Small | 6 | 512 | 8 | ~90M |
| Tiny | 4 | 384 | 6 | ~70M |
| Micro | 2 | 192 | 3 | ~55M |

---

## 4. 单流 vs 双流对比

### 4.1 架构差异

| 特性 | 单流 RGBD | 双流 TwoStream |
|-----|----------|----------------|
| CNN Backbone | 1 × ResNet-50 | 2 × ResNet-50 |
| 输入处理 | 5通道直接拼接 | RGB(4) + Depth(1) 分离 |
| 融合位置 | 输入层 (早期融合) | CNN 特征层 (中期融合) |
| 融合方式 | 无 (直接拼接) | concat/add/attention |
| 参数量 | 较少 | 较多 (~1.5x) |
| 计算量 | 较少 | 较多 (~1.5x) |

### 4.2 理论优劣

**单流 RGBD 优点:**
- 结构简单，训练稳定
- 参数量和计算量较少
- 允许模型学习跨模态的早期交互

**双流 TwoStream 优点:**
- RGB 和 Depth 独立建模，避免特征干扰
- 深度分支可独立调整复杂度
- 支持更灵活的特征融合策略
- 理论上能更好地保留深度信息的独特性

### 4.3 数据流对比

```
单流 RGBD:
[RGB|Depth|Mask] → ResNet → Transformer → Heads
      5通道        1024ch    hidden_dim

双流 TwoStream:
[RGB|Mask] → RGB-ResNet ──┐
   4通道       1024ch      │
                          ├→ Fusion → Transformer → Heads
[Depth] → Depth-ResNet ───┘
  1通道      1024ch
```

---

## 5. 分割头设计

### 5.1 带 Skip Connections (标准)

```
Transformer 输出 (B, 1200, hidden_dim)
         │
         ▼ reshape
    (B, hidden_dim, 30, 40)  # H/16, W/16
         │
         ▼ Conv 1x1
    (B, 256, 30, 40)
         │
    ┌────┴────┐
    │ UpBlock │ ← skip: layer3 (1024 ch)
    └────┬────┘
    (B, 128, 60, 80)  # H/8
         │
    ┌────┴────┐
    │ UpBlock │ ← skip: layer2 (512 ch)
    └────┬────┘
    (B, 64, 120, 160)  # H/4
         │
    ┌────┴────┐
    │ UpBlock │ ← skip: layer1 (256 ch)
    └────┬────┘
    (B, 32, 240, 320)  # H/2
         │
    ┌────┴────┐
    │ UpBlock │ ← skip: conv1 (64 ch)
    └────┬────┘
    (B, 32, 480, 640)  # H
         │
         ▼ Final Conv
    (B, 1, 480, 640)
```

### 5.2 SegFromEncoded (无 Skip Connections)

```
Transformer 输出 (B, 1200, hidden_dim)
         │
         ▼ reshape
    (B, hidden_dim, 30, 40)
         │
         ▼ Conv 1x1
    (B, 256, 30, 40)
         │
    ┌────┴────┐
    │UpNoSkip│  # 纯上采样 + 卷积精化
    └────┬────┘
    (B, 128, 60, 80)
         │
    ┌────┴────┐
    │UpNoSkip│
    └────┬────┘
    (B, 64, 120, 160)
         │
    ┌────┴────┐
    │UpNoSkip│
    └────┬────┘
    (B, 32, 240, 320)
         │
    ┌────┴────┐
    │UpNoSkip│
    └────┬────┘
    (B, 32, 480, 640)
         │
         ▼ Final Conv
    (B, 1, 480, 640)
```

---

## 6. 代码使用示例

### 6.1 创建单流 RGBD 模型

```python
from train.models.transunet_rgbd import transunet_rgbd_tiny

# 创建模型
model = transunet_rgbd_tiny(pretrained=True)

# 前向传播
x = torch.randn(2, 5, 480, 640)  # B, C, H, W
cls_logits, seg_mask = model(x)

print(f"分类输出: {cls_logits.shape}")  # (2, 1)
print(f"分割输出: {seg_mask.shape}")    # (2, 1, 480, 640)
```

### 6.2 创建双流模型

```python
from train.models.transunet_twostream import (
    twostream_cls_only_tiny,
    twostream_seg_from_encoded_tiny
)

# 仅分类版本
model_cls = twostream_cls_only_tiny(
    pretrained=True,
    fusion_type='concat',
    depth_branch_type='resnet50'
)

x = torch.randn(2, 5, 480, 640)
cls_logits = model_cls(x)  # (2, 1)

# 分类+分割版本
model_seg = twostream_seg_from_encoded_tiny(pretrained=True)
cls_logits, seg_mask = model_seg(x)  # (2, 1), (2, 1, 480, 640)
```

### 6.3 自定义配置

```python
from train.models.transunet_twostream import (
    TransUNetTwoStreamConfig,
    TransUNetTwoStreamClsOnly
)

config = TransUNetTwoStreamConfig(
    # 输入配置
    img_height=480,
    img_width=640,
    rgb_channels=4,
    depth_channels=1,
    
    # 融合配置
    fusion_type='attention',  # 使用注意力融合
    fusion_dim=1024,
    
    # Transformer 配置
    hidden_dim=512,
    num_heads=8,
    num_layers=6,
    mlp_dim=2048,
    dropout=0.1,
)

model = TransUNetTwoStreamClsOnly(config)
```

---

## 7. 相关文件

```
train/models/
├── transunet.py                    # 基础 TransUNet (RGB 4通道)
├── transunet_rgbd.py               # 单流 RGBD 模型
├── transunet_twostream.py          # 双流模型
├── transunet_seg_from_encoded.py   # SegFromEncoded 分割头
├── transunet_rgbd_cls_only.py      # RGBD 仅分类版本
└── transunet_rgbd_fusion.py        # RGBD 分割感知融合版本
```

---

## 8. 总结

1. **单流 RGBD** 模型通过修改 ResNet-50 第一层卷积实现 5 通道输入，结构简单高效
2. **双流 TwoStream** 模型使用两个独立分支分别编码 RGB 和 Depth，在特征级别进行融合
3. 融合策略支持 concat、add、attention 三种方式
4. 分割任务支持带 Skip Connections 和 SegFromEncoded (无 Skip) 两种方式
5. 所有模型都提供 Base/Small/Tiny/Micro 等不同规模的变体
