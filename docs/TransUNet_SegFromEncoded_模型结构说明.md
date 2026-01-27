# TransUNet Seg-From-Encoded 模型结构详细说明

## 1. 模型概述

`TransUNet Seg-From-Encoded` 是 TransUNet 架构的一个变体，专门设计用于验证**纯 Transformer 特征对分割任务的表达能力**。与标准 TransUNet 的主要区别在于：

- **分割头直接从 Transformer Encoded 层解码**，不使用 CNN Encoder 的跳跃连接（skip connections）
- CNN Encoder 仅用于初始特征提取，不提供多尺度特征给分割头
- 实验目的：验证移除 CNN Encoder 多尺度特征后，纯 Transformer 特征对分割任务的表达能力

### 1.1 双任务设计

模型同时完成两个任务：

1. **Task 1: 稳定性二分类** (stable / unstable)
   - 使用全部 Transformer tokens（包括 CLS token）进行分类
   - 复用标准 TransUNet 的分类头实现

2. **Task 2: 受影响区域分割**
   - 直接从 Transformer encoded patch tokens 解码
   - **无 CNN skip connections**，纯粹基于 Transformer 特征进行上采样

### 1.2 支持的输入模式

- **RGB 模式**: 4 通道输入 (RGB 3 + Target Mask 1)
- **RGBD 模式**: 5 通道输入 (RGB 3 + Depth 1 + Target Mask 1)

---

## 2. 整体架构流程

```
输入图像 (B, C, H, W)
    ↓
[CNN Encoder (ResNet-50)]
    ↓
CNN 特征图 (B, 1024, H/16, W/16)
    ↓
[Patch Embedding]
    ↓
Patch Tokens (B, num_patches, hidden_dim)
    ↓
[添加 CLS Token + 位置编码]
    ↓
[Transformer Encoder] (N 层)
    ↓
Encoded Tokens (B, 1 + num_patches, hidden_dim)
    ↓
    ├─→ [Classification Head] → 分类 logits (B, 1)
    └─→ [Segmentation Head] → 分割掩码 (B, seg_classes, H, W)
```

### 2.1 关键设计点

1. **CNN Encoder 仅用于特征提取**
   - ResNet-50 提取到 `layer3`，输出 `(B, 1024, H/16, W/16)`
   - **不保存 skip features**，分割头无法使用 CNN 的多尺度特征

2. **Transformer 作为核心推理模块**
   - 通过多头自注意力机制进行全局特征交互
   - 编码后的 patch tokens 包含全局上下文信息

3. **分割头纯上采样设计**
   - 从 `(B, num_patches, hidden_dim)` reshape 回 2D
   - 通过 4 个上采样块逐步恢复到原始分辨率
   - **无跳跃连接**，完全依赖 Transformer 编码的特征

---

## 3. 核心组件详解

### 3.1 CNN Encoder (ResNet-50)

**位置**: `train/models/transunet.py` - `ResNetEncoder`

**功能**: 将输入图像编码为高维特征图

**结构**:
```
输入: (B, in_channels, H, W)
    ↓
conv1 + bn1 + relu + maxpool  → (B, 64, H/4, W/4)
    ↓
layer1 (ResNet blocks)         → (B, 256, H/4, W/4)
    ↓
layer2 (ResNet blocks)         → (B, 512, H/8, W/8)
    ↓
layer3 (ResNet blocks)         → (B, 1024, H/16, W/16)  [输出]
```

**关键实现**:
```python
# 修改第一层卷积以支持多通道输入
self.conv1 = nn.Conv2d(
    config.in_channels, 64,  # 支持 4 或 5 通道
    kernel_size=7, stride=2, padding=3, bias=False
)

# 初始化：RGB 通道复制预训练权重，额外通道初始化为 0
self.conv1.weight[:, :3, :, :] = original_conv1.weight
self.conv1.weight[:, 3:, :, :] = 0
```

**输出**:
- `cnn_features`: `(B, 1024, H/16, W/16)` - 用于后续 Transformer 编码
- `skip_features`: 在 `seg_from_encoded` 中**被忽略**（不返回给分割头）

---

### 3.2 Patch Embedding

**位置**: `train/models/transunet.py` - `PatchEmbedding`

**功能**: 将 2D CNN 特征图转换为 1D 序列，供 Transformer 处理

**实现**:
```python
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, patch_size: int = 1):
        self.proj = nn.Conv2d(in_channels, hidden_dim, 
                             kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        # x: (B, 1024, H/16, W/16)
        x = self.proj(x)  # (B, hidden_dim, H/16, W/16)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, hidden_dim)
        x = self.norm(x)
        return x, H, W
```

**输出**:
- `patch_embeds`: `(B, num_patches, hidden_dim)`，其中 `num_patches = (H/16) * (W/16)`
- `feat_h, feat_w`: 特征图的高度和宽度，用于后续 reshape

---

### 3.3 位置编码

**位置**: `train/models/transunet.py` - `PositionalEncoding`

**功能**: 为序列添加位置信息，使 Transformer 能够理解空间关系

**实现**:
```python
class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int, dropout: float = 0.1):
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        return self.dropout(x)
```

**特点**:
- 使用**可学习的位置编码**（而非固定的正弦/余弦编码）
- 支持最大序列长度 `max_patches + 1`（+1 为 CLS token）

---

### 3.4 CLS Token

**功能**: 用于分类任务的全局表示

**实现**:
```python
self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
nn.init.trunc_normal_(self.cls_token, std=0.02)
```

**使用**:
- 在序列开头添加：`tokens = torch.cat([cls_tokens, patch_embeds], dim=1)`
- 分类头使用全部 tokens（包括 CLS token）进行全局平均池化
- 分割头只使用 patch tokens（排除 CLS token）

---

### 3.5 Transformer Encoder

**位置**: `train/models/transunet.py` - `TransformerEncoder`

**功能**: 通过多头自注意力机制进行全局特征交互和编码

**结构**:
```
TransformerEncoder
    ↓
TransformerBlock × N 层
    ↓
每个 TransformerBlock:
    ├─ LayerNorm
    ├─ MultiHeadAttention (多头自注意力)
    ├─ LayerNorm
    └─ MLP (两层全连接 + GELU)
```

**MultiHeadAttention 实现**:
```python
class MultiHeadAttention(nn.Module):
    def forward(self, x):
        # x: (B, N, C)
        qkv = self.qkv(x).reshape(B, N, 3, num_heads, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)
```

**模型变体配置**:

| 变体 | Transformer 层数 | Hidden Dim | 注意力头数 | MLP Dim |
|------|-----------------|------------|-----------|---------|
| Micro | 2 | 192 | 3 | 768 |
| Tiny | 4 | 384 | 6 | 1536 |
| Small | 6 | 512 | 8 | 2048 |
| Base | 12 | 768 | 12 | 3072 |

**输出**:
- `encoded`: `(B, 1 + num_patches, hidden_dim)` - 包含 CLS token 和 patch tokens

---

### 3.6 Classification Head

**位置**: `train/models/transunet.py` - `ClassificationHead`

**功能**: 从 Transformer 编码特征中预测稳定性分类

**结构**:
```python
class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim, cls_hidden_dim, num_classes, dropout):
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, cls_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(cls_hidden_dim, cls_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(cls_hidden_dim // 2, num_classes),
        )
    
    def forward(self, x):
        # x: (B, seq_len, hidden_dim)
        x = x.mean(dim=1)  # 全局平均池化: (B, hidden_dim)
        return self.mlp(x)  # (B, num_classes)
```

**特点**:
- 使用**全局平均池化**聚合所有 tokens（包括 CLS token）
- 三层 MLP 进行最终分类
- 输出：`(B, 1)` - 二分类 logits

---

### 3.7 Segmentation Head (Seg-From-Encoded)

**位置**: `train/models/transunet_seg_from_encoded.py` - `SegmentationHeadFromEncoded`

**功能**: 直接从 Transformer encoded patch tokens 解码为分割掩码，**不使用 CNN skip connections**

#### 3.7.1 核心设计

**关键特点**:
- **无跳跃连接**: 不接收 CNN Encoder 的多尺度特征
- **纯上采样**: 通过 4 个上采样块逐步恢复到原始分辨率
- **依赖 Transformer 特征**: 完全基于 Transformer 编码的全局上下文信息

#### 3.7.2 结构流程

```
输入: patch_tokens (B, num_patches, hidden_dim)
    ↓
[Reshape] → (B, hidden_dim, H/16, W/16)
    ↓
[reshape_proj] 1×1 Conv → (B, 256, H/16, W/16)
    ↓
[UpBlock 1] 2× 上采样 → (B, 128, H/8, W/8)
    ↓
[UpBlock 2] 2× 上采样 → (B, 64, H/4, W/4)
    ↓
[UpBlock 3] 2× 上采样 → (B, 32, H/2, W/2)
    ↓
[UpBlock 4] 2× 上采样 → (B, 32, H, W)
    ↓
[final_conv] → (B, seg_classes, H, W)
```

#### 3.7.3 UpBlockNoSkip 实现

**位置**: `train/models/transunet_seg_from_encoded.py` - `UpBlockNoSkip`

**功能**: 上采样块，不使用跳跃连接

```python
class UpBlockNoSkip(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, use_bn=True):
        self.scale_factor = scale_factor
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, 
                         mode='bilinear', align_corners=False)
        return self.conv(x)
```

**特点**:
- 使用**双线性插值**进行上采样
- 两个 3×3 卷积 + BatchNorm + ReLU 进行特征精化
- **无跳跃连接输入**

#### 3.7.4 完整实现

```python
class SegmentationHeadFromEncoded(nn.Module):
    def __init__(self, config):
        # 从 Transformer 特征恢复到 2D
        self.reshape_proj = nn.Conv2d(
            config.hidden_dim, 
            decoder_channels[0],  # 默认 256
            1
        )
        
        # 4 个上采样块
        self.up_blocks = nn.ModuleList([
            UpBlockNoSkip(256, 128, use_bn=config.seg_decoder_use_bn),  # H/16 → H/8
            UpBlockNoSkip(128, 64, use_bn=config.seg_decoder_use_bn),   # H/8 → H/4
            UpBlockNoSkip(64, 32, use_bn=config.seg_decoder_use_bn),   # H/4 → H/2
            UpBlockNoSkip(32, 32, use_bn=config.seg_decoder_use_bn),   # H/2 → H
        ])
        
        # 最终卷积
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, config.seg_classes, 1),
        )
    
    def forward(self, x, feat_h, feat_w):
        # x: (B, num_patches, hidden_dim)
        B = x.shape[0]
        
        # Reshape 回 2D
        x = x.transpose(1, 2).reshape(B, -1, feat_h, feat_w)
        x = self.reshape_proj(x)  # (B, 256, H/16, W/16)
        
        # 上采样路径
        for up_block in self.up_blocks:
            x = up_block(x)
        
        # 最终输出
        seg_mask = self.final_conv(x)  # (B, seg_classes, H, W)
        return seg_mask
```

**配置参数**:
- `seg_decoder_channels`: 默认 `(256, 128, 64, 32)` - 各上采样块的通道数
- `seg_decoder_use_bn`: 默认 `True` - 是否使用 BatchNorm

---

## 4. 完整前向传播流程

### 4.1 输入处理

```python
def forward(self, x, return_attention=False):
    # x: (B, C, H, W)
    # C = 4 (RGB + Target Mask) 或 5 (RGBD + Target Mask)
    B, C, H, W = x.shape
```

### 4.2 CNN 编码

```python
# CNN Encoder (忽略 skip_features)
cnn_features, _ = self.cnn_encoder(x)
# cnn_features: (B, 1024, H/16, W/16)
```

### 4.3 Patch Embedding

```python
# 转换为序列
patch_embeds, feat_h, feat_w = self.patch_embed(cnn_features)
# patch_embeds: (B, num_patches, hidden_dim)
# num_patches = (H/16) * (W/16)
```

### 4.4 Transformer 编码

```python
# 添加 CLS Token
cls_tokens = self.cls_token.expand(B, -1, -1)
tokens = torch.cat([cls_tokens, patch_embeds], dim=1)
# tokens: (B, 1 + num_patches, hidden_dim)

# 位置编码
tokens = self.pos_encoding(tokens)

# Transformer Encoder
encoded = self.transformer(tokens)
# encoded: (B, 1 + num_patches, hidden_dim)
```

### 4.5 双任务输出

```python
# 分离 CLS Token 和 patch tokens
patch_tokens = encoded[:, 1:, :]  # (B, num_patches, hidden_dim)

# Task 1: Classification
cls_logits = self.cls_head(encoded)  # (B, 1)

# Task 2: Segmentation
seg_mask, _ = self.seg_head(patch_tokens, feat_h, feat_w)
# seg_mask: (B, seg_classes, H, W)

# 调整到原始输入尺寸（如果需要）
if seg_mask.shape[2:] != (H, W):
    seg_mask = F.interpolate(seg_mask, size=(H, W), 
                            mode='bilinear', align_corners=False)

return cls_logits, seg_mask
```

---

## 5. 与标准 TransUNet 的对比

### 5.1 架构差异

| 特性 | 标准 TransUNet | Seg-From-Encoded |
|------|---------------|-----------------|
| **分割头输入** | Transformer tokens + CNN skip features | 仅 Transformer tokens |
| **跳跃连接** | ✅ 使用 (4 个层级) | ❌ 不使用 |
| **上采样方式** | UpBlock (带 skip) | UpBlockNoSkip (纯上采样) |
| **特征来源** | Transformer + CNN 多尺度 | 纯 Transformer |
| **实验目的** | 标准双任务模型 | 验证 Transformer 特征表达能力 |

### 5.2 代码对比

**标准 TransUNet 分割头**:
```python
# 使用 skip_features
seg_mask = self.seg_head(patch_tokens, feat_h, feat_w, skip_features)
```

**Seg-From-Encoded 分割头**:
```python
# 不使用 skip_features
cnn_features, _ = self.cnn_encoder(x)  # 忽略 skip_features
seg_mask, _ = self.seg_head(patch_tokens, feat_h, feat_w)  # 无 skip 参数
```

---

## 6. 模型配置

### 6.1 配置类

**位置**: `train/models/transunet_seg_from_encoded.py` - `TransUNetSegFromEncodedConfig`

```python
@dataclass
class TransUNetSegFromEncodedConfig(TransUNetConfig):
    # 分割解码器配置
    seg_decoder_channels: Tuple[int, ...] = (256, 128, 64, 32)
    seg_decoder_use_bn: bool = True
```

**继承的配置** (来自 `TransUNetConfig`):
- `img_height`, `img_width`: 输入图像尺寸
- `in_channels`: 输入通道数 (4 或 5)
- `pretrained`: 是否使用预训练 ResNet
- `hidden_dim`, `num_heads`, `num_layers`, `mlp_dim`: Transformer 配置
- `dropout`: Dropout 率
- `num_classes`: 分类类别数 (默认 1，二分类)
- `seg_classes`: 分割类别数 (默认 1)

### 6.2 模型变体

**位置**: `train/models/transunet_seg_from_encoded.py`

```python
# Micro: 2 层, 192 hidden dim
transunet_seg_from_encoded_micro()

# Tiny: 4 层, 384 hidden dim
transunet_seg_from_encoded_tiny()

# Small: 6 层, 512 hidden dim
transunet_seg_from_encoded_small()

# Base: 12 层, 768 hidden dim
transunet_seg_from_encoded_base()
```

---

## 7. 实验设计意义

### 7.1 研究问题

**核心问题**: 移除 CNN Encoder 多尺度特征后，纯 Transformer 特征对分割任务的表达能力如何？

### 7.2 设计动机

1. **验证 Transformer 的全局建模能力**
   - Transformer 通过自注意力机制捕获全局上下文
   - 验证这种全局特征是否足以支持分割任务

2. **简化架构**
   - 移除跳跃连接可以减少模型复杂度
   - 降低内存占用（不需要保存多尺度 CNN 特征）

3. **消融实验**
   - 对比标准 TransUNet（有 skip）和 Seg-From-Encoded（无 skip）
   - 评估 CNN 多尺度特征对分割任务的重要性

### 7.3 预期结果

- **如果性能接近**: 说明 Transformer 特征已经包含足够的信息，CNN skip connections 不是必需的
- **如果性能下降**: 说明 CNN 多尺度特征对分割任务很重要，跳跃连接是必要的

---

## 8. 使用示例

### 8.1 创建模型

```python
from train.models.transunet_seg_from_encoded import (
    transunet_seg_from_encoded_tiny,
    TransUNetSegFromEncodedConfig
)

# 方式 1: 使用预定义变体
model = transunet_seg_from_encoded_tiny(
    pretrained=True,
    img_height=480,
    img_width=640,
    in_channels=5  # RGBD
)

# 方式 2: 自定义配置
config = TransUNetSegFromEncodedConfig(
    img_height=480,
    img_width=640,
    in_channels=5,
    num_layers=4,
    hidden_dim=384,
    num_heads=6,
    mlp_dim=1536,
    seg_decoder_channels=(256, 128, 64, 32),
    seg_decoder_use_bn=True
)
model = TransUNetSegFromEncoded(config)
```

### 8.2 前向传播

```python
# 输入: RGBD + Target Mask
x = torch.randn(2, 5, 480, 640)  # (B, C, H, W)

# 前向传播
cls_logits, seg_mask = model(x)

# 输出
print(f"分类输出: {cls_logits.shape}")  # (2, 1)
print(f"分割输出: {seg_mask.shape}")    # (2, 1, 480, 640)
```

### 8.3 训练脚本

**RGBD 版本**: `train/train_rgbd_seg_from_encoded.py`

```bash
python train/train_rgbd_seg_from_encoded.py \
    --config small \
    --model tiny \
    --batch_size 40 \
    --epochs 150 \
    --lr 1e-4
```

---

## 9. 关键代码位置

| 组件 | 文件路径 | 类名 |
|------|---------|------|
| 主模型 | `train/models/transunet_seg_from_encoded.py` | `TransUNetSegFromEncoded` |
| 分割头 | `train/models/transunet_seg_from_encoded.py` | `SegmentationHeadFromEncoded` |
| 上采样块 | `train/models/transunet_seg_from_encoded.py` | `UpBlockNoSkip` |
| CNN Encoder | `train/models/transunet.py` | `ResNetEncoder` |
| Transformer | `train/models/transunet.py` | `TransformerEncoder` |
| 分类头 | `train/models/transunet.py` | `ClassificationHead` |
| 训练脚本 | `train/train_rgbd_seg_from_encoded.py` | - |

---

## 10. 总结

`TransUNet Seg-From-Encoded` 是一个专门设计的实验性架构，用于验证纯 Transformer 特征对分割任务的表达能力。其核心特点是：

1. **移除 CNN skip connections**: 分割头不接收 CNN Encoder 的多尺度特征
2. **纯 Transformer 解码**: 完全基于 Transformer encoded tokens 进行上采样
3. **双任务设计**: 同时完成分类和分割任务
4. **简化架构**: 降低模型复杂度，便于分析和对比实验

通过对比标准 TransUNet 和 Seg-From-Encoded 的性能，可以评估 CNN 多尺度特征对分割任务的重要性，以及 Transformer 全局建模能力的有效性。
