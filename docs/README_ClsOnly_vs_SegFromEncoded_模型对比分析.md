# TransUNet ClsOnly vs SegFromEncoded 模型结构对比分析

本文档详细分析 `TransUNetClsOnly` 和 `TransUNetSegFromEncoded` 两种模型架构的差异。

---

## 1. 概述

| 特性 | ClsOnly | SegFromEncoded |
|------|---------|----------------|
| **实验目的** | 验证分割任务是否对分类任务起到正则化/特征增强作用 | 验证移除CNN多尺度特征后，纯Transformer特征对分割的表达能力 |
| **任务类型** | 单任务（仅分类） | 双任务（分类 + 分割） |
| **输出** | `cls_logits` | `cls_logits`, `seg_mask` |
| **Skip Connections** | ❌ 无 | ❌ 无 |
| **分割头** | ❌ 无 | ✅ 有（纯Transformer特征解码） |

---

## 2. 架构流程图

### 2.1 TransUNetClsOnly 架构

```
输入 (B, 4, H, W)
      │
      ▼
┌─────────────────┐
│  ResNetEncoder  │  (仅特征提取，skip_features 被丢弃)
│   (ResNet-50)   │
└────────┬────────┘
         │ cnn_features (B, 1024, H/16, W/16)
         ▼
┌─────────────────┐
│  PatchEmbedding │
└────────┬────────┘
         │ patch_embeds (B, num_patches, hidden_dim)
         ▼
┌─────────────────┐
│  + CLS Token    │
└────────┬────────┘
         │ tokens (B, 1 + num_patches, hidden_dim)
         ▼
┌─────────────────┐
│ PositionalEnc   │
└────────┬────────┘
         ▼
┌─────────────────┐
│ TransformerEnc  │
└────────┬────────┘
         │ encoded (B, 1 + num_patches, hidden_dim)
         ▼
┌─────────────────┐
│ClassificationHead│
└────────┬────────┘
         │
         ▼
  cls_logits (B, 1)   ← 唯一输出
```

### 2.2 TransUNetSegFromEncoded 架构

```
输入 (B, 4, H, W)
      │
      ▼
┌─────────────────┐
│  ResNetEncoder  │  (仅特征提取，skip_features 被丢弃)
│   (ResNet-50)   │
└────────┬────────┘
         │ cnn_features (B, 1024, H/16, W/16)
         ▼
┌─────────────────┐
│  PatchEmbedding │
└────────┬────────┘
         │ patch_embeds (B, num_patches, hidden_dim)
         ▼
┌─────────────────┐
│  + CLS Token    │
└────────┬────────┘
         │ tokens (B, 1 + num_patches, hidden_dim)
         ▼
┌─────────────────┐
│ PositionalEnc   │
└────────┬────────┘
         ▼
┌─────────────────┐
│ TransformerEnc  │
└────────┬────────┘
         │ encoded (B, 1 + num_patches, hidden_dim)
         │
    ┌────┴────────────────────┐
    │                         │
    ▼                         ▼
┌──────────────┐    ┌─────────────────────────┐
│ClassificationHead│ │SegmentationHeadFromEncoded│
└───────┬──────┘    └───────────┬─────────────┘
        │                       │
        ▼                       ▼
 cls_logits (B, 1)      seg_mask (B, 1, H, W)
```

---

## 3. 核心组件对比

### 3.1 共同组件

两个模型共享以下核心组件：

| 组件 | 描述 | 输入/输出 |
|------|------|----------|
| `ResNetEncoder` | ResNet-50 编码器（前3个Stage） | `(B, 4, H, W)` → `(B, 1024, H/16, W/16)` |
| `PatchEmbedding` | CNN特征转为Transformer序列 | `(B, 1024, H/16, W/16)` → `(B, num_patches, hidden_dim)` |
| `PositionalEncoding` | 可学习的位置编码 | 添加位置信息 |
| `TransformerEncoder` | Transformer编码器块 | 序列建模 |
| `ClassificationHead` | 分类头（MLP） | `(B, seq_len, hidden_dim)` → `(B, 1)` |

### 3.2 差异组件

| 组件 | ClsOnly | SegFromEncoded |
|------|---------|----------------|
| `SegmentationHead` | ❌ | ❌ (标准版本) |
| `SegmentationHeadFromEncoded` | ❌ | ✅ |
| `UpBlockNoSkip` | ❌ | ✅ (4个) |
| Skip Connections 使用 | ❌ | ❌ |

---

## 4. 分割头实现差异

### 4.1 标准 TransUNet SegmentationHead（作为参考）

标准TransUNet使用带Skip Connections的分割头：

```python
class SegmentationHead(nn.Module):
    def forward(self, x, feat_h, feat_w, skip_features):
        # skip_features: [layer3, layer2, layer1, conv1]
        for i, up_block in enumerate(self.up_blocks):
            skip = skip_features[i]  # 使用CNN编码器的skip连接
            x = up_block(x, skip)    # UpBlock 需要 skip 输入
        return self.final_conv(x)
```

### 4.2 SegFromEncoded 的 SegmentationHeadFromEncoded

无Skip Connections，纯粹从Transformer特征解码：

```python
class SegmentationHeadFromEncoded(nn.Module):
    def forward(self, x, feat_h, feat_w, return_intermediate=False):
        # x: Transformer输出的 patch tokens (B, num_patches, hidden_dim)
        
        # 1. Reshape 到 2D 空间
        x = x.transpose(1, 2).reshape(B, -1, feat_h, feat_w)
        x = self.reshape_proj(x)  # 投影到decoder通道
        
        # 2. 连续上采样 (无skip connection)
        for up_block in self.up_blocks:
            x = up_block(x)  # UpBlockNoSkip: 纯粹双线性上采样 + 卷积
            
        return self.final_conv(x)
```

### 4.3 UpBlock 对比

| 特性 | 标准 UpBlock | UpBlockNoSkip |
|------|-------------|---------------|
| 上采样方式 | 双线性插值 | 双线性插值 |
| Skip Connection | ✅ 拼接CNN特征 | ❌ 无 |
| 输入通道计算 | `in_channels + skip_channels` | `in_channels` |
| 卷积结构 | Conv-BN-ReLU-Conv-BN-ReLU | Conv-[BN]-ReLU-Conv-[BN]-ReLU |

---

## 5. 配置类对比

### 5.1 TransUNetClsOnlyConfig

```python
@dataclass
class TransUNetClsOnlyConfig(TransUNetConfig):
    """继承所有配置，无额外参数"""
    pass
```

### 5.2 TransUNetSegFromEncodedConfig

```python
@dataclass
class TransUNetSegFromEncodedConfig(TransUNetConfig):
    """额外的分割解码器配置"""
    seg_decoder_channels: Tuple[int, ...] = (256, 128, 64, 32)  # 解码器通道数
    seg_decoder_use_bn: bool = True  # 是否使用 BatchNorm
```

---

## 6. Forward 函数对比

### 6.1 TransUNetClsOnly.forward()

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # 1. CNN 编码（丢弃 skip_features）
    cnn_features, _ = self.cnn_encoder(x)
    
    # 2. Patch Embedding
    patch_embeds, feat_h, feat_w = self.patch_embed(cnn_features)
    
    # 3. 添加 CLS Token + 位置编码
    tokens = torch.cat([cls_tokens, patch_embeds], dim=1)
    tokens = self.pos_encoding(tokens)
    
    # 4. Transformer 编码
    encoded = self.transformer(tokens)
    
    # 5. 仅分类输出
    cls_logits = self.cls_head(encoded)
    
    return cls_logits  # 单一输出
```

### 6.2 TransUNetSegFromEncoded.forward()

```python
def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    # 1. CNN 编码（丢弃 skip_features）
    cnn_features, _ = self.cnn_encoder(x)
    
    # 2. Patch Embedding
    patch_embeds, feat_h, feat_w = self.patch_embed(cnn_features)
    
    # 3. 添加 CLS Token + 位置编码
    tokens = torch.cat([cls_tokens, patch_embeds], dim=1)
    tokens = self.pos_encoding(tokens)
    
    # 4. Transformer 编码
    encoded = self.transformer(tokens)
    
    # 5. 分离 CLS Token 和 Patch Tokens
    patch_tokens = encoded[:, 1:, :]  # 用于分割
    
    # 6. 双任务输出
    cls_logits = self.cls_head(encoded)      # 分类（使用全部tokens）
    seg_mask, _ = self.seg_head(patch_tokens, feat_h, feat_w)  # 分割（使用patch tokens）
    
    return cls_logits, seg_mask  # 双输出
```

---

## 7. 参数量对比（估算）

以 **Tiny 变体** 为例（4层Transformer, hidden_dim=384）：

| 组件 | ClsOnly | SegFromEncoded |
|------|---------|----------------|
| ResNetEncoder | ~23.5M | ~23.5M |
| PatchEmbedding | ~0.4M | ~0.4M |
| PositionalEncoding | ~0.5M | ~0.5M |
| TransformerEncoder (4层) | ~7.1M | ~7.1M |
| ClassificationHead | ~0.05M | ~0.05M |
| SegmentationHeadFromEncoded | ❌ | ~0.3M |
| **总计** | **~31.5M** | **~31.8M** |

> 注：由于 SegFromEncoded 没有使用 skip_channels，分割头参数量较小

---

## 8. 实验设计目的

### 8.1 ClsOnly 消融实验

**假设**: 多任务学习（分类+分割）可能对分类任务有正则化效果

**实验对比**:
- **Baseline**: 标准 TransUNet（分类+分割）
- **ClsOnly**: 仅分类任务

**预期结果**:
- 如果 ClsOnly 性能下降 → 分割任务对分类有辅助作用
- 如果 ClsOnly 性能相当/提升 → 分割任务不必要或引入了噪声

### 8.2 SegFromEncoded 消融实验

**假设**: CNN Encoder 的多尺度 skip connections 对分割质量至关重要

**实验对比**:
- **Baseline**: 标准 TransUNet（带 skip connections）
- **SegFromEncoded**: 无 skip connections

**预期结果**:
- 如果 SegFromEncoded 分割性能大幅下降 → skip connections 对分割很重要
- 如果 SegFromEncoded 分割性能接近 → Transformer 特征自身足以重建细节

---

## 9. 代码文件结构

```
train/models/
├── transunet.py               # 基础 TransUNet（带 skip connections）
├── transunet_cls_only.py      # ClsOnly 变体
├── transunet_seg_from_encoded.py  # SegFromEncoded 变体
├── transunet_rgbd.py          # RGBD 版本（5通道输入）
├── transunet_rgbd_cls_only.py # RGBD + ClsOnly
└── __init__.py
```

---

## 10. 使用示例

### 10.1 ClsOnly

```python
from train.models import transunet_cls_only_tiny

model = transunet_cls_only_tiny(pretrained=True)
x = torch.randn(2, 4, 480, 640)

cls_logits = model(x)  # 返回 (B, 1)
```

### 10.2 SegFromEncoded

```python
from train.models import transunet_seg_from_encoded_tiny

model = transunet_seg_from_encoded_tiny(pretrained=True)
x = torch.randn(2, 4, 480, 640)

cls_logits, seg_mask = model(x)  # 返回 (B, 1), (B, 1, H, W)
```

---

## 11. 总结

| 维度 | ClsOnly | SegFromEncoded |
|------|---------|----------------|
| **核心差异** | 移除分割任务 | 移除 skip connections |
| **实验变量** | 多任务 vs 单任务 | 多尺度特征 vs 纯Transformer特征 |
| **架构复杂度** | 更简单 | 略复杂（多一个分割头） |
| **训练监督** | 仅分类Loss | 分类Loss + 分割Loss |
| **输出维度** | 1D logits | 1D logits + 2D mask |

这两个模型的设计目的是进行**消融实验 (Ablation Study)**，帮助理解多任务学习和skip connections在物理稳定性预测任务中的作用。
