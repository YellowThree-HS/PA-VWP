# TransUNet 物理稳定性预测训练模块

基于 Attention 的 TransUNet 架构，用于预测移除箱子后场景的物理稳定性。

## 模型版本总览

本项目提供多个版本的模型，按输入模态和任务设计分类：

| 版本 | 训练脚本 | 输入通道 | 任务 | 特点 |
|------|----------|----------|------|------|
| **RGB** | `train.py` | 4 (RGB+Mask) | 分类+分割 | 基础双任务模型 |
| **RGBD** | `train_with_depth.py` | 5 (RGB+Depth+Mask) | 分类+分割 | 引入深度信息 |
| **RGB Fusion** | `train_fusion.py` | 4 (RGB+Mask) | 分类+分割 | 分割感知融合 |
| **RGBD Fusion** | `train_rgbd_fusion.py` | 5 (RGB+Depth+Mask) | 分类+分割 | 分割感知融合+深度 |
| **RGB Cls-Only** | `train_cls_only.py` | 4 (RGB+Mask) | 仅分类 | 轻量级分类 |
| **RGBD Cls-Only** | `train_rgbd_cls_only.py` | 5 (RGB+Depth+Mask) | 仅分类 | 轻量级分类+深度 |

## 架构设计

### 设计理念

物理稳定性预测任务的核心挑战是：
1. **全局推理**：需要理解整个场景的物理结构关系
2. **局部定位**：需要精确预测哪些区域会受影响
3. **任务关联**：分类（是否稳定）和分割（影响区域）任务高度相关

我们的架构设计针对这些挑战：

```
                    ┌─────────────────────────────────────────┐
                    │            TransUNet 架构族              │
                    └─────────────────────────────────────────┘
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          │                            │                            │
    ┌─────▼─────┐              ┌───────▼───────┐            ┌───────▼───────┐
    │  Base     │              │   Fusion      │            │   Cls-Only    │
    │ (双任务)  │              │ (分割感知融合) │            │  (仅分类)     │
    └─────┬─────┘              └───────┬───────┘            └───────┬───────┘
          │                            │                            │
    ┌─────┴─────┐              ┌───────┴───────┐            ┌───────┴───────┐
    │ RGB/RGBD  │              │  RGB/RGBD     │            │  RGB/RGBD     │
    └───────────┘              └───────────────┘            └───────────────┘
```

---

### 1. Base 模型 (TransUNet / TransUNetRGBD)

基础双任务模型，分类和分割独立解码。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TransUNet Base Architecture                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  输入: RGB (3) + Target Mask (1) = 4 通道  [RGBD 版本: +Depth = 5 通道]       │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     CNN Encoder (ResNet-50)                         │    │
│  │                                                                     │    │
│  │  Input ──► Conv1 ──► Layer1 ──► Layer2 ──► Layer3 ──► Features     │    │
│  │  (4ch)    (64ch)    (256ch)    (512ch)    (1024ch)    (1024ch)     │    │
│  │            /2        /4         /8         /16         /16         │    │
│  │             │          │          │          │                      │    │
│  │             └──────────┴──────────┴──────────┼─── Skip Connections  │    │
│  └─────────────────────────────────────────────┼───────────────────────┘    │
│                                                │                             │
│  ┌─────────────────────────────────────────────▼───────────────────────┐    │
│  │                   Transformer Bottleneck                            │    │
│  │                                                                     │    │
│  │  ┌─────────────┐   ┌─────────────────────────────────────────────┐ │    │
│  │  │ Patch Embed │──►│  N × Transformer Layers (Self-Attention)    │ │    │
│  │  │ + Pos Enc   │   │  - 全局物理关联建模                          │ │    │
│  │  │ + CLS Token │   │  - 跨区域依赖推理                            │ │    │
│  │  └─────────────┘   └─────────────────────────────────────────────┘ │    │
│  └────────────────────────────────┬────────────────────────────────────┘    │
│                                   │                                          │
│                    ┌──────────────┴──────────────┐                          │
│                    │                             │                          │
│  ┌─────────────────▼────────────┐  ┌─────────────▼──────────────────────┐   │
│  │    Classification Head       │  │       Segmentation Head            │   │
│  │                              │  │                                    │   │
│  │  CLS Token ──► MLP ──► σ    │  │  Patches ──► UpBlocks ──► Mask    │   │
│  │                              │  │         (+ Skip Connections)       │   │
│  │  输出: 稳定/不稳定 (0/1)     │  │  输出: 受影响区域 (H×W)            │   │
│  └──────────────────────────────┘  └────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**特点**：
- 分类头使用 CLS Token 的全局特征
- 分割头使用 UNet 风格的跳跃连接恢复空间细节
- 两个任务独立解码，互不干扰

---

### 2. Fusion 模型 (TransUNetFusion / TransUNetRGBDFusion) ⭐推荐

**核心创新**：分割感知双路融合（Segmentation-Aware Dual-Path Fusion）

分类和分割任务高度相关：如果能预测出受影响区域，就能更好地判断稳定性。Fusion 模型通过三种融合机制将分割信息注入分类决策：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       TransUNet Fusion Architecture                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Shared Encoder (ResNet-50 + Transformer)                           │    │
│  │  - 与 Base 模型相同                                                  │    │
│  │  - 输出: CLS Token + Patch Tokens + Skip Features                   │    │
│  └───────────────────────────────────┬─────────────────────────────────┘    │
│                                      │                                       │
│                    ┌─────────────────┴─────────────────┐                    │
│                    │                                   │                    │
│                    ▼                                   ▼                    │
│  ┌────────────────────────────────┐  ┌────────────────────────────────┐    │
│  │   Segmentation Head (先执行)   │  │        CLS Token               │    │
│  │                                │  │     (全局语义特征)              │    │
│  │   生成:                        │  │                                │    │
│  │   1. 分割掩码 (H×W)            │  │   f_cls ∈ R^768               │    │
│  │   2. 中间特征 (多尺度)         │  │                                │    │
│  └─────────────┬──────────────────┘  └────────────────┬───────────────┘    │
│                │                                      │                     │
│      ┌─────────┴─────────┐                            │                     │
│      │                   │                            │                     │
│      ▼                   ▼                            ▼                     │
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────────────────────┐     │
│  │ Mask-Guided   │  │ Multi-Scale   │  │                             │     │
│  │ Attention     │  │ Feature       │  │        Feature Fusion       │     │
│  │ Pooling       │  │ Pooling       │  │                             │     │
│  │               │  │               │  │  Concat ──► MLP ──► σ      │     │
│  │ 用分割掩码    │  │ 从解码器提取   │  │                             │     │
│  │ 加权聚合      │  │ 多尺度特征     │  │  融合维度:                   │     │
│  │ Patch Tokens  │  │ 并池化         │  │  768 + 768 + 224 = 1760    │     │
│  │               │  │               │  │                             │     │
│  │ f_seg ∈ R^768│  │ f_dec ∈ R^224 │  │                             │     │
│  └───────┬───────┘  └───────┬───────┘  └──────────────┬──────────────┘     │
│          │                  │                         │                     │
│          └──────────────────┴─────────────────────────┘                     │
│                                      │                                       │
│                                      ▼                                       │
│                        ┌─────────────────────────┐                          │
│                        │   Classification Output  │                          │
│                        │   稳定/不稳定 (0/1)       │                          │
│                        └─────────────────────────┘                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 融合策略详解

**策略 1: CLS Token 全局语义 (f_cls)**
```
Transformer 的 CLS Token 捕获全局场景语义
f_cls = Transformer_Output[:, 0, :]  # (B, 768)
```

**策略 2: Mask-Guided Attention Pooling (f_seg)**
```
使用预测的分割掩码作为空间注意力权重，引导 Patch Tokens 的聚合

    分割掩码                     Patch Tokens
    (B, 1, H, W)                (B, num_patches, 768)
         │                            │
         ▼                            │
    下采样到 (B, 1, h, w)             │
         │                            │
         ▼                            ▼
    Attention Weights ─────────► 加权求和
    (B, num_patches)                  │
                                      ▼
                              f_seg ∈ R^768

直觉: 让模型关注"预测会受影响的区域"的特征
```

**策略 3: Multi-Scale Decoder Features (f_dec)**
```
从分割解码器的各阶段提取特征，捕获不同尺度的语义信息

    UpBlock 1 输出 ──► GlobalAvgPool ──► (B, 128)  ─┐
    (H/8 × W/8)                                      │
                                                     │
    UpBlock 2 输出 ──► GlobalAvgPool ──► (B, 64)   ─┼──► Concat ──► f_dec ∈ R^224
    (H/4 × W/4)                                      │
                                                     │
    UpBlock 3 输出 ──► GlobalAvgPool ──► (B, 32)   ─┘
    (H/2 × W/2)

直觉: 分割解码器已经学习了"哪些区域会受影响"的知识，复用这些特征
```

**最终融合**：
```python
fused = Concat([f_cls, f_seg, f_dec])  # (B, 1760)
logits = MLP(fused)                     # (B, 1)
```

---

### 3. Cls-Only 模型 (TransUNetClsOnly / TransUNetRGBDClsOnly)

仅做稳定性分类，不进行分割。适用于只需要判断稳定性的场景。

```
┌─────────────────────────────────────────────────────────────────┐
│                    Cls-Only Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input ──► ResNet-50 ──► Transformer ──► CLS Token ──► MLP ──► σ │
│                                                                  │
│  无分割头，参数量更少，推理更快                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 模型变体参数

### RGB 版本

| 变体 | Transformer 层数 | Hidden Dim | MLP Dim | Heads | 参数量 |
|------|------------------|------------|---------|-------|--------|
| Tiny | 4 | 384 | 1536 | 6 | ~30M |
| Small | 6 | 512 | 2048 | 8 | ~50M |
| Base | 12 | 768 | 3072 | 12 | ~100M |

### RGBD 版本

与 RGB 版本参数相同，仅第一层卷积从 4 通道扩展到 5 通道：
- RGB 通道 (0-2): 使用 ImageNet 预训练权重
- Depth 通道 (3): 使用 RGB 通道权重的平均值初始化
- Target Mask 通道 (4): 初始化为 0

---

## 快速开始

### 1. 安装依赖

```bash
cd train
pip install -r requirements.txt
```

### 2. 开始训练

#### Base 模型

```bash
# RGB 版本
python train.py --config base --epochs 150

# RGBD 版本
python train_with_depth.py --config base --epochs 150
```

#### Fusion 模型 (推荐)

```bash
# RGB Fusion
python train_fusion.py --config base --epochs 150

# RGBD Fusion
python train_rgbd_fusion.py --config base --epochs 150

# 使用 H100 训练脚本 (推荐)
bash run_train_rgbd_fusion_h100.sh
```

#### Cls-Only 模型

```bash
# RGB 仅分类
python train_cls_only.py --config base --epochs 150

# RGBD 仅分类
python train_rgbd_cls_only.py --config base --epochs 150
```

### 3. 评估模型

```bash
python evaluate.py --checkpoint checkpoints/transunet_fusion_base/checkpoint_best.pth --visualize
```

### 4. 单图推理

```bash
python inference.py \
    --checkpoint best.pth \
    --image path/to/rgb.png \
    --mask path/to/removed_mask.png
```

---

## 配置预设

| 预设 | 模型 | 批次 | Epochs | 说明 |
|------|------|------|--------|------|
| `debug` | Tiny | 2 | 5 | 快速调试 |
| `default` | Small | 8 | 100 | 默认训练 |
| `small` | Small | 8 | 100 | 小模型 |
| `base` | Base | 4 | 150 | 完整模型 |
| `full_data` | Small | 8 | 100 | 使用所有数据 |

---

## 命令行参数

```bash
python train_fusion.py --help
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置预设 | `default` |
| `--exp_name` | 实验名称 | 自动生成 |
| `--data_dirs` | 数据目录 | `dataset_messy_small` |
| `--batch_size` | 批次大小 | 8 |
| `--epochs` | 训练轮数 | 100 |
| `--lr` | 学习率 | 1e-4 |
| `--model` | 模型变体 | `small` |
| `--device` | 设备 | `cuda` |
| `--wandb` | 启用 WandB | False |
| `--no_amp` | 禁用混合精度 | False |

---

## 数据格式

### RGB 版本数据格式

```
dataset/
├── round_001_65_top/
│   ├── rgb.png                 # RGB 图像 (640x480)
│   └── removals/
│       ├── 0/
│       │   ├── removed_mask.png    # 被移除箱子掩码 (输入)
│       │   ├── affected_mask.png   # 受影响区域掩码 (GT)
│       │   └── result.json         # 稳定性标签
│       ├── 1/
│       ...
```

### RGBD 版本数据格式

RGBD 版本需要额外的深度图文件：

```
dataset/
├── round_001_65_top/
│   ├── rgb.png                 # RGB 图像 (640x480)
│   ├── depth.npy               # 深度图 (numpy array, RGBD 版本必需)
│   └── removals/
│       ├── 0/
│       │   ├── removed_mask.png    # 被移除箱子掩码 (输入)
│       │   ├── affected_mask.png   # 受影响区域掩码 (GT)
│       │   └── result.json         # 稳定性标签
│       ├── 1/
│       ...
```

**注意**: 
- RGBD 版本会自动跳过没有 `depth.npy` 的样本
- 深度图应为 numpy array 格式，尺寸与 RGB 图像一致
- 深度图会在加载时自动归一化（支持 `minmax`、`mean_std`、`fixed` 三种方式）

`result.json` 格式:
```json
{
  "is_stable": false,
  "stability_label": 0,
  "affected_boxes": [...]
}
```

---

## 损失函数

- **分类损失**: BCE 或 Focal Loss (处理类别不平衡)
- **分割损失**: Dice + BCE 组合损失

总损失 = λ_cls × L_cls + λ_seg × L_seg

---

## 评估指标

### 分类指标
- Accuracy: 准确率
- Precision: 精确率
- Recall: 召回率
- F1 Score: F1 分数

### 分割指标
- IoU: 交并比
- Dice: Dice 系数

---

## 输出文件

### RGB 版本

训练完成后，检查点保存在项目目录的 `checkpoints/<exp_name>/`:

```
checkpoints/transunet_fusion_base/
├── checkpoint_epoch_10.pth
├── checkpoint_epoch_20.pth
├── checkpoint_epoch_20_best.pth  # 最佳模型
...
```

### RGBD 版本

训练完成后，检查点保存在 `/DATA/disk0/hs_25/pa/checkpoints/<exp_name>/`:

```
/DATA/disk0/hs_25/pa/checkpoints/transunet_rgbd_fusion_base/
├── checkpoint_epoch_5.pth
├── checkpoint_epoch_10.pth
├── checkpoint_epoch_10_best.pth  # 最佳模型
...
```

### 检查点内容

检查点包含:
- `model_state_dict`: 模型权重
- `optimizer_state_dict`: 优化器状态
- `scheduler_state_dict`: 调度器状态
- `epoch`: 当前 epoch
- `metrics`: 训练/验证指标

---

## 技巧与建议

1. **类别不平衡**: 使用 `WeightedRandomSampler` 和 `pos_weight` 参数
2. **过拟合**: 增加 dropout，使用数据增强
3. **训练不稳定**: 降低学习率，增加 warmup epochs
4. **显存不足**: 减小 batch_size，使用 Tiny/Small 模型
5. **训练速度**: 启用混合精度训练 (默认开启)

---

## 模型选择建议

| 场景 | 推荐模型 | 理由 |
|------|----------|------|
| 数据集有深度图 | **RGBD Fusion** | 深度信息 + 融合机制，最佳性能 |
| 数据集只有 RGB | **RGB Fusion** | 融合机制提升分类准确性 |
| 需要快速推理 | **Cls-Only** | 无分割头，参数少 |
| 显存受限 | **Small/Tiny** 变体 | 参数量小 |
| 基线对比 | **Base** 模型 | 标准双任务架构 |
