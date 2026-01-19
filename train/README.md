# TransUNet 物理稳定性预测训练模块

基于 Attention 的 TransUNet 架构，用于预测移除箱子后场景的物理稳定性。

## 模型版本

本项目提供两个版本的模型：

- **RGB 版本** (`train.py`): 4 通道输入 - RGB (3) + Target Mask (1)
- **RGBD 版本** (`train_with_depth.py`): 5 通道输入 - RGB (3) + Depth (1) + Target Mask (1)

RGBD 版本通过引入深度图信息，可以更好地理解场景的 3D 结构，提升物理稳定性预测的准确性。

## 架构概述

### RGB 版本 (4 通道)

```
Hybrid-Encoder + Dual-Decoder

输入: RGB (3) + Target Mask (1) = 4 通道

┌─────────────────────────────────────────────────────────────┐
│  CNN Stem (ResNet-50)                                       │
│  - Conv1 (4 → 64 channels, /2)                              │
│  - Layer1 (256 channels, /4) ─────────────────────┐         │
│  - Layer2 (512 channels, /8) ────────────────────┐│         │
│  - Layer3 (1024 channels, /16) ─────────────────┐││         │
└──────────────────────────────────────────────────┼┼┼────────┘
                                                   │││
┌──────────────────────────────────────────────────┼┼┼────────┐
│  Transformer Bottleneck (物理推理核心)            │││        │
│  - Patch Embedding (1024 → hidden_dim)           │││        │
│  - Position Encoding                             │││        │
│  - N × Transformer Layers                        │││        │
│  - Self-Attention: 建立全局物理关联              │││        │
└───────────────────┬──────────────────────────────┼┼┼────────┘
                    │                              │││
        ┌───────────┴───────────┐                  │││
        │                       │                  │││
┌───────▼─────────┐   ┌─────────▼─────────────────┼┼┼────────┐
│ 分类头 (Task 1)  │   │  分割头 (Task 2)           ↓↓↓       │
│                 │   │  - UpBlock + Skip Connect ───────────│
│ Global Pooling  │   │  - 逐级上采样恢复分辨率              │
│ MLP → Sigmoid   │   │  - 输出受影响区域掩码                │
│                 │   │                                      │
│ 输出: 稳定/不稳定│   │  输出: Mask (1, 480, 640)            │
└─────────────────┘   └──────────────────────────────────────┘
```

### RGBD 版本 (5 通道)

```
Hybrid-Encoder + Dual-Decoder

输入: RGB (3) + Depth (1) + Target Mask (1) = 5 通道

┌─────────────────────────────────────────────────────────────┐
│  CNN Stem (ResNet-50)                                       │
│  - Conv1 (5 → 64 channels, /2)  # 支持深度图输入             │
│  - Layer1 (256 channels, /4) ─────────────────────┐         │
│  - Layer2 (512 channels, /8) ────────────────────┐│         │
│  - Layer3 (1024 channels, /16) ─────────────────┐││         │
└──────────────────────────────────────────────────┼┼┼────────┘
                                                   │││
┌──────────────────────────────────────────────────┼┼┼────────┐
│  Transformer Bottleneck (物理推理核心)            │││        │
│  - Patch Embedding (1024 → hidden_dim)           │││        │
│  - Position Encoding                             │││        │
│  - N × Transformer Layers                        │││        │
│  - Self-Attention: 建立全局物理关联              │││        │
└───────────────────┬──────────────────────────────┼┼┼────────┘
                    │                              │││
        ┌───────────┴───────────┐                  │││
        │                       │                  │││
┌───────▼─────────┐   ┌─────────▼─────────────────┼┼┼────────┐
│ 分类头 (Task 1)  │   │  分割头 (Task 2)           ↓↓↓       │
│                 │   │  - UpBlock + Skip Connect ───────────│
│ Global Pooling  │   │  - 逐级上采样恢复分辨率              │
│ MLP → Sigmoid   │   │  - 输出受影响区域掩码                │
│                 │   │                                      │
│ 输出: 稳定/不稳定│   │  输出: Mask (1, 480, 640)            │
└─────────────────┘   └──────────────────────────────────────┘
```

## 快速开始

### 1. 安装依赖

```bash
cd train
pip install -r requirements.txt
```

### 2. 开始训练

#### RGB 版本 (4 通道)

```bash
# 默认配置
python train.py --config default

# 调试模式 (小批次，快速迭代)
python train.py --config debug

# 使用完整模型
python train.py --config base --epochs 150

# 使用所有数据集
python train.py --config full_data
```

#### RGBD 版本 (5 通道，带深度图)

```bash
# 默认配置
python train/train_with_depth.py --config default

# 调试模式
python train/train_with_depth.py --config debug

# 使用完整模型
python train/train_with_depth.py --config base --epochs 150

# 使用 H100 训练脚本 (推荐)
bash run_train_rgbd_h100.sh
# 或前台运行
bash run_train_rgbd_h100.sh --foreground
```

**注意**: RGBD 版本需要数据集中包含 `depth.npy` 文件。

### 3. 评估模型

```bash
python evaluate.py --checkpoint checkpoints/transunet_small/checkpoint_best.pth --visualize
```

### 4. 单图推理

```bash
python inference.py \
    --checkpoint best.pth \
    --image path/to/rgb.png \
    --mask path/to/removed_mask.png
```

## 配置预设

| 预设 | 模型 | 批次 | Epochs | 说明 |
|------|------|------|--------|------|
| `debug` | Tiny | 2 | 5 | 快速调试 |
| `default` | Small | 8 | 100 | 默认训练 |
| `small` | Small | 8 | 100 | 小模型 |
| `base` | Base | 4 | 150 | 完整模型 |
| `full_data` | Small | 8 | 100 | 使用所有数据 |

## 模型变体

### RGB 版本模型

| 变体 | Transformer 层数 | Hidden Dim | 参数量 | 输入通道 |
|------|------------------|------------|--------|----------|
| Tiny | 4 | 384 | ~30M | 4 |
| Small | 6 | 512 | ~50M | 4 |
| Base | 12 | 768 | ~100M | 4 |

### RGBD 版本模型

| 变体 | Transformer 层数 | Hidden Dim | 参数量 | 输入通道 |
|------|------------------|------------|--------|----------|
| Tiny | 4 | 384 | ~30M | 5 |
| Small | 6 | 512 | ~50M | 5 |
| Base | 12 | 768 | ~100M | 5 |

RGBD 版本的第一层卷积从 3 通道扩展到 5 通道：
- RGB 通道 (0-2): 使用 ImageNet 预训练权重
- Depth 通道 (3): 使用 RGB 通道权重的平均值初始化
- Target Mask 通道 (4): 初始化为 0

## 命令行参数

```bash
python train.py --help
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

## 数据格式

训练数据需要遵循以下目录结构:

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

## 损失函数

- **分类损失**: BCE 或 Focal Loss (处理类别不平衡)
- **分割损失**: Dice + BCE 组合损失

总损失 = λ_cls × L_cls + λ_seg × L_seg

## 评估指标

### 分类指标
- Accuracy: 准确率
- Precision: 精确率
- Recall: 召回率
- F1 Score: F1 分数

### 分割指标
- IoU: 交并比
- Dice: Dice 系数

## 技巧与建议

1. **类别不平衡**: 使用 `WeightedRandomSampler` 和 `pos_weight` 参数
2. **过拟合**: 增加 dropout，使用数据增强
3. **训练不稳定**: 降低学习率，增加 warmup epochs
4. **显存不足**: 减小 batch_size，使用 Tiny/Small 模型
5. **训练速度**: 启用混合精度训练 (默认开启)

## 输出文件

### RGB 版本

训练完成后，检查点保存在项目目录的 `checkpoints/<exp_name>/`:

```
checkpoints/transunet_small/
├── checkpoint_epoch_10.pth
├── checkpoint_epoch_20.pth
├── checkpoint_epoch_20_best.pth  # 最佳模型
...
```

### RGBD 版本

训练完成后，检查点保存在 `/DATA/disk0/hs_25/pa/checkpoints/<exp_name>/`:

```
/DATA/disk0/hs_25/pa/checkpoints/transunet_rgbd_base_h100/
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

## RGB vs RGBD 对比

| 特性 | RGB 版本 | RGBD 版本 |
|------|----------|-----------|
| 输入通道 | 4 (RGB + Mask) | 5 (RGB + Depth + Mask) |
| 数据要求 | 仅需 RGB 图像 | 需要 RGB + 深度图 |
| 模型文件 | `train.py` | `train_with_depth.py` |
| 数据集类 | `BoxWorldDataset` | `BoxWorldDatasetWithDepth` |
| 模型类 | `TransUNet` | `TransUNetRGBD` |
| 保存路径 | `checkpoints/` | `/DATA/disk0/hs_25/pa/checkpoints/` |
| 优势 | 简单快速 | 更好的 3D 理解能力 |

**建议**: 
- 如果数据集包含深度图，推荐使用 RGBD 版本以获得更好的性能
- 如果只有 RGB 图像，使用 RGB 版本即可
