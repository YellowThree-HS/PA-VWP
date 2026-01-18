# TransUNet 物理稳定性预测训练模块

基于 Attention 的 TransUNet 架构，用于预测移除箱子后场景的物理稳定性。

## 架构概述

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

## 快速开始

### 1. 安装依赖

```bash
cd train
pip install -r requirements.txt
```

### 2. 开始训练

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

| 变体 | Transformer 层数 | Hidden Dim | 参数量 |
|------|------------------|------------|--------|
| Tiny | 4 | 384 | ~30M |
| Small | 6 | 512 | ~50M |
| Base | 12 | 768 | ~100M |

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

训练完成后，检查点保存在 `checkpoints/<exp_name>/`:

```
checkpoints/transunet_small/
├── checkpoint_epoch_10.pth
├── checkpoint_epoch_20.pth
├── checkpoint_epoch_20_best.pth  # 最佳模型
...
```

检查点包含:
- `model_state_dict`: 模型权重
- `optimizer_state_dict`: 优化器状态
- `scheduler_state_dict`: 调度器状态
- `epoch`: 当前 epoch
- `metrics`: 训练/验证指标
