# BoxWorld-MVP

基于深度学习的纸箱堆叠稳定性预测系统。通过 Isaac Sim 仿真采集数据，训练神经网络预测移除某个箱子后堆叠结构是否会倒塌。

## 核心功能

### 1. 仿真数据采集
- 基于 NVIDIA Isaac Sim 的物理仿真环境
- 自动生成随机纸箱堆叠场景
- 循环测试每个可见箱子的移除稳定性
- 自动标注：稳定(1) / 不稳定(0)

### 2. 稳定性预测模型

提供三种模型架构：

| 模型 | 文件 | 特点 |
|------|------|------|
| ResNet | `model.py` | 单流 ResNet，RGB+Mask 4通道输入 |
| Two-Stream ResNet | `model_twostream.py` | 双流 ResNet，全局+局部视图 |
| **Two-Stream ViT** | `model_vit_twostream.py` | 双流 ViT + Cross-Attention 融合 |

### 3. ViT 双流网络架构 (推荐)

```
Global Stream (ViT)              Local Stream (ViT)
输入: 全图 + Mask (4ch)          输入: 局部裁剪 (3ch)
      ↓                                ↓
  所有 Patch Tokens              CLS Token (Query)
      ↓                                ↓
      └──────── Cross-Attention ───────┘
                      ↓
                 MLP Head
                      ↓
              稳定性预测 (0/1)
```

**设计思路：**
- **Global Stream**: 理解目标箱子在整体堆叠结构中的位置
- **Local Stream**: 观察目标箱子周围的接触细节
- **Cross-Attention**: Local CLS 查询 Global Patches，融合宏观结构与微观细节

## 项目结构

```
BoxWorld-MVP/
├── main.py                 # 可视化仿真脚本（带GUI）
├── collect_dataset.py      # 数据集采集脚本（无头模式）
├── lib/                    # 公共库模块
│   ├── scene_builder.py    # 场景构建（地面、光源、纹理）
│   ├── camera_manager.py   # 相机和标注器管理
│   ├── stability_checker.py # 稳定性检测
│   └── image_utils.py      # 图像处理工具
├── src/
│   └── box_generator.py    # 纸箱生成器
├── train/
│   ├── model.py                    # 单流 ResNet
│   ├── model_twostream.py          # 双流 ResNet
│   ├── model_vit_twostream.py      # 双流 ViT (推荐)
│   ├── dataset.py                  # 单流数据集
│   ├── dataset_twostream.py        # 双流 ResNet 数据集
│   ├── dataset_vit_twostream.py    # 双流 ViT 数据集
│   ├── train.py                    # 单流训练脚本
│   ├── train_twostream.py          # 双流 ResNet 训练
│   └── train_vit_twostream.py      # 双流 ViT 训练
├── dataset/                # 采集的数据集
└── curobo/                 # NVIDIA CuRobo 库
```

## 环境要求

- NVIDIA Isaac Sim 5.1+
- Python 3.10+
- PyTorch 2.0+
- NVIDIA GPU (RTX 2070+)
- timm (用于加载 ViT 预训练权重)

## 使用方法

### 1. 数据采集

```bash
# 可视化模式（观察场景）
python main.py

# 无头模式批量采集（推荐）
python collect_dataset.py --rounds 100 --output dataset

# 带GUI的采集模式
python collect_dataset.py --gui --rounds 10
```

### 2. 训练模型

```bash
cd train

# 训练 ViT 双流模型 (推荐)
python train_vit_twostream.py --model_size small --pretrained --amp

# 训练双流 ResNet
python train_twostream.py --backbone resnet50

# 训练单流 ResNet
python train.py
```

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `../dataset` | 数据集目录 |
| `--epochs` | 50 | 训练轮数 |
| `--batch_size` | 16 | 批大小 |
| `--lr` | 1e-4 | 学习率 |
| `--model_size` | small | ViT 大小: tiny/small/base |
| `--amp` | False | 启用混合精度训练 |

## 模型规格

### ViT Two-Stream

| Size | Embed Dim | Depth | Heads | Params |
|------|-----------|-------|-------|--------|
| tiny | 192 | 12 | 3 | ~12M |
| small | 384 | 12 | 6 | ~47M |
| base | 768 | 12 | 12 | ~180M |

## 数据集格式

```
dataset/
├── round_001/
│   ├── initial/
│   │   └── rgb.png           # 初始场景图像
│   └── removals/
│       ├── 00/
│       │   ├── mask.png      # 目标箱子掩码
│       │   └── result.json   # 稳定性标签
│       ├── 01/
│       ...
├── round_002/
...
```

## License

MIT
