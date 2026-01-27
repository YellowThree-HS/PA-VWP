# 模型训练指南

本文档说明 `/DATA/disk0/hs_25/pa/checkpoints` 目录下所有模型的训练参数和代码。

## 目录结构说明

所有模型检查点保存在：`/DATA/disk0/hs_25/pa/checkpoints/`

训练日志保存在：`/DATA/disk0/hs_25/pa/logs/` 或项目根目录的 `logs/` 目录

## 模型命名规则

模型名称遵循以下规则：
- **模型架构**: `transunet_*` (TransUNet)
- **输入类型**: 
  - `rgbd_*` - RGBD 输入（RGB + Depth）
  - 无前缀 - RGB 输入
- **任务类型**:
  - `cls_only_*` - 仅分类任务
  - `seg_only_*` - 仅分割任务
  - `seg_from_encoded_*` - 从编码特征进行分割
  - `fusion_*` - 融合模型（分类+分割）
  - 无后缀 - 默认多任务（分类+分割）
- **骨干网络**:
  - `convnext_*` - ConvNeXt 骨干网络
  - `twostream_*` - 双流架构
  - 无前缀 - ResNet-50 骨干网络
- **模型大小**: `base` / `small` / `tiny` / `micro`
- **训练环境**: `_h100` - H100 GPU 训练
- **时间戳**: `_YYYYMMDD_HHMMSS` - 部分模型包含训练时间戳

## 训练脚本映射

### 1. RGB 输入模型

#### 1.1 基础 TransUNet (RGB)
- **检查点**: `transunet_base_h100`
- **训练脚本**: `run_train_h100.sh`
- **Python 脚本**: `train/train.py`
- **参数**:
  - Model: `base`
  - Batch Size: ~36
  - Learning Rate: ~5e-5
  - Epochs: 150
  - Config: `base`

#### 1.2 Fusion 模型 (RGB)
- **检查点**: 
  - `transunet_fusion_base_h100`
  - `transunet_fusion_small_h100`
  - `transunet_fusion_tiny_h100`
- **训练脚本**: 
  - `run_train_fusion_h100.sh` (base)
  - `run_train_fusion_small_h100.sh` (small)
  - `run_train_fusion_tiny_h100.sh` (tiny)
- **Python 脚本**: `train/train_fusion.py`
- **参数** (small 示例):
  - Model: `small`
  - Batch Size: 50
  - Learning Rate: 1e-4
  - Epochs: 150
  - Config: `small`
  - Num Workers: 16

#### 1.3 仅分类模型 (RGB)
- **检查点**: 
  - `transunet_cls_only_base_h100_*` (多个时间戳版本)
  - `transunet_cls_only_small_h100`
  - `transunet_cls_only_tiny_h100`
  - `transunet_cls_only_micro_h100`
- **训练脚本**: 
  - `run_train_cls_only_h100.sh` (base)
  - `run_train_cls_only_small_h100.sh` (small)
  - `run_train_cls_only_tiny_h100.sh` (tiny)
  - `run_train_cls_only_micro_h100.sh` (micro)
- **Python 脚本**: `train/train_cls_only.py`
- **参数** (base 示例):
  - Model: `base`
  - Batch Size: 36
  - Learning Rate: 5e-5
  - Epochs: 150
  - Config: `base`
  - Num Workers: 8

#### 1.4 从编码特征分割 (RGB)
- **检查点**: 
  - `transunet_seg_from_encoded_tiny_h100`
  - `transunet_seg_from_encoded_micro_h100`
- **训练脚本**: 
  - `run_train_seg_from_encoded_h100.sh` (base)
  - `run_train_seg_from_encoded_tiny_h100.sh` (tiny)
  - `run_train_seg_from_encoded_micro_h100.sh` (micro)
  - `run_train_seg_from_encoded_small_h100.sh` (small)
- **Python 脚本**: `train/train_seg_from_encoded.py`
- **参数**: 参考其他模型，根据大小调整 batch size

#### 1.5 仅分割模型 (RGB)
- **检查点**: 
  - `transunet_seg_only_base_h100_20260119_223014`
  - `transunet_seg_only_base_h100_20260120_110712`
- **训练脚本**: 可能使用 `train/train.py` 并禁用分类任务
- **参数**: 参考基础模型

### 2. RGBD 输入模型

#### 2.1 基础 RGBD 模型
- **检查点**: `transunet_rgbd_base_h100`
- **训练脚本**: `run_train_rgbd_h100.sh`
- **Python 脚本**: `train/train_with_depth.py` 或 `train/train_rgbd_fusion.py`
- **参数**:
  - Model: `base`
  - Batch Size: ~32 (RGBD 输入更大)
  - Learning Rate: ~5e-5
  - Epochs: 150

#### 2.2 RGBD Fusion 模型
- **检查点**: 
  - `transunet_rgbd_fusion_base_h100`
  - `transunet_rgbd_fusion_small_h100`
  - `transunet_rgbd_fusion_tiny_h100`
- **训练脚本**: 
  - `run_train_rgbd_fusion_h100.sh` (base)
  - `run_train_rgbd_fusion_small_h100.sh` (small)
  - `run_train_rgbd_fusion_tiny_h100.sh` (tiny)
  - `run_train_rgbd_fusion_micro_h100.sh` (micro)
- **Python 脚本**: `train/train_rgbd_fusion.py`
- **参数**: 参考 RGB Fusion 模型，batch size 稍小

#### 2.3 RGBD 仅分类模型
- **检查点**: 
  - `transunet_rgbd_cls_only_base_h100_20260119_223142`
  - `transunet_rgbd_cls_only_small_h100`
  - `transunet_rgbd_cls_only_tiny_h100`
- **训练脚本**: 
  - `run_train_rgbd_cls_only_h100.sh` (base)
  - `run_train_rgbd_cls_only_small_h100.sh` (small)
  - `run_train_rgbd_cls_only_tiny_h100.sh` (tiny)
  - `run_train_rgbd_cls_only_micro_h100.sh` (micro)
- **Python 脚本**: `train/train_rgbd_cls_only.py`
- **参数** (base 示例):
  - Model: `base`
  - Batch Size: 32
  - Learning Rate: 5e-5
  - Epochs: 150
  - Config: `base`
  - Num Workers: 8

#### 2.4 RGBD 从编码特征分割
- **检查点**: 
  - `transunet_rgbd_seg_from_encoded_tiny_h100`
  - `transunet_rgbd_seg_from_encoded_tiny_trainv2_h100`
- **训练脚本**: 
  - `run_train_rgbd_seg_from_encoded_tiny_h100.sh`
  - `run_train_rgbd_seg_from_encoded_tiny_trainv2_h100.sh`
- **Python 脚本**: `train/train_rgbd_seg_from_encoded.py`
- **参数**: 参考 RGB 版本

#### 2.5 RGBD 仅分割模型
- **检查点**: 
  - `transunet_rgbd_seg_only_base_h100_20260120_105127`
  - `transunet_rgbd_seg_only_base_h100_20260120_120545`
- **训练脚本**: 可能使用 RGBD 训练脚本并禁用分类任务
- **参数**: 参考 RGBD 基础模型

### 3. ConvNeXt 骨干网络模型

#### 3.1 ConvNeXt Base (RGB)
- **检查点**: 
  - `transunet_convnext_base_h100`
  - `transunet_convnext_base_h100_eval`
- **训练脚本**: 可能使用 `train/train.py` 并指定 ConvNeXt 骨干网络
- **参数**: 参考基础模型

#### 3.2 ConvNeXt RGBD 模型
- **检查点**: 
  - `transunet_convnext_rgbd_base_h100`
  - `transunet_convnext_rgbd_cls_only_base_h100`
  - `transunet_convnext_rgbd_seg_only_base_h100`
- **训练脚本**: 使用 ConvNeXt 版本的 RGBD 训练脚本
- **参数**: 参考对应的 RGBD 模型

#### 3.3 ConvNeXt 仅分类/分割
- **检查点**: 
  - `transunet_convnext_cls_only_base_h100`
  - `transunet_convnext_seg_only_base_h100`
- **训练脚本**: 使用 ConvNeXt 版本的对应任务训练脚本
- **参数**: 参考对应的任务模型

### 4. 双流模型

#### 4.1 双流仅分类
- **检查点**: `transunet_twostream_cls_only_tiny_h100`
- **训练脚本**: `run_train_twostream_cls_only_tiny_h100.sh`
- **Python 脚本**: `train/train_twostream_cls_only.py`
- **参数**: 
  - Model: `tiny`
  - 其他参数参考 tiny 模型

#### 4.2 双流从编码特征分割
- **检查点**: `transunet_twostream_seg_from_encoded_tiny_h100`
- **训练脚本**: `run_train_twostream_seg_from_encoded_tiny_h100.sh`
- **Python 脚本**: `train/train_twostream_seg_from_encoded.py`
- **参数**: 
  - Model: `tiny`
  - 其他参数参考 tiny 模型

### 5. 其他模型

#### 5.1 ResNet+RGB
- **检查点**: `resnet+rgb`
- **训练脚本**: 可能是早期实验模型
- **说明**: 使用 ResNet 骨干网络的 RGB 输入模型

## 通用训练参数

### 模型大小对应的典型参数

| 模型大小 | Batch Size | Learning Rate | 参数量 |
|---------|-----------|---------------|--------|
| micro   | ~60-80    | 1e-4 - 2e-4  | 最小   |
| tiny    | ~50-60    | 1e-4         | 小     |
| small   | ~40-50    | 1e-4         | 中     |
| base    | ~32-36    | 5e-5         | 大     |

### 输入类型对应的 Batch Size 调整

- **RGB (3通道)**: 标准 batch size
- **RGBD (5通道)**: batch size 减少约 10-20%

### 训练环境

- **GPU**: H100
- **Conda 环境**: `isaac`
- **工作目录**: `/home/hs_25/projs/PA-VWP`
- **保存目录**: `/DATA/disk0/hs_25/pa/checkpoints`
- **日志目录**: `/DATA/disk0/hs_25/pa/logs` 或 `logs/`

### SLURM 配置 (部分脚本)

```bash
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
```

## 训练命令示例

### 基础训练命令格式

```bash
cd /home/hs_25/projs/PA-VWP
source /home/hs_25/miniconda3/etc/profile.d/conda.sh
conda activate isaac

python train/train_<task>.py \
    --config <size> \
    --model <variant> \
    --exp_name <exp_name> \
    --batch_size <batch_size> \
    --epochs 150 \
    --lr <learning_rate> \
    --num_workers <num_workers> \
    --save_dir /DATA/disk0/hs_25/pa/checkpoints
```

### 恢复训练

大部分脚本支持自动检测并恢复训练：
- 优先使用最佳 checkpoint (`checkpoint_*best*.pth`)
- 其次使用最新 checkpoint (`checkpoint_epoch_*.pth`)
- 使用 `--resume <checkpoint_path>` 参数

## 模型检查点结构

每个模型检查点目录包含：
- `checkpoint_epoch_<N>.pth` - 每个 epoch 的检查点
- `checkpoint_*best*.pth` - 最佳性能检查点
- 可能包含训练配置和日志文件

## 注意事项

1. **时间戳版本**: 部分模型名称包含时间戳（如 `_20260119_215816`），表示同一配置的多次训练
2. **自动恢复**: 大部分训练脚本支持自动检测并恢复训练
3. **日志位置**: 训练日志可能在项目 `logs/` 目录或 `/DATA/disk0/hs_25/pa/logs/`
4. **配置参数**: 具体配置参数可能在 `config/` 目录下的配置文件中定义

## 快速查找训练脚本

根据模型名称快速定位训练脚本：

1. 确定输入类型（RGB/RGBD）
2. 确定任务类型（cls_only/seg_only/fusion/seg_from_encoded）
3. 确定模型大小（base/small/tiny/micro）
4. 确定骨干网络（convnext/twostream/默认）
5. 组合为脚本名称：`run_train_<输入>_<任务>_<大小>_h100.sh`

例如：
- `transunet_rgbd_fusion_small_h100` → `run_train_rgbd_fusion_small_h100.sh`
- `transunet_cls_only_base_h100` → `run_train_cls_only_h100.sh`
- `transunet_twostream_cls_only_tiny_h100` → `run_train_twostream_cls_only_tiny_h100.sh`
