# 数据采集文档

本文档详细说明 BoxWorld-MVP 项目的数据采集流程、数据格式和关键算法。

## 目录

- [快速开始](#快速开始)
- [采集流程](#采集流程)
- [相机配置](#相机配置)
- [可见性判断](#可见性判断)
- [数据格式](#数据格式)
- [数据增强](#数据增强)
- [配置参数](#配置参数)

## 快速开始

```bash
# 可视化模式（观察场景）
python main.py

# 无头模式批量采集（推荐）
python collect_dataset.py --rounds 100 --output dataset

# 带GUI的采集模式
python collect_dataset.py --gui --rounds 10
```

## 采集流程

每轮数据采集包含以下步骤：

1. **场景生成**
   - 随机生成 30-100 个纸箱
   - 箱子尺寸范围：5-20cm
   - 随机光照条件（强度、色温、方向）
   - 随机地面纹理

2. **物理仿真**
   - 等待箱子堆叠稳定
   - 动态速度收敛检测，自适应等待时间

3. **初始状态采集**
   - 9 个视角同时拍摄 RGB、深度图、实例分割掩码
   - 计算每个视角的可见箱子列表

4. **稳定性测试**
   - 对所有视角可见箱子的并集进行测试
   - 逐个移除箱子，运行物理仿真
   - 记录是否有其他箱子发生位移（阈值 2cm）
   - 恢复场景，测试下一个箱子

5. **数据保存**
   - 每个视角只保存该视角下可见的箱子数据
   - 生成标注图和掩码文件

## 相机配置

### 9 个相机视角

| 视角 | 位置 | 说明 |
|------|------|------|
| top | 正上方 | 俯视图 |
| north | +Y 方向 | 前方 |
| south | -Y 方向 | 后方 |
| east | +X 方向 | 右侧 |
| west | -X 方向 | 左侧 |
| northeast | +X+Y | 右前 45° |
| northwest | -X+Y | 左前 45° |
| southeast | +X-Y | 右后 45° |
| southwest | -X-Y | 左后 45° |

### 相机参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 图像尺寸 | 640×480 | 宽×高 |
| 焦距 | 18.0 mm | Isaac Sim 默认 |
| 水平孔径 | 20.955 mm | Isaac Sim 默认 |

## 可见性判断

采用基于 3D 投影的可见性判断算法，解决传统像素比例方法的问题。

### 算法原理

```
可见比例 = 实际像素数 / 理论投影像素数
```

### 理论投影像素数计算

根据箱子的 3D 尺寸、位置和相机参数估算理论上应该占据的像素数：

1. **计算视线方向**：从箱子指向相机的单位向量
2. **计算可见面**：只有法向量与视线方向夹角 < 90° 的面可见
3. **投影面积**：`投影面积 = 实际面积 × cos(θ)`
4. **像素转换**：`pixels_per_meter = (focal_length / horizontal_aperture) × image_width / distance`

### 阈值设置

| 参数 | 值 | 说明 |
|------|-----|------|
| min_visible_ratio | 0.20 | 可见比例阈值 |

### 优势

- 大箱子被遮挡 90% → 正确判定为不可见
- 小箱子完全暴露 → 正确判定为可见
- 避免了简单像素计数对箱子大小的偏差

## 数据格式

### 目录结构

```
dataset/
├── round_001_65_top/           # round_轮次_箱子数_视角名
│   ├── rgb.png                 # RGB图像 (640x480)
│   ├── depth.png               # 深度图可视化
│   ├── depth.npy               # 深度原始数据 (float32, 米)
│   ├── mask.npy                # 实例分割掩码 (int32)
│   ├── visible_mask.png        # 该视角可见箱子掩码
│   ├── visible_boxes.json      # 该视角可见箱子列表
│   ├── summary.json            # 该视角统计信息
│   ├── summary.png             # 稳定性总结图
│   ├── visibility_stats.json   # 可见性统计数据
│   └── removals/               # 移除测试结果
│       ├── 0/
│       │   ├── annotated.png       # 标注图
│       │   ├── removed_mask.png    # 被移除箱子掩码
│       │   ├── affected_mask.png   # 受影响箱子掩码
│       │   └── result.json         # 稳定性结果
│       ├── 1/
│       ...
├── round_001_65_north/
├── round_001_65_south/
...
```

### result.json 格式

```json
{
  "removed_box": {
    "name": "box_42",
    "prim_path": "/World/box_42"
  },
  "is_stable": false,
  "stability_label": 0,
  "affected_boxes": [
    {
      "path": "/World/box_15",
      "name": "box_15",
      "displacement": 0.035
    }
  ]
}
```

### summary.json 格式

```json
{
  "total_visible": 25,
  "stable_count": 18,
  "unstable_count": 7,
  "stable_boxes": [...],
  "unstable_boxes": [...]
}
```

### 标注图颜色说明

| 图像 | 颜色 | 含义 |
|------|------|------|
| annotated.png | 绿色 | 被移除的箱子 |
| annotated.png | 红色 | 受影响的箱子 |
| summary.png | 黄色 | 稳定的箱子 |
| summary.png | 蓝色 | 不稳定的箱子 |

## 数据增强

采集时自动添加模拟真实相机的噪声，提高模型泛化能力。

### RGB 图像噪声

| 噪声类型 | 参数范围 | 说明 |
|----------|----------|------|
| 散粒噪声 | 0.005-0.02 | 模拟光子噪声 |
| 高斯噪声 | σ = 0.5-2.5 | 模拟读取噪声 |
| 颜色抖动 | 0.9875-1.0125 | 模拟白平衡偏移 |

### 深度图噪声

| 噪声类型 | 参数范围 | 说明 |
|----------|----------|------|
| 高斯噪声 | σ = 0.002-0.008 m | 测量噪声 |
| 距离相关噪声 | 0.001-0.003 | 远处噪声更大 |
| 随机缺失 | 0.1%-0.5% | 模拟反射失败 |

## 配置参数

### collect_dataset.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--rounds` | 10 | 采集轮数 |
| `--output` | dataset | 输出目录 |
| `--gui` | False | 是否显示 GUI |
| `--min_boxes` | 30 | 最少箱子数 |
| `--max_boxes` | 100 | 最多箱子数 |

### 稳定性判定参数

| 参数 | 值 | 说明 |
|------|-----|------|
| displacement_threshold | 0.02 m | 位移阈值 |
| stability_wait_time | 2.0 s | 仿真等待时间 |

## 分析工具

### analyze_visibility.py

用于分析可见性统计数据，帮助确定合理的阈值：

```bash
python analyze_visibility.py
```

输出包括：
- visibility_ratio 分布统计
- actual_pixels 分布统计
- 不同阈值下的数据保留比例
- 组合阈值效果分析
