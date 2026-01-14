# PA-VWP (Physics-Aware VideoWorld Planner)

基于视频世界模型的物理感知机器人抓取规划系统。

## 核心思想

利用 VideoWorld 作为世界模拟器 (World Simulator)，在抓取前通过动作条件视频预测 (Action-Conditioned Video Prediction)，在潜在空间 (Latent Space) 中并行模拟抓取不同箱子的后果，利用 MPC (模型预测控制) 选择"画面最稳定"的动作序列。

## 当前进度

目前正在搭建基于 Isaac Sim 的仿真环境，用于数据采集和验证：

- [x] UR5 机械臂运动控制与轨迹插值
- [x] CuRobo GPU 加速逆运动学求解（带碰撞检测）
- [x] 真空吸盘末端执行器仿真
- [x] 动态纸箱生成与物理仿真
- [ ] VideoWorld 视频预测模型
- [ ] MPC 规划器
- [ ] 多箱子场景

## 项目结构

```
PA-VWP/
├── main.py                 # 主仿真脚本
├── config/
│   └── robot_config.yaml   # 机器人配置文件
├── src/
│   ├── ur5_controller.py   # UR5 控制器 + CuRobo IK
│   ├── suction_gripper.py  # 吸盘末端执行器
│   └── box_generator.py    # 纸箱生成器
└── curobo/                 # NVIDIA CuRobo 运动规划库
```

## 环境要求

- NVIDIA Isaac Sim 4.5+
- Python 3.10+
- NVIDIA GPU (RTX 2070+)
- CUDA 11.8+

## 运行方法

```bash
# Windows (使用 Isaac Sim Python 环境)
path\to\isaac_sim\python.bat main.py

# Linux
~/.local/share/ov/pkg/isaac_sim-*/python.sh main.py
```

## 工作流程

```
INIT -> MOVE_TO_SAFE -> APPROACH -> PICK -> GRASP -> LIFT -> DONE
```

1. 初始化场景，UR5 移动到直立位置
2. 生成纸箱，等待物理稳定
3. 移动到安全高度
4. 接近纸箱上方
5. 下降到抓取位置
6. 吸盘激活，吸附纸箱
7. 提升纸箱

## 配置参数

可在 `config/robot_config.yaml` 中修改：

- 纸箱尺寸和生成位置
- 吸盘吸附距离阈值
- 运动速度参数
