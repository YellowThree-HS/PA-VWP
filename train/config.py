"""
训练配置
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path


@dataclass
class DataConfig:
    """数据配置"""
    # 数据目录
    train_dirs: List[str] = field(default_factory=lambda: ['dataset_messy_small'])
    val_dirs: Optional[List[str]] = None  # 如果为 None，从训练集划分
    
    # 图像尺寸
    img_height: int = 480
    img_width: int = 640
    
    # 数据加载
    batch_size: int = 8
    num_workers: int = 4
    val_split: float = 0.2
    use_weighted_sampler: bool = True
    
    # 数据增强
    use_augmentation: bool = True
    

@dataclass
class ModelConfig:
    """模型配置"""
    # 模型变体: 'tiny', 'small', 'base'
    variant: str = 'small'
    
    # 预训练
    pretrained: bool = True
    freeze_bn: bool = False
    
    # Transformer 配置 (当 variant='custom' 时使用)
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    mlp_dim: int = 2048
    dropout: float = 0.1
    
    # 任务权重
    cls_weight: float = 1.0   # 分类任务权重
    seg_weight: float = 1.0   # 分割任务权重


@dataclass
class TrainConfig:
    """训练配置"""
    # 训练参数
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-4
    
    # 学习率调度
    scheduler: str = 'cosine'  # 'cosine', 'step', 'plateau'
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # 优化器
    optimizer: str = 'adamw'  # 'adamw', 'adam', 'sgd'
    
    # 梯度裁剪
    grad_clip: float = 1.0
    
    # 混合精度训练
    use_amp: bool = True
    
    # 早停
    early_stopping_patience: int = 15
    
    # 保存
    save_dir: str = 'checkpoints'
    save_freq: int = 5  # 每 N 个 epoch 保存一次
    save_best_only: bool = True
    
    # 日志
    log_freq: int = 10  # 每 N 个 batch 打印一次
    use_wandb: bool = False
    wandb_project: str = 'boxworld-stability'
    wandb_run_name: Optional[str] = None
    
    # 随机种子
    seed: int = 42


@dataclass  
class LossConfig:
    """损失函数配置"""
    # 分类损失
    cls_loss: str = 'bce'  # 'bce', 'focal'
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    
    # 分割损失
    seg_loss: str = 'dice_bce'  # 'bce', 'dice', 'dice_bce'
    dice_smooth: float = 1.0
    
    # 类别权重 (用于处理不平衡)
    use_class_weights: bool = True
    pos_weight: float = 2.0  # 不稳定类的权重


@dataclass
class Config:
    """完整配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    
    # 实验名称
    exp_name: str = 'transunet_stability'
    
    # 设备
    device: str = 'cuda'  # 'cuda', 'cpu'
    
    def __post_init__(self):
        # 创建保存目录
        save_path = Path(self.train.save_dir) / self.exp_name
        save_path.mkdir(parents=True, exist_ok=True)


def get_config(preset: str = 'default') -> Config:
    """获取预设配置"""
    
    if preset == 'default':
        return Config()
    
    elif preset == 'debug':
        # 调试配置：小批次，快速迭代
        return Config(
            data=DataConfig(
                batch_size=2,
                num_workers=0,
            ),
            model=ModelConfig(
                variant='tiny',
            ),
            train=TrainConfig(
                epochs=5,
                lr=1e-4,
                log_freq=1,
                use_amp=False,
            ),
            exp_name='debug',
        )
    
    elif preset == 'small':
        # 小模型配置
        return Config(
            model=ModelConfig(
                variant='small',
            ),
            train=TrainConfig(
                epochs=100,
                lr=1e-4,
            ),
            exp_name='transunet_small',
        )
    
    elif preset == 'base':
        # 完整模型配置
        return Config(
            data=DataConfig(
                batch_size=4,  # 较大模型，减小批次
            ),
            model=ModelConfig(
                variant='base',
            ),
            train=TrainConfig(
                epochs=150,
                lr=5e-5,
                grad_clip=0.5,
            ),
            exp_name='transunet_base',
        )
    
    elif preset == 'full_data':
        # 使用所有数据集
        return Config(
            data=DataConfig(
                train_dirs=['dataset3', 'dataset_messy_small'],
                batch_size=8,
            ),
            model=ModelConfig(
                variant='small',
            ),
            train=TrainConfig(
                epochs=100,
            ),
            exp_name='transunet_full',
        )
    
    else:
        raise ValueError(f"未知的预设配置: {preset}")


if __name__ == "__main__":
    # 打印默认配置
    config = get_config('default')
    print("默认配置:")
    print(f"  数据目录: {config.data.train_dirs}")
    print(f"  模型变体: {config.model.variant}")
    print(f"  批次大小: {config.data.batch_size}")
    print(f"  训练轮数: {config.train.epochs}")
    print(f"  学习率: {config.train.lr}")
