"""
TransUNet ConvNeXt 训练脚本
使用 ConvNeXt 作为 CNN Encoder

用法:
    python train_convnext.py --config base
    python train_convnext.py --config debug
    python train_convnext.py --config base --epochs 200 --lr 1e-4

与 ResNet 版本 (train.py) 的区别:
- 使用 ConvNeXt 替代 ResNet-50 作为 backbone
- 使用 TransUNetConvNeXt 模型
- ConvNeXt 通常性能更强，借鉴了 Transformer 设计
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from tqdm import tqdm

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train.config import Config, get_config, DataConfig, ModelConfig, TrainConfig, LossConfig
from train.dataset import BoxWorldDataset, create_dataloaders, get_train_transform, get_val_transform
from train.models.transunet_convnext import (
    TransUNetConvNeXt, 
    TransUNetConvNeXtConfig,
    transunet_convnext_base, 
    transunet_convnext_small, 
    transunet_convnext_tiny
)
from train.utils import (
    set_seed,
    get_device,
    CombinedLoss,
    MetricCalculator,
    create_optimizer,
    create_scheduler,
    save_checkpoint,
    load_checkpoint,
    AverageMeter,
    ProgressLogger,
    EarlyStopping,
)


def create_model_convnext(config: Config, device: torch.device) -> nn.Module:
    """创建 ConvNeXt 模型"""
    model_config = config.model
    
    if model_config.variant == 'tiny':
        model = transunet_convnext_tiny(
            pretrained=model_config.pretrained,
            img_height=config.data.img_height,
            img_width=config.data.img_width,
            dropout=model_config.dropout,
        )
    elif model_config.variant == 'small':
        model = transunet_convnext_small(
            pretrained=model_config.pretrained,
            img_height=config.data.img_height,
            img_width=config.data.img_width,
            dropout=model_config.dropout,
        )
    elif model_config.variant == 'base':
        model = transunet_convnext_base(
            pretrained=model_config.pretrained,
            img_height=config.data.img_height,
            img_width=config.data.img_width,
            dropout=model_config.dropout,
        )
    else:
        # 自定义配置
        transunet_config = TransUNetConvNeXtConfig(
            img_height=config.data.img_height,
            img_width=config.data.img_width,
            pretrained=model_config.pretrained,
            hidden_dim=model_config.hidden_dim,
            num_heads=model_config.num_heads,
            num_layers=model_config.num_layers,
            mlp_dim=model_config.mlp_dim,
            dropout=model_config.dropout,
            in_channels=4,  # RGB + Mask
        )
        model = TransUNetConvNeXt(transunet_config)
        
    model = model.to(device)
    
    # 打印模型信息
    n_params = model.get_num_params() / 1e6
    n_total_params = model.get_num_params(trainable_only=False) / 1e6
    print(f"模型: TransUNet-ConvNeXt-{model_config.variant}")
    print(f"Backbone: ConvNeXt-Base (现代化卷积网络)")
    print(f"输入通道: 4 (RGB + Target Mask)")
    print(f"可训练参数: {n_params:.2f}M")
    print(f"总参数: {n_total_params:.2f}M")
    
    return model


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: Config,
    scaler: Optional[GradScaler] = None,
    epoch: int = 0,
) -> Dict[str, float]:
    """训练一个 epoch"""
    model.train()
    
    metric_calc = MetricCalculator()
    loss_meter = AverageMeter('loss')
    cls_loss_meter = AverageMeter('cls_loss')
    seg_loss_meter = AverageMeter('seg_loss')
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]', leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        # 准备数据
        inputs = batch['input'].to(device)
        cls_targets = batch['stability_label'].to(device)
        seg_targets = batch['affected_mask'].to(device)
        
        optimizer.zero_grad()
        
        # 前向传播 (混合精度)
        if config.train.use_amp and scaler is not None:
            with autocast(device_type='cuda'):
                cls_logits, seg_logits = model(inputs)
            # 在 autocast 外部计算 loss，强制使用 FP32，避免数值不稳定
            loss, loss_dict = criterion(
                cls_logits.float(), 
                seg_logits.float(), 
                cls_targets.float(), 
                seg_targets.float()
            )
        else:
            cls_logits, seg_logits = model(inputs)
            loss, loss_dict = criterion(cls_logits, seg_logits, cls_targets, seg_targets)
            
        # 反向传播
        if config.train.use_amp and scaler is not None:
            scaler.scale(loss).backward()
            
            if config.train.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
                
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            
            if config.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
                
            optimizer.step()
            
        # 更新指标
        batch_size = inputs.size(0)
        loss_meter.update(loss_dict['total_loss'], batch_size)
        cls_loss_meter.update(loss_dict['cls_loss'], batch_size)
        seg_loss_meter.update(loss_dict['seg_loss'], batch_size)
        
        metric_calc.update(cls_logits, seg_logits, cls_targets, seg_targets)
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'cls': f'{cls_loss_meter.avg:.4f}',
            'seg': f'{seg_loss_meter.avg:.4f}',
        })
        
    # 计算 epoch 指标
    metrics = metric_calc.compute()
    metrics['loss'] = loss_meter.avg
    metrics['cls_loss'] = cls_loss_meter.avg
    metrics['seg_loss'] = seg_loss_meter.avg
    
    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: Config,
) -> Dict[str, float]:
    """验证"""
    model.eval()
    
    metric_calc = MetricCalculator()
    loss_meter = AverageMeter('loss')
    cls_loss_meter = AverageMeter('cls_loss')
    seg_loss_meter = AverageMeter('seg_loss')
    
    pbar = tqdm(val_loader, desc='[Validate]', leave=False)
    
    for batch in pbar:
        inputs = batch['input'].to(device)
        cls_targets = batch['stability_label'].to(device)
        seg_targets = batch['affected_mask'].to(device)
        
        # 前向传播
        cls_logits, seg_logits = model(inputs)
        loss, loss_dict = criterion(cls_logits, seg_logits, cls_targets, seg_targets)
        
        # 更新指标
        batch_size = inputs.size(0)
        loss_meter.update(loss_dict['total_loss'], batch_size)
        cls_loss_meter.update(loss_dict['cls_loss'], batch_size)
        seg_loss_meter.update(loss_dict['seg_loss'], batch_size)
        
        metric_calc.update(cls_logits, seg_logits, cls_targets, seg_targets)
        
    # 计算指标
    metrics = metric_calc.compute()
    metrics['loss'] = loss_meter.avg
    metrics['cls_loss'] = cls_loss_meter.avg
    metrics['seg_loss'] = seg_loss_meter.avg
    
    return metrics


def train(config: Config):
    """主训练函数"""
    # 设置随机种子
    set_seed(config.train.seed)
    
    # 获取设备
    device = get_device(config.device)
    
    # 创建数据加载器
    print("\n" + "=" * 60)
    print("加载数据...")
    print("=" * 60)
    
    # 获取完整路径
    train_dirs = config.data.get_full_paths(config.data.train_dirs)
    val_dirs = config.data.get_full_paths(config.data.val_dirs) if config.data.val_dirs else None
    test_dirs = config.data.get_full_paths(config.data.test_dirs) if config.data.test_dirs else None
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dirs=train_dirs,
        val_dirs=val_dirs,
        test_dirs=test_dirs,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        img_size=(config.data.img_height, config.data.img_width),
        val_split=config.data.val_split,
        use_weighted_sampler=config.data.use_weighted_sampler,
        seed=config.train.seed,
    )
    
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    if test_loader is not None:
        print(f"测试批次数: {len(test_loader)}")
    
    # 创建模型 (ConvNeXt 版本)
    print("\n" + "=" * 60)
    print("创建模型 (ConvNeXt Backbone)...")
    print("=" * 60)
    
    model = create_model_convnext(config, device)
    
    # 创建损失函数
    criterion = CombinedLoss(
        cls_loss=config.loss.cls_loss,
        seg_loss=config.loss.seg_loss,
        cls_weight=config.model.cls_weight,
        seg_weight=config.model.seg_weight,
        pos_weight=config.loss.pos_weight,
        focal_gamma=config.loss.focal_gamma,
        focal_alpha=config.loss.focal_alpha,
        dice_smooth=config.loss.dice_smooth,
    )
    criterion.cls_criterion = criterion.cls_criterion.to(device)
    
    # 创建优化器和调度器
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # 混合精度训练
    scaler = GradScaler() if config.train.use_amp else None
    
    # 早停
    early_stopping = EarlyStopping(
        patience=config.train.early_stopping_patience,
        mode='max',
    )
    
    # 检查点目录
    save_dir = Path(config.train.save_dir) / config.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"模型检查点保存目录: {save_dir}")
    
    # 恢复训练
    start_epoch = 1
    resumed_best_f1 = 0.0
    if hasattr(config, 'resume_path') and config.resume_path:
        print(f"\n从检查点恢复: {config.resume_path}")
        checkpoint = load_checkpoint(
            model, config.resume_path, optimizer, scheduler, config.device
        )
        start_epoch = checkpoint.get('epoch', 0) + 1
        if 'metrics' in checkpoint and 'val' in checkpoint['metrics']:
            resumed_best_f1 = checkpoint['metrics']['val'].get('cls_f1', 0.0)
            print(f"恢复到 epoch {start_epoch - 1}, 之前最佳 F1: {resumed_best_f1:.4f}")
    
    # WandB 日志
    if config.train.use_wandb:
        try:
            import wandb
            wandb.init(
                project=config.train.wandb_project,
                name=config.train.wandb_run_name or config.exp_name,
                config={
                    'data': config.data.__dict__,
                    'model': config.model.__dict__,
                    'train': config.train.__dict__,
                    'loss': config.loss.__dict__,
                    'backbone': 'ConvNeXt-Base',
                    'input_channels': 4,
                },
            )
        except ImportError:
            print("警告: wandb 未安装，跳过日志记录")
            config.train.use_wandb = False
            
    # 训练循环
    print("\n" + "=" * 60)
    print("开始训练 (ConvNeXt Backbone)...")
    print("=" * 60)
    
    best_f1 = resumed_best_f1
    best_epoch = start_epoch - 1 if resumed_best_f1 > 0 else 0
    
    for epoch in range(start_epoch, config.train.epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}/{config.train.epochs} (lr: {current_lr:.2e})")
        
        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            config, scaler, epoch
        )
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, device, config)
        
        # 更新学习率
        if config.train.scheduler == 'plateau':
            scheduler.step(val_metrics['loss'])
        else:
            scheduler.step()
            
        # 打印结果
        print(f"\n训练: loss={train_metrics['loss']:.4f}, "
              f"acc={train_metrics['cls_accuracy']:.4f}, "
              f"f1={train_metrics['cls_f1']:.4f}, "
              f"iou={train_metrics['seg_iou']:.4f}")
        print(f"验证: loss={val_metrics['loss']:.4f}, "
              f"acc={val_metrics['cls_accuracy']:.4f}, "
              f"f1={val_metrics['cls_f1']:.4f}, "
              f"iou={val_metrics['seg_iou']:.4f}")
        
        # WandB 日志
        if config.train.use_wandb:
            import wandb
            wandb.log({
                'epoch': epoch,
                'lr': current_lr,
                **{f'train/{k}': v for k, v in train_metrics.items()},
                **{f'val/{k}': v for k, v in val_metrics.items()},
            })
            
        # 保存检查点
        is_best = val_metrics['cls_f1'] > best_f1
        if is_best:
            best_f1 = val_metrics['cls_f1']
            best_epoch = epoch
            
        if is_best or (epoch % config.train.save_freq == 0):
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch}.pth'
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'train': train_metrics, 'val': val_metrics},
                str(checkpoint_path),
                is_best=is_best,
            )
            if is_best:
                print(f"  ✓ 保存最佳模型 (F1: {best_f1:.4f})")
                
        # 早停检查
        if early_stopping(val_metrics['cls_f1']):
            print(f"\n早停触发! 最佳 epoch: {best_epoch}, 最佳 F1: {best_f1:.4f}")
            break
            
    print("\n" + "=" * 60)
    print(f"训练完成! 最佳验证 F1: {best_f1:.4f} (Epoch {best_epoch})")
    print(f"模型保存于: {save_dir}")
    print("=" * 60)
    
    # 测试集评估
    if test_loader is not None:
        print("\n" + "=" * 60)
        print("测试集评估...")
        print("=" * 60)
        
        # 加载最佳模型
        best_checkpoint_path = save_dir / f'checkpoint_epoch_{best_epoch}.pth'
        if best_checkpoint_path.exists():
            print(f"加载最佳模型: {best_checkpoint_path}")
            load_checkpoint(model, str(best_checkpoint_path), None, None, config.device)
        
        test_metrics = validate(model, test_loader, criterion, device, config)
        
        print(f"\n测试集结果:")
        print(f"  loss: {test_metrics['loss']:.4f}")
        print(f"  accuracy: {test_metrics['cls_accuracy']:.4f}")
        print(f"  F1: {test_metrics['cls_f1']:.4f}")
        print(f"  IoU: {test_metrics['seg_iou']:.4f}")
        
        if config.train.use_wandb:
            import wandb
            wandb.log({f'test/{k}': v for k, v in test_metrics.items()})
    
    if config.train.use_wandb:
        import wandb
        wandb.finish()
        
    return best_f1


def get_config_convnext(preset: str = 'default') -> Config:
    """获取 ConvNeXt 预设配置"""
    
    # 统一的保存目录
    CHECKPOINTS_DIR = '/DATA/disk0/hs_25/pa/checkpoints'
    
    if preset == 'default':
        return Config(
            data=DataConfig(
                data_root='/DATA/disk0/hs_25/pa',
                train_dirs=['all_dataset/train'],
                val_dirs=['all_dataset/val'],
                test_dirs=['all_dataset/test'],
            ),
            train=TrainConfig(
                save_dir=CHECKPOINTS_DIR,
            ),
            exp_name='transunet_convnext',
        )
    
    elif preset == 'debug':
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
                save_dir=CHECKPOINTS_DIR,
            ),
            exp_name='debug_convnext',
        )
    
    elif preset == 'small':
        return Config(
            model=ModelConfig(
                variant='small',
            ),
            train=TrainConfig(
                epochs=100,
                lr=1e-4,
                save_dir=CHECKPOINTS_DIR,
            ),
            exp_name='transunet_convnext_small',
        )
    
    elif preset == 'base':
        return Config(
            data=DataConfig(
                batch_size=4,
            ),
            model=ModelConfig(
                variant='base',
            ),
            train=TrainConfig(
                epochs=150,
                lr=5e-5,
                grad_clip=0.5,
                save_dir=CHECKPOINTS_DIR,
            ),
            exp_name='transunet_convnext_base',
        )
    
    elif preset == 'full_data':
        return Config(
            data=DataConfig(
                data_root='/DATA/disk0/hs_25/pa',
                train_dirs=['all_dataset/train'],
                val_dirs=['all_dataset/val'],
                test_dirs=['all_dataset/test'],
                batch_size=8,
            ),
            model=ModelConfig(
                variant='small',
            ),
            train=TrainConfig(
                epochs=100,
                save_dir=CHECKPOINTS_DIR,
            ),
            exp_name='transunet_convnext_full',
        )
    
    else:
        raise ValueError(f"未知的预设配置: {preset}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='TransUNet ConvNeXt 训练脚本')
    
    # 配置预设
    parser.add_argument('--config', type=str, default='default',
                        choices=['default', 'debug', 'small', 'base', 'full_data'],
                        help='配置预设')
    
    # 覆盖参数
    parser.add_argument('--exp_name', type=str, default=None, help='实验名称')
    parser.add_argument('--data_dirs', type=str, nargs='+', default=None, help='数据目录')
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--model', type=str, default=None, 
                        choices=['tiny', 'small', 'base'], help='模型变体')
    parser.add_argument('--device', type=str, default=None, help='设备 (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=None, help='数据加载线程数')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--wandb', action='store_true', help='启用 WandB 日志')
    parser.add_argument('--no_amp', action='store_true', help='禁用混合精度训练')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 加载 ConvNeXt 专用配置
    config = get_config_convnext(args.config)
    
    # 覆盖配置
    if args.exp_name:
        config.exp_name = args.exp_name
    if args.data_dirs:
        config.data.train_dirs = args.data_dirs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.epochs:
        config.train.epochs = args.epochs
    if args.lr:
        config.train.lr = args.lr
    if args.model:
        config.model.variant = args.model
    if args.device:
        config.device = args.device
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    if args.wandb:
        config.train.use_wandb = True
    if args.no_amp:
        config.train.use_amp = False
    if args.resume:
        config.resume_path = args.resume
        
    # 打印配置
    print("\n" + "=" * 60)
    print("训练配置 (ConvNeXt Backbone)")
    print("=" * 60)
    print(f"实验名称: {config.exp_name}")
    print(f"Backbone: ConvNeXt-Base (现代化卷积网络)")
    print(f"输入类型: RGB + Mask (4 通道)")
    print(f"数据根目录: {config.data.data_root}")
    print(f"训练集: {config.data.train_dirs}")
    print(f"验证集: {config.data.val_dirs if config.data.val_dirs else '从训练集划分'}")
    print(f"测试集: {config.data.test_dirs if config.data.test_dirs else '无'}")
    print(f"模型: TransUNet-ConvNeXt-{config.model.variant}")
    print(f"批次大小: {config.data.batch_size}")
    print(f"训练轮数: {config.train.epochs}")
    print(f"学习率: {config.train.lr}")
    print(f"设备: {config.device}")
    print(f"混合精度: {config.train.use_amp}")
    
    # 开始训练
    train(config)


if __name__ == '__main__':
    main()
