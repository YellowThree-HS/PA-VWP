"""
TransUNet Fusion 训练脚本 - 分割感知双路融合

用法:
    python train_fusion.py --config base
    python train_fusion.py --config debug
    python train_fusion.py --config base --epochs 200 --lr 1e-4

与原版 train.py 的区别:
- 使用 TransUNetFusion 模型 (分割特征增强分类)
- 分割掩码引导的注意力池化
- 分割解码器多尺度特征融合
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

from train.config import Config, get_config
from train.dataset import BoxWorldDataset, create_dataloaders, get_train_transform, get_val_transform
from train.models.transunet_fusion import (
    TransUNetFusion, 
    TransUNetFusionConfig,
    transunet_fusion_base, 
    transunet_fusion_small,
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


def create_model_fusion(config: Config, device: torch.device) -> nn.Module:
    """创建融合模型"""
    model_config = config.model
    
    if model_config.variant == 'small':
        model = transunet_fusion_small(
            pretrained=model_config.pretrained,
            img_height=config.data.img_height,
            img_width=config.data.img_width,
            dropout=model_config.dropout,
        )
    elif model_config.variant == 'base':
        model = transunet_fusion_base(
            pretrained=model_config.pretrained,
            img_height=config.data.img_height,
            img_width=config.data.img_width,
            dropout=model_config.dropout,
        )
    else:
        # 自定义配置或默认 base
        transunet_config = TransUNetFusionConfig(
            img_height=config.data.img_height,
            img_width=config.data.img_width,
            pretrained=model_config.pretrained,
            hidden_dim=model_config.hidden_dim,
            num_heads=model_config.num_heads,
            num_layers=model_config.num_layers,
            mlp_dim=model_config.mlp_dim,
            dropout=model_config.dropout,
            use_mask_guided=True,
            use_decoder_features=True,
        )
        model = TransUNetFusion(transunet_config)
        
    model = model.to(device)
    
    # 打印模型信息
    n_params = model.get_num_params() / 1e6
    n_total_params = model.get_num_params(trainable_only=False) / 1e6
    print(f"模型: TransUNet-Fusion-{model_config.variant}")
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
    
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    seg_loss_meter = AverageMeter()
    metric_calc = MetricCalculator()
    
    pbar = tqdm(train_loader, desc=f'Train Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        # 获取数据
        images = batch['input'].to(device)
        cls_targets = batch['stability_label'].to(device)
        seg_targets = batch['affected_mask'].to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        if config.train.use_amp and scaler is not None:
            with autocast(device_type='cuda'):
                cls_logits, seg_logits = model(images)
                loss, loss_dict = criterion(
                    cls_logits, seg_logits, 
                    cls_targets, seg_targets
                )
        else:
            cls_logits, seg_logits = model(images)
            loss, loss_dict = criterion(
                cls_logits, seg_logits, 
                cls_targets, seg_targets
            )
            
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
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        cls_loss_meter.update(loss_dict['cls_loss'], batch_size)
        seg_loss_meter.update(loss_dict['seg_loss'], batch_size)
        
        # 计算指标
        metric_calc.update(cls_logits, seg_logits, cls_targets, seg_targets)
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'cls': f'{cls_loss_meter.avg:.4f}',
            'seg': f'{seg_loss_meter.avg:.4f}',
        })
        
    metrics = metric_calc.compute()
    
    return {
        'loss': loss_meter.avg,
        'cls_loss': cls_loss_meter.avg,
        'seg_loss': seg_loss_meter.avg,
        **metrics
    }


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
    
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    seg_loss_meter = AverageMeter()
    metric_calc = MetricCalculator()
    
    pbar = tqdm(val_loader, desc='Validate')
    
    for batch in pbar:
        images = batch['input'].to(device)
        cls_targets = batch['stability_label'].to(device)
        seg_targets = batch['affected_mask'].to(device)
        
        if config.train.use_amp:
            with autocast(device_type='cuda'):
                cls_logits, seg_logits = model(images)
                loss, loss_dict = criterion(
                    cls_logits, seg_logits, 
                    cls_targets, seg_targets
                )
        else:
            cls_logits, seg_logits = model(images)
            loss, loss_dict = criterion(
                cls_logits, seg_logits, 
                cls_targets, seg_targets
            )
            
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        cls_loss_meter.update(loss_dict['cls_loss'], batch_size)
        seg_loss_meter.update(loss_dict['seg_loss'], batch_size)
        
        # 计算指标
        metric_calc.update(cls_logits, seg_logits, cls_targets, seg_targets)
        
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
        
    metrics = metric_calc.compute()
    
    return {
        'loss': loss_meter.avg,
        'cls_loss': cls_loss_meter.avg,
        'seg_loss': seg_loss_meter.avg,
        **metrics
    }


def train(config: Config):
    """训练主函数"""
    # 设置随机种子
    set_seed(config.train.seed)
    
    # 设置设备
    device = get_device(config.device)
    print(f"使用设备: {device}")
    
    # 获取完整路径
    train_dirs = config.data.get_full_paths(config.data.train_dirs)
    val_dirs = config.data.get_full_paths(config.data.val_dirs) if config.data.val_dirs else None
    test_dirs = config.data.get_full_paths(config.data.test_dirs) if config.data.test_dirs else None
    
    # 创建数据加载器
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
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")
    
    # 创建模型 - 使用融合版本
    model = create_model_fusion(config, device)
    
    # 创建损失函数
    criterion = CombinedLoss(
        cls_loss=config.loss.cls_loss,
        seg_loss=config.loss.seg_loss,
        cls_weight=config.model.cls_weight,
        seg_weight=config.model.seg_weight,
        pos_weight=config.loss.pos_weight,
    )
    
    # 创建优化器和调度器
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # 混合精度
    scaler = GradScaler() if config.train.use_amp else None
    
    # 早停
    early_stopping = EarlyStopping(
        patience=config.train.early_stopping_patience,
        mode='max',
    )
    
    # 恢复训练
    start_epoch = 0
    best_f1 = 0.0
    
    if hasattr(config, 'resume_path') and config.resume_path:
        checkpoint = load_checkpoint(
            model, config.resume_path, optimizer, scheduler, config.device
        )
        start_epoch = checkpoint.get('epoch', 0) + 1
        if 'metrics' in checkpoint and 'val' in checkpoint['metrics']:
            best_f1 = checkpoint['metrics']['val'].get('f1', 0.0)
        else:
            best_f1 = checkpoint.get('best_f1', 0.0)
        print(f"从 epoch {start_epoch} 恢复训练，最佳 F1: {best_f1:.4f}")
    
    # 创建保存目录
    save_dir = Path(config.train.save_dir) / config.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练循环
    for epoch in range(start_epoch, config.train.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.train.epochs}")
        print(f"{'='*60}")
        
        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config, scaler, epoch + 1
        )
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, device, config)
        
        # 更新学习率
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['cls_f1'])
            else:
                scheduler.step()
                
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印指标
        print(f"\n训练: Loss={train_metrics['loss']:.4f}, F1={train_metrics['cls_f1']:.4f}")
        print(f"验证: Loss={val_metrics['loss']:.4f}, F1={val_metrics['cls_f1']:.4f}")
        
        # 保存最佳模型
        is_best = val_metrics['cls_f1'] > best_f1
        if is_best:
            best_f1 = val_metrics['cls_f1']
            print(f"✨ 新的最佳 F1: {best_f1:.4f}")
            
        # 保存检查点
        checkpoint_path = save_dir / f'checkpoint_epoch_{epoch + 1}.pth'
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            metrics={'train': train_metrics, 'val': val_metrics},
            save_path=str(checkpoint_path),
            is_best=is_best,
        )
        
        # 早停检查
        early_stopping(val_metrics['cls_f1'])
        if early_stopping.early_stop:
            print(f"\n早停触发，训练结束。最佳 F1: {best_f1:.4f}")
            break
            
    print(f"\n训练完成！最佳 F1: {best_f1:.4f}")
    print(f"模型保存在: {save_dir}")
    
    return best_f1


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='TransUNet Fusion 训练脚本 (分割感知双路融合)')
    
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
                        choices=['small', 'base'], help='模型变体 (fusion 只支持 small/base)')
    parser.add_argument('--device', type=str, default=None, help='设备 (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=None, help='数据加载线程数')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--wandb', action='store_true', help='启用 WandB 日志')
    parser.add_argument('--no_amp', action='store_true', help='禁用混合精度训练')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    config = get_config(args.config)
    
    # 覆盖配置
    if args.exp_name:
        config.exp_name = args.exp_name
    else:
        # 默认实验名称
        config.exp_name = f'transunet_fusion_{args.config}'
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
    print("TransUNet Fusion 训练配置")
    print("=" * 60)
    print(f"实验名称: {config.exp_name}")
    print(f"数据根目录: {config.data.data_root}")
    print(f"训练集: {config.data.train_dirs}")
    print(f"验证集: {config.data.val_dirs if config.data.val_dirs else '从训练集划分'}")
    print(f"模型: TransUNet-Fusion-{config.model.variant}")
    print(f"融合策略: Mask-Guided Attention + Multi-Scale Decoder Features")
    print(f"批次大小: {config.data.batch_size}")
    print(f"训练轮数: {config.train.epochs}")
    print(f"学习率: {config.train.lr}")
    print(f"设备: {config.device}")
    print(f"混合精度: {config.train.use_amp}")
    print("=" * 60)
    
    # 开始训练
    train(config)


if __name__ == '__main__':
    main()
