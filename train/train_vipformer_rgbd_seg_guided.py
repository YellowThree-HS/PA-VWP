"""
VIPFormer RGBD Seg-Guided 训练脚本 - 双流版本

用法:
    python train_vipformer_rgbd_seg_guided.py --config base
    python train_vipformer_rgbd_seg_guided.py --config debug
    python train_vipformer_rgbd_seg_guided.py --config base --epochs 200 --lr 1e-4

特点:
- 使用 VIPFormerRGBDSegGuided 模型 (双流分割引导分类)
- RGB+Mask 与 Depth 分开编码，特征级融合
- SEG Query Tokens 在 Transformer 中学习分割语义
- Cross-Attention Fusion: CLS token 聚合分割信息
- 输入: RGB (3) + Depth (1) + Target Mask (1) = 5 通道
"""

import argparse
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

from train.config import Config, DataConfig, ModelConfig, TrainConfig, LossConfig
from train.dataset_with_depth import create_dataloaders_with_depth
from train.models.vipformer_rgbd_seg_guided import (
    VIPFormerRGBDSegGuided,
    VIPFormerRGBDSegGuidedConfig,
    create_vipformer_rgbd_seg_guided,
    vipformer_rgbd_seg_guided_base,
    vipformer_rgbd_seg_guided_tiny,
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
    EarlyStopping,
)


def create_model_vipformer_rgbd(config: Config, device: torch.device) -> nn.Module:
    """创建 VIPFormer RGBD Seg-Guided 模型"""
    model_config = config.model

    fusion_type = getattr(model_config, 'fusion_type', 'cross_attention')
    depth_fusion_type = getattr(model_config, 'depth_fusion_type', 'concat')

    if model_config.variant == 'tiny':
        model = vipformer_rgbd_seg_guided_tiny(
            pretrained=model_config.pretrained,
            img_height=config.data.img_height,
            img_width=config.data.img_width,
            dropout=model_config.dropout,
            fusion_type=fusion_type,
            depth_fusion_type=depth_fusion_type,
        )
    elif model_config.variant == 'base':
        model = vipformer_rgbd_seg_guided_base(
            pretrained=model_config.pretrained,
            img_height=config.data.img_height,
            img_width=config.data.img_width,
            dropout=model_config.dropout,
            fusion_type=fusion_type,
            depth_fusion_type=depth_fusion_type,
        )
    else:
        # 自定义配置
        vipformer_config = VIPFormerRGBDSegGuidedConfig(
            img_height=config.data.img_height,
            img_width=config.data.img_width,
            pretrained=model_config.pretrained,
            hidden_dim=model_config.hidden_dim,
            num_heads=model_config.num_heads,
            num_layers=model_config.num_layers,
            mlp_dim=model_config.mlp_dim,
            dropout=model_config.dropout,
            fusion_type=fusion_type,
            depth_fusion_type=depth_fusion_type,
        )
        model = VIPFormerRGBDSegGuided(vipformer_config)

    model = model.to(device)

    n_params = model.get_num_params() / 1e6
    n_total_params = model.get_num_params(trainable_only=False) / 1e6
    print(f"模型: VIPFormer-RGBD-SegGuided-{model_config.variant}")
    print(f"融合类型: {fusion_type}, 深度融合: {depth_fusion_type}")
    print(f"可训练参数: {n_params:.2f}M, 总参数: {n_total_params:.2f}M")

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

    for batch in pbar:
        images = batch['input'].to(device)  # (B, 5, H, W)
        cls_targets = batch['stability_label'].to(device)
        seg_targets = batch['affected_mask'].to(device)

        optimizer.zero_grad()

        if config.train.use_amp and scaler is not None:
            with autocast(device_type='cuda'):
                cls_logits, seg_logits = model(images)
                loss, loss_dict = criterion(
                    cls_logits, seg_logits, cls_targets, seg_targets
                )
        else:
            cls_logits, seg_logits = model(images)
            loss, loss_dict = criterion(
                cls_logits, seg_logits, cls_targets, seg_targets
            )

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

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        cls_loss_meter.update(loss_dict['cls_loss'], batch_size)
        seg_loss_meter.update(loss_dict['seg_loss'], batch_size)
        metric_calc.update(cls_logits, seg_logits, cls_targets, seg_targets)

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
                    cls_logits, seg_logits, cls_targets, seg_targets
                )
        else:
            cls_logits, seg_logits = model(images)
            loss, loss_dict = criterion(
                cls_logits, seg_logits, cls_targets, seg_targets
            )

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        cls_loss_meter.update(loss_dict['cls_loss'], batch_size)
        seg_loss_meter.update(loss_dict['seg_loss'], batch_size)
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
    set_seed(config.train.seed)
    device = get_device(config.device)
    print(f"使用设备: {device}")

    # 获取完整路径
    train_dirs = config.data.get_full_paths(config.data.train_dirs)
    val_dirs = config.data.get_full_paths(config.data.val_dirs) if config.data.val_dirs else None
    test_dirs = config.data.get_full_paths(config.data.test_dirs) if config.data.test_dirs else None

    # 创建数据加载器 (RGBD)
    train_loader, val_loader, test_loader = create_dataloaders_with_depth(
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

    # 创建模型
    model = create_model_vipformer_rgbd(config, device)

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
    scaler = GradScaler() if config.train.use_amp else None

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

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config, scaler, epoch + 1
        )
        val_metrics = validate(model, val_loader, criterion, device, config)

        # 更新学习率
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['cls_f1'])
            else:
                scheduler.step()

        print(f"\n训练: Loss={train_metrics['loss']:.4f}, F1={train_metrics['cls_f1']:.4f}")
        print(f"验证: Loss={val_metrics['loss']:.4f}, F1={val_metrics['cls_f1']:.4f}")

        # 保存最佳模型
        is_best = val_metrics['cls_f1'] > best_f1
        if is_best:
            best_f1 = val_metrics['cls_f1']
            print(f"新的最佳 F1: {best_f1:.4f}")

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

        early_stopping(val_metrics['cls_f1'])
        if early_stopping.early_stop:
            print(f"\n早停触发，训练结束。最佳 F1: {best_f1:.4f}")
            break

    print(f"\n训练完成！最佳 F1: {best_f1:.4f}")
    print(f"模型保存在: {save_dir}")
    return best_f1


def get_config_vipformer_rgbd(preset: str = 'default') -> Config:
    """获取 VIPFormer RGBD 配置"""
    if preset == 'debug':
        return Config(
            exp_name='vipformer_rgbd_seg_guided_debug',
            data=DataConfig(
                data_root='/DATA/disk0/hs_25/pa',
                train_dirs=['all_dataset/train'],
                val_dirs=['all_dataset/val'],
                batch_size=2,
                num_workers=2,
            ),
            model=ModelConfig(variant='tiny'),
            train=TrainConfig(epochs=5, lr=1e-4),
            loss=LossConfig(),
        )
    elif preset == 'base':
        return Config(
            exp_name='vipformer_rgbd_seg_guided_base',
            data=DataConfig(
                data_root='/DATA/disk0/hs_25/pa',
                train_dirs=['all_dataset/train'],
                val_dirs=['all_dataset/val'],
                test_dirs=['all_dataset/test'],
                batch_size=4,
                num_workers=8,
            ),
            model=ModelConfig(variant='base'),
            train=TrainConfig(epochs=150, lr=5e-5),
            loss=LossConfig(),
        )
    else:  # default
        return Config(
            exp_name='vipformer_rgbd_seg_guided_default',
            data=DataConfig(
                data_root='/DATA/disk0/hs_25/pa',
                train_dirs=['all_dataset/train'],
                val_dirs=['all_dataset/val'],
                batch_size=8,
                num_workers=8,
            ),
            model=ModelConfig(variant='tiny'),
            train=TrainConfig(epochs=100, lr=1e-4),
            loss=LossConfig(),
        )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='VIPFormer RGBD Seg-Guided 训练')

    parser.add_argument('--config', type=str, default='default',
                        choices=['default', 'debug', 'base'])
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--model', type=str, default=None,
                        choices=['tiny', 'base'])
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no_amp', action='store_true')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    config = get_config_vipformer_rgbd(args.config)

    # 覆盖配置
    if args.exp_name:
        config.exp_name = args.exp_name
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.epochs:
        config.train.epochs = args.epochs
    if args.lr:
        config.train.lr = args.lr
    if args.save_dir:
        config.train.save_dir = args.save_dir
    if args.model:
        config.model.variant = args.model
    if args.device:
        config.device = args.device
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    if args.no_amp:
        config.train.use_amp = False
    if args.resume:
        config.resume_path = args.resume

    # 默认保存目录
    if getattr(config.train, 'save_dir', None) in (None, '', 'checkpoints'):
        config.train.save_dir = '/DATA/disk0/hs_25/pa/checkpoints'

    # 打印配置
    print("\n" + "=" * 60)
    print("VIPFormer RGBD Seg-Guided 训练配置")
    print("=" * 60)
    print(f"实验名称: {config.exp_name}")
    print(f"模型: VIPFormer-RGBD-SegGuided-{config.model.variant}")
    print(f"输入通道: 5 (RGB + Depth + Target Mask)")
    print(f"批次大小: {config.data.batch_size}")
    print(f"训练轮数: {config.train.epochs}")
    print(f"学习率: {config.train.lr}")
    print(f"保存目录: {config.train.save_dir}")
    print("=" * 60)

    train(config)


if __name__ == '__main__':
    main()
