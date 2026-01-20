"""
TransUNet ResNet-50 RGBD 仅分类训练脚本
用于多任务学习消融实验 - 仅分类任务 (带深度图)

用法:
    python train_rgbd_cls_only.py --config base
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

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train.config import Config, DataConfig, ModelConfig, TrainConfig, LossConfig
from train.dataset_with_depth import create_dataloaders_with_depth
from train.models.transunet_rgbd_cls_only import (
    TransUNetRGBDClsOnly,
    transunet_rgbd_cls_only_base,
    transunet_rgbd_cls_only_small,
    transunet_rgbd_cls_only_tiny,
)
from train.utils import (
    set_seed, get_device, create_optimizer, create_scheduler,
    save_checkpoint, load_checkpoint, AverageMeter, EarlyStopping,
)


class ClassificationOnlyLoss(nn.Module):
    def __init__(self, pos_weight: float = 2.0):
        super().__init__()
        self.cls_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        
    def forward(self, cls_logits, cls_targets):
        cls_logits = cls_logits.squeeze(-1)
        cls_targets = cls_targets.float()
        loss = self.cls_criterion(cls_logits, cls_targets)
        return loss, {'total_loss': loss.item(), 'cls_loss': loss.item()}


class ClassificationMetricCalculator:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.cls_preds = []
        self.cls_targets = []
        
    def update(self, cls_logits, cls_targets):
        cls_preds = (torch.sigmoid(cls_logits.squeeze(-1)) > 0.5).float()
        self.cls_preds.append(cls_preds.cpu())
        self.cls_targets.append(cls_targets.cpu())
        
    def compute(self):
        cls_preds = torch.cat(self.cls_preds)
        cls_targets = torch.cat(self.cls_targets)
        
        accuracy = (cls_preds == cls_targets).sum().item() / cls_targets.numel()
        
        tp = ((cls_preds == 1) & (cls_targets == 1)).sum().item()
        fp = ((cls_preds == 1) & (cls_targets == 0)).sum().item()
        fn = ((cls_preds == 0) & (cls_targets == 1)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        self.reset()
        return {'cls_accuracy': accuracy, 'cls_precision': precision, 'cls_recall': recall, 'cls_f1': f1}


def create_model(config: Config, device: torch.device) -> nn.Module:
    model_config = config.model
    
    if model_config.variant == 'tiny':
        model = transunet_rgbd_cls_only_tiny(pretrained=model_config.pretrained,
            img_height=config.data.img_height, img_width=config.data.img_width, dropout=model_config.dropout)
    elif model_config.variant == 'small':
        model = transunet_rgbd_cls_only_small(pretrained=model_config.pretrained,
            img_height=config.data.img_height, img_width=config.data.img_width, dropout=model_config.dropout)
    else:
        model = transunet_rgbd_cls_only_base(pretrained=model_config.pretrained,
            img_height=config.data.img_height, img_width=config.data.img_width, dropout=model_config.dropout)
        
    model = model.to(device)
    print(f"模型: TransUNet-ResNet50-RGBD-ClsOnly-{model_config.variant}")
    print(f"任务: 仅分类 (无分割头)")
    print(f"输入: 5 通道 (RGB + Depth + Mask)")
    print(f"参数量: {model.get_num_params() / 1e6:.2f}M")
    return model


def train_one_epoch(model, train_loader, criterion, optimizer, device, config, scaler=None, epoch=0):
    model.train()
    metric_calc = ClassificationMetricCalculator()
    loss_meter = AverageMeter('loss')
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]', leave=False)
    for batch in pbar:
        inputs = batch['input'].to(device)
        cls_targets = batch['stability_label'].to(device)
        
        optimizer.zero_grad()
        
        if config.train.use_amp and scaler:
            with autocast(device_type='cuda'):
                cls_logits = model(inputs)
                loss, loss_dict = criterion(cls_logits, cls_targets)
        else:
            cls_logits = model(inputs)
            loss, loss_dict = criterion(cls_logits, cls_targets)
            
        if config.train.use_amp and scaler:
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
            
        loss_meter.update(loss_dict['total_loss'], inputs.size(0))
        metric_calc.update(cls_logits, cls_targets)
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
        
    metrics = metric_calc.compute()
    metrics['loss'] = loss_meter.avg
    return metrics


@torch.no_grad()
def validate(model, val_loader, criterion, device, config):
    model.eval()
    metric_calc = ClassificationMetricCalculator()
    loss_meter = AverageMeter('loss')
    
    for batch in tqdm(val_loader, desc='[Validate]', leave=False):
        inputs = batch['input'].to(device)
        cls_targets = batch['stability_label'].to(device)
        
        cls_logits = model(inputs)
        loss, loss_dict = criterion(cls_logits, cls_targets)
        
        loss_meter.update(loss_dict['total_loss'], inputs.size(0))
        metric_calc.update(cls_logits, cls_targets)
        
    metrics = metric_calc.compute()
    metrics['loss'] = loss_meter.avg
    return metrics


def train(config: Config):
    set_seed(config.train.seed)
    device = get_device(config.device)
    
    train_dirs = config.data.get_full_paths(config.data.train_dirs)
    val_dirs = config.data.get_full_paths(config.data.val_dirs) if config.data.val_dirs else None
    test_dirs = config.data.get_full_paths(config.data.test_dirs) if config.data.test_dirs else None
    
    train_loader, val_loader, test_loader = create_dataloaders_with_depth(
        train_dirs=train_dirs, val_dirs=val_dirs, test_dirs=test_dirs,
        batch_size=config.data.batch_size, num_workers=config.data.num_workers,
        img_size=(config.data.img_height, config.data.img_width),
        val_split=config.data.val_split, use_weighted_sampler=config.data.use_weighted_sampler,
        seed=config.train.seed)
    
    print(f"训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")
    
    model = create_model(config, device)
    criterion = ClassificationOnlyLoss(pos_weight=config.loss.pos_weight)
    criterion.cls_criterion = criterion.cls_criterion.to(device)
    
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    scaler = GradScaler() if config.train.use_amp else None
    early_stopping = EarlyStopping(patience=config.train.early_stopping_patience, mode='max')
    
    save_dir = Path(config.train.save_dir) / config.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    start_epoch, best_f1 = 1, 0.0
    if hasattr(config, 'resume_path') and config.resume_path:
        checkpoint = load_checkpoint(model, config.resume_path, optimizer, scheduler, config.device)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_f1 = checkpoint.get('metrics', {}).get('val', {}).get('cls_f1', 0.0)
    
    best_epoch = 0
    for epoch in range(start_epoch, config.train.epochs + 1):
        print(f"\nEpoch {epoch}/{config.train.epochs} (lr: {optimizer.param_groups[0]['lr']:.2e})")
        
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, config, scaler, epoch)
        val_metrics = validate(model, val_loader, criterion, device, config)
        
        if config.train.scheduler == 'plateau':
            scheduler.step(val_metrics['loss'])
        else:
            scheduler.step()
            
        print(f"训练: loss={train_metrics['loss']:.4f}, acc={train_metrics['cls_accuracy']:.4f}, f1={train_metrics['cls_f1']:.4f}")
        print(f"验证: loss={val_metrics['loss']:.4f}, acc={val_metrics['cls_accuracy']:.4f}, f1={val_metrics['cls_f1']:.4f}")
        
        is_best = val_metrics['cls_f1'] > best_f1
        if is_best:
            best_f1 = val_metrics['cls_f1']
            best_epoch = epoch
            
        if is_best or (epoch % config.train.save_freq == 0):
            save_checkpoint(model, optimizer, scheduler, epoch,
                {'train': train_metrics, 'val': val_metrics},
                str(save_dir / f'checkpoint_epoch_{epoch}.pth'), is_best=is_best)
            if is_best:
                print(f"  ✓ 最佳模型 (F1: {best_f1:.4f})")
                
        if early_stopping(val_metrics['cls_f1']):
            print(f"\n早停! 最佳: epoch {best_epoch}, F1: {best_f1:.4f}")
            break
            
    print(f"\n训练完成! 最佳 F1: {best_f1:.4f} (Epoch {best_epoch})")
    
    if test_loader:
        best_ckpt = save_dir / f'checkpoint_epoch_{best_epoch}.pth'
        if best_ckpt.exists():
            load_checkpoint(model, str(best_ckpt), None, None, config.device)
        test_metrics = validate(model, test_loader, criterion, device, config)
        print(f"测试集: acc={test_metrics['cls_accuracy']:.4f}, f1={test_metrics['cls_f1']:.4f}")


def get_config(preset: str = 'default') -> Config:
    CHECKPOINTS_DIR = '/DATA/disk0/hs_25/pa/checkpoints'
    
    if preset == 'debug':
        return Config(
            data=DataConfig(batch_size=2, num_workers=0),
            model=ModelConfig(variant='tiny'),
            train=TrainConfig(epochs=5, lr=1e-4, use_amp=False, save_dir=CHECKPOINTS_DIR),
            exp_name='debug_resnet_rgbd_cls_only')
    elif preset == 'base':
        return Config(
            data=DataConfig(batch_size=4),
            model=ModelConfig(variant='base'),
            train=TrainConfig(epochs=150, lr=5e-5, grad_clip=0.5, save_dir=CHECKPOINTS_DIR),
            exp_name='transunet_rgbd_cls_only_base')
    else:
        return Config(
            data=DataConfig(data_root='/DATA/disk0/hs_25/pa',
                train_dirs=['all_dataset/train'], val_dirs=['all_dataset/val'], test_dirs=['all_dataset/test']),
            train=TrainConfig(save_dir=CHECKPOINTS_DIR),
            exp_name='transunet_rgbd_cls_only')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='default', choices=['default', 'debug', 'base'])
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--model', type=str, default=None, choices=['tiny', 'small', 'base'])
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no_amp', action='store_true')
    args = parser.parse_args()
    
    config = get_config(args.config)
    if args.exp_name: config.exp_name = args.exp_name
    if args.batch_size: config.data.batch_size = args.batch_size
    if args.epochs: config.train.epochs = args.epochs
    if args.lr: config.train.lr = args.lr
    if args.model: config.model.variant = args.model
    if args.num_workers is not None: config.data.num_workers = args.num_workers
    if args.no_amp: config.train.use_amp = False
    if args.resume: config.resume_path = args.resume
    
    print(f"\n{'='*60}\n消融实验: ResNet-50 RGBD 仅分类\n{'='*60}")
    print(f"模型: TransUNet-ResNet50-RGBD-ClsOnly-{config.model.variant}")
    train(config)


if __name__ == '__main__':
    main()
