"""
TransUNet ResNet-50 仅分割训练脚本
用于多任务学习消融实验 - 仅分割任务

用法:
    python train_seg_only.py --config base
    python train_seg_only.py --config debug
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
from train.dataset import create_dataloaders
from train.models.transunet_seg_only import (
    TransUNetSegOnly,
    transunet_seg_only_base,
    transunet_seg_only_small,
    transunet_seg_only_tiny,
)
from train.utils import (
    set_seed, get_device, create_optimizer, create_scheduler,
    save_checkpoint, load_checkpoint, AverageMeter, EarlyStopping,
)


class DiceLoss(nn.Module):
    """Dice Loss for segmentation (数值稳定版本)"""
    def __init__(self, smooth: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps
        
    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()
        
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        denominator = pred.sum() + target.sum() + self.smooth + self.eps
        dice = (2. * intersection + self.smooth) / denominator
        dice = torch.clamp(dice, min=self.eps, max=1.0 - self.eps)
        return 1 - dice


class SegmentationOnlyLoss(nn.Module):
    """仅分割损失函数 (BCE + Dice) - 只对不稳定样本计算 loss
    
    关键改进：
    1. 只对不稳定样本（有 affected 区域）计算分割 loss
    2. 增加正样本权重 (pos_weight) 处理像素级不平衡
    """
    def __init__(self, dice_smooth: float = 1.0, pos_weight: float = 10.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.dice = DiceLoss(smooth=dice_smooth)
        self.pos_weight = pos_weight
        
    def forward(self, seg_logits, seg_targets, stability_labels=None):
        """
        Args:
            seg_logits: (B, 1, H, W)
            seg_targets: (B, 1, H, W) 
            stability_labels: (B,) - 0=不稳定, 1=稳定
        """
        seg_logits = seg_logits.float()
        seg_targets = seg_targets.float()
        
        if stability_labels is not None:
            unstable_mask = (stability_labels == 0)
            n_unstable = unstable_mask.sum().item()
            
            if n_unstable > 0:
                seg_logits_unstable = seg_logits[unstable_mask]
                seg_targets_unstable = seg_targets[unstable_mask]
                
                if self.bce.pos_weight.device != seg_logits.device:
                    self.bce.pos_weight = self.bce.pos_weight.to(seg_logits.device)
                
                bce_loss = self.bce(seg_logits_unstable, seg_targets_unstable)
                dice_loss = self.dice(seg_logits_unstable, seg_targets_unstable)
                loss = bce_loss + dice_loss
            else:
                loss = torch.tensor(0.0, device=seg_logits.device, requires_grad=True)
        else:
            has_positive = (seg_targets.view(seg_targets.size(0), -1).sum(dim=1) > 0)
            n_positive = has_positive.sum().item()
            
            if n_positive > 0:
                seg_logits_pos = seg_logits[has_positive]
                seg_targets_pos = seg_targets[has_positive]
                
                if self.bce.pos_weight.device != seg_logits.device:
                    self.bce.pos_weight = self.bce.pos_weight.to(seg_logits.device)
                
                bce_loss = self.bce(seg_logits_pos, seg_targets_pos)
                dice_loss = self.dice(seg_logits_pos, seg_targets_pos)
                loss = bce_loss + dice_loss
            else:
                loss = torch.tensor(0.0, device=seg_logits.device, requires_grad=True)
        
        return loss, {'total_loss': loss.item(), 'seg_loss': loss.item()}


class SegmentationMetricCalculator:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.intersection = 0
        self.union = 0
        self.dice_sum = 0
        self.count = 0
        
    def update(self, seg_logits, seg_targets):
        seg_preds = (torch.sigmoid(seg_logits) > 0.5).float()
        intersection = (seg_preds * seg_targets).sum().item()
        union = seg_preds.sum().item() + seg_targets.sum().item() - intersection
        self.intersection += intersection
        self.union += union
        
        for i in range(seg_preds.size(0)):
            pred = seg_preds[i].view(-1)
            target = seg_targets[i].view(-1)
            inter = (pred * target).sum().item()
            dice = (2 * inter + 1) / (pred.sum().item() + target.sum().item() + 1)
            self.dice_sum += dice
            self.count += 1
        
    def compute(self):
        iou = self.intersection / (self.union + 1e-6)
        dice = self.dice_sum / (self.count + 1e-6)
        self.reset()
        return {'seg_iou': iou, 'seg_dice': dice}


def create_model(config: Config, device: torch.device) -> nn.Module:
    model_config = config.model
    
    if model_config.variant == 'tiny':
        model = transunet_seg_only_tiny(pretrained=model_config.pretrained,
            img_height=config.data.img_height, img_width=config.data.img_width, dropout=model_config.dropout)
    elif model_config.variant == 'small':
        model = transunet_seg_only_small(pretrained=model_config.pretrained,
            img_height=config.data.img_height, img_width=config.data.img_width, dropout=model_config.dropout)
    else:
        model = transunet_seg_only_base(pretrained=model_config.pretrained,
            img_height=config.data.img_height, img_width=config.data.img_width, dropout=model_config.dropout)
        
    model = model.to(device)
    print(f"模型: TransUNet-ResNet50-SegOnly-{model_config.variant}")
    print(f"任务: 仅分割 (无分类头)")
    print(f"参数量: {model.get_num_params() / 1e6:.2f}M")
    return model


def train_one_epoch(model, train_loader, criterion, optimizer, device, config, scaler=None, epoch=0):
    model.train()
    metric_calc = SegmentationMetricCalculator()
    loss_meter = AverageMeter('loss')
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]', leave=False)
    for batch in pbar:
        inputs = batch['input'].to(device)
        seg_targets = batch['affected_mask'].to(device)
        stability_labels = batch['stability_label'].to(device)
        
        optimizer.zero_grad()
        
        if config.train.use_amp and scaler:
            with autocast(device_type='cuda'):
                seg_logits = model(inputs)
            loss, loss_dict = criterion(seg_logits.float(), seg_targets.float(), stability_labels)
        else:
            seg_logits = model(inputs)
            loss, loss_dict = criterion(seg_logits, seg_targets, stability_labels)
        
        if loss.item() == 0:
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}', 'skip': 'stable'})
            continue
            
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
        unstable_mask = (stability_labels == 0)
        if unstable_mask.sum() > 0:
            metric_calc.update(seg_logits[unstable_mask], seg_targets[unstable_mask])
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
        
    metrics = metric_calc.compute()
    metrics['loss'] = loss_meter.avg
    return metrics


@torch.no_grad()
def validate(model, val_loader, criterion, device, config):
    model.eval()
    metric_calc = SegmentationMetricCalculator()
    loss_meter = AverageMeter('loss')
    n_unstable_batches = 0
    
    for batch in tqdm(val_loader, desc='[Validate]', leave=False):
        inputs = batch['input'].to(device)
        seg_targets = batch['affected_mask'].to(device)
        stability_labels = batch['stability_label'].to(device)
        
        seg_logits = model(inputs)
        loss, loss_dict = criterion(seg_logits, seg_targets, stability_labels)
        
        unstable_mask = (stability_labels == 0)
        if unstable_mask.sum() > 0:
            loss_meter.update(loss_dict['total_loss'], unstable_mask.sum().item())
            metric_calc.update(seg_logits[unstable_mask], seg_targets[unstable_mask])
            n_unstable_batches += 1
        
    metrics = metric_calc.compute()
    metrics['loss'] = loss_meter.avg if n_unstable_batches > 0 else 0.0
    return metrics


def train(config: Config):
    set_seed(config.train.seed)
    device = get_device(config.device)
    
    train_dirs = config.data.get_full_paths(config.data.train_dirs)
    val_dirs = config.data.get_full_paths(config.data.val_dirs) if config.data.val_dirs else None
    test_dirs = config.data.get_full_paths(config.data.test_dirs) if config.data.test_dirs else None
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dirs=train_dirs, val_dirs=val_dirs, test_dirs=test_dirs,
        batch_size=config.data.batch_size, num_workers=config.data.num_workers,
        img_size=(config.data.img_height, config.data.img_width),
        val_split=config.data.val_split, use_weighted_sampler=config.data.use_weighted_sampler,
        seed=config.train.seed)
    
    print(f"训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")
    
    model = create_model(config, device)
    criterion = SegmentationOnlyLoss(dice_smooth=config.loss.dice_smooth)
    
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    scaler = GradScaler() if config.train.use_amp else None
    early_stopping = EarlyStopping(patience=config.train.early_stopping_patience, mode='max')
    
    save_dir = Path(config.train.save_dir) / config.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    start_epoch, best_iou = 1, 0.0
    if hasattr(config, 'resume_path') and config.resume_path:
        checkpoint = load_checkpoint(model, config.resume_path, optimizer, scheduler, config.device)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_iou = checkpoint.get('metrics', {}).get('val', {}).get('seg_iou', 0.0)
    
    best_epoch = 0
    for epoch in range(start_epoch, config.train.epochs + 1):
        print(f"\nEpoch {epoch}/{config.train.epochs} (lr: {optimizer.param_groups[0]['lr']:.2e})")
        
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, config, scaler, epoch)
        val_metrics = validate(model, val_loader, criterion, device, config)
        
        if config.train.scheduler == 'plateau':
            scheduler.step(val_metrics['loss'])
        else:
            scheduler.step()
            
        print(f"训练: loss={train_metrics['loss']:.4f}, iou={train_metrics['seg_iou']:.4f}, dice={train_metrics['seg_dice']:.4f}")
        print(f"验证: loss={val_metrics['loss']:.4f}, iou={val_metrics['seg_iou']:.4f}, dice={val_metrics['seg_dice']:.4f}")
        
        is_best = val_metrics['seg_iou'] > best_iou
        if is_best:
            best_iou = val_metrics['seg_iou']
            best_epoch = epoch
            
        if is_best or (epoch % config.train.save_freq == 0):
            save_checkpoint(model, optimizer, scheduler, epoch,
                {'train': train_metrics, 'val': val_metrics},
                str(save_dir / f'checkpoint_epoch_{epoch}.pth'), is_best=is_best)
            if is_best:
                print(f"  ✓ 最佳模型 (IoU: {best_iou:.4f})")
                
        if early_stopping(val_metrics['seg_iou']):
            print(f"\n早停! 最佳: epoch {best_epoch}, IoU: {best_iou:.4f}")
            break
            
    print(f"\n训练完成! 最佳 IoU: {best_iou:.4f} (Epoch {best_epoch})")
    
    if test_loader:
        best_ckpt = save_dir / f'checkpoint_epoch_{best_epoch}.pth'
        if best_ckpt.exists():
            load_checkpoint(model, str(best_ckpt), None, None, config.device)
        test_metrics = validate(model, test_loader, criterion, device, config)
        print(f"测试集: iou={test_metrics['seg_iou']:.4f}, dice={test_metrics['seg_dice']:.4f}")


def get_config(preset: str = 'default') -> Config:
    CHECKPOINTS_DIR = '/DATA/disk0/hs_25/pa/checkpoints'
    
    if preset == 'debug':
        return Config(
            data=DataConfig(batch_size=2, num_workers=0),
            model=ModelConfig(variant='tiny'),
            train=TrainConfig(epochs=5, lr=1e-4, use_amp=False, save_dir=CHECKPOINTS_DIR),
            exp_name='debug_resnet_seg_only')
    elif preset == 'base':
        return Config(
            data=DataConfig(batch_size=4),
            model=ModelConfig(variant='base'),
            train=TrainConfig(epochs=150, lr=1e-5, grad_clip=0.5, warmup_epochs=3, save_dir=CHECKPOINTS_DIR),
            exp_name='transunet_seg_only_base')
    else:
        return Config(
            data=DataConfig(data_root='/DATA/disk0/hs_25/pa',
                train_dirs=['all_dataset/train'], val_dirs=['all_dataset/val'], test_dirs=['all_dataset/test']),
            train=TrainConfig(save_dir=CHECKPOINTS_DIR),
            exp_name='transunet_seg_only')


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
    
    print(f"\n{'='*60}\n消融实验: ResNet-50 仅分割\n{'='*60}")
    print(f"模型: TransUNet-ResNet50-SegOnly-{config.model.variant}")
    train(config)


if __name__ == '__main__':
    main()
