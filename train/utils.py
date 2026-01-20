"""
训练工具函数
"""

import os
import random
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def set_seed(seed: int = 42):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = 'cuda') -> torch.device:
    """获取计算设备"""
    if device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("使用 CPU")
    return device


# =============================================================================
# 损失函数
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    """Dice Loss for segmentation (数值稳定版本)"""
    
    def __init__(self, smooth: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps  # 防止除零的小常数
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 强制转换为 FP32，避免 AMP 训练时的数值不稳定
        inputs = inputs.float()
        targets = targets.float()
        
        inputs = torch.sigmoid(inputs)
        
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        denominator = inputs.sum() + targets.sum() + self.smooth + self.eps
        dice = (2. * intersection + self.smooth) / denominator
        
        # 确保结果在有效范围内
        dice = torch.clamp(dice, min=self.eps, max=1.0 - self.eps)
        
        return 1 - dice


class DiceBCELoss(nn.Module):
    """Dice + BCE Loss (数值稳定版本)"""
    
    def __init__(self, smooth: float = 1.0, dice_weight: float = 0.5, eps: float = 1e-7):
        super().__init__()
        self.dice_loss = DiceLoss(smooth, eps)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 强制转换为 FP32，避免 AMP 训练时的数值不稳定
        inputs = inputs.float()
        targets = targets.float()
        
        dice = self.dice_loss(inputs, targets)
        bce = self.bce_loss(inputs, targets)
        return self.dice_weight * dice + (1 - self.dice_weight) * bce


class CombinedLoss(nn.Module):
    """组合损失：分类 + 分割
    
    关键改进：
    1. 只对不稳定样本（cls_targets=0）计算分割 loss
    2. 增加正样本权重处理像素级不平衡
    """
    
    def __init__(
        self,
        cls_loss: str = 'bce',
        seg_loss: str = 'dice_bce',
        cls_weight: float = 1.0,
        seg_weight: float = 1.0,
        pos_weight: float = 1.0,
        seg_pos_weight: float = 10.0,  # 分割任务的正样本权重
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        dice_smooth: float = 1.0,
    ):
        super().__init__()
        
        self.cls_weight = cls_weight
        self.seg_weight = seg_weight
        self.seg_pos_weight = seg_pos_weight
        
        # 分类损失
        if cls_loss == 'bce':
            self.cls_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        elif cls_loss == 'focal':
            self.cls_criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            raise ValueError(f"未知的分类损失: {cls_loss}")
            
        # 分割损失 - 使用带权重的 BCE
        if seg_loss == 'bce':
            self.seg_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([seg_pos_weight]))
            self.seg_dice = None
        elif seg_loss == 'dice':
            self.seg_bce = None
            self.seg_dice = DiceLoss(smooth=dice_smooth)
        elif seg_loss == 'dice_bce':
            self.seg_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([seg_pos_weight]))
            self.seg_dice = DiceLoss(smooth=dice_smooth)
        else:
            raise ValueError(f"未知的分割损失: {seg_loss}")
            
    def forward(
        self,
        cls_logits: torch.Tensor,
        seg_logits: torch.Tensor,
        cls_targets: torch.Tensor,
        seg_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            cls_logits: (B, 1) 分类 logits
            seg_logits: (B, 1, H, W) 分割 logits
            cls_targets: (B,) 分类标签 - 0=不稳定, 1=稳定
            seg_targets: (B, 1, H, W) 分割掩码
            
        Returns:
            total_loss: 总损失
            loss_dict: 各部分损失
        """
        # 确保 pos_weight 在正确的设备上
        if self.seg_bce is not None and self.seg_bce.pos_weight.device != seg_logits.device:
            self.seg_bce.pos_weight = self.seg_bce.pos_weight.to(seg_logits.device)
        if hasattr(self.cls_criterion, 'pos_weight') and self.cls_criterion.pos_weight.device != cls_logits.device:
            self.cls_criterion.pos_weight = self.cls_criterion.pos_weight.to(cls_logits.device)
        
        # 分类损失
        cls_loss = self.cls_criterion(cls_logits.squeeze(-1), cls_targets)
        
        # 分割损失 - 只对不稳定样本计算（cls_targets=0 表示不稳定）
        unstable_mask = (cls_targets == 0)
        n_unstable = unstable_mask.sum().item()
        
        if n_unstable > 0:
            seg_logits_unstable = seg_logits[unstable_mask]
            seg_targets_unstable = seg_targets[unstable_mask]
            
            if self.seg_bce is not None and self.seg_dice is not None:
                # Dice + BCE
                bce_loss = self.seg_bce(seg_logits_unstable, seg_targets_unstable)
                dice_loss = self.seg_dice(seg_logits_unstable, seg_targets_unstable)
                seg_loss = 0.5 * bce_loss + 0.5 * dice_loss
            elif self.seg_bce is not None:
                seg_loss = self.seg_bce(seg_logits_unstable, seg_targets_unstable)
            else:
                seg_loss = self.seg_dice(seg_logits_unstable, seg_targets_unstable)
        else:
            # 批次中全是稳定样本，分割 loss 为 0
            seg_loss = torch.tensor(0.0, device=seg_logits.device, requires_grad=True)
        
        # 总损失
        total_loss = self.cls_weight * cls_loss + self.seg_weight * seg_loss
        
        loss_dict = {
            'cls_loss': cls_loss.item(),
            'seg_loss': seg_loss.item(),
            'total_loss': total_loss.item(),
        }
        
        return total_loss, loss_dict


# =============================================================================
# 评估指标
# =============================================================================

class MetricCalculator:
    """评估指标计算器
    
    关键改进：只对不稳定样本计算分割指标
    """
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()
        
    def reset(self):
        """重置所有计数器"""
        # 分类指标
        self.cls_tp = 0
        self.cls_fp = 0
        self.cls_tn = 0
        self.cls_fn = 0
        
        # 分割指标 (只统计不稳定样本)
        self.seg_intersection = 0
        self.seg_union = 0
        self.seg_pred_sum = 0
        self.seg_target_sum = 0
        
        # 样本计数
        self.n_samples = 0
        self.n_unstable_samples = 0
        
    def update(
        self,
        cls_logits: torch.Tensor,
        seg_logits: torch.Tensor,
        cls_targets: torch.Tensor,
        seg_targets: torch.Tensor,
    ):
        """更新指标"""
        with torch.no_grad():
            # 分类预测
            cls_probs = torch.sigmoid(cls_logits.squeeze(-1))
            cls_preds = (cls_probs > self.threshold).float()
            
            # 分类混淆矩阵
            self.cls_tp += ((cls_preds == 1) & (cls_targets == 1)).sum().item()
            self.cls_fp += ((cls_preds == 1) & (cls_targets == 0)).sum().item()
            self.cls_tn += ((cls_preds == 0) & (cls_targets == 0)).sum().item()
            self.cls_fn += ((cls_preds == 0) & (cls_targets == 1)).sum().item()
            
            # 分割指标 - 只对不稳定样本计算（cls_targets=0）
            unstable_mask = (cls_targets == 0)
            n_unstable = unstable_mask.sum().item()
            
            if n_unstable > 0:
                seg_logits_unstable = seg_logits[unstable_mask]
                seg_targets_unstable = seg_targets[unstable_mask]
                
                seg_probs = torch.sigmoid(seg_logits_unstable)
                seg_preds = (seg_probs > self.threshold).float()
                
                # 分割 IoU
                intersection = (seg_preds * seg_targets_unstable).sum().item()
                union = ((seg_preds + seg_targets_unstable) > 0).float().sum().item()
                
                self.seg_intersection += intersection
                self.seg_union += union
                self.seg_pred_sum += seg_preds.sum().item()
                self.seg_target_sum += seg_targets_unstable.sum().item()
                self.n_unstable_samples += n_unstable
            
            self.n_samples += cls_targets.size(0)
            
    def compute(self) -> Dict[str, float]:
        """计算所有指标"""
        metrics = {}
        
        # 分类指标
        eps = 1e-7
        
        # 准确率
        metrics['cls_accuracy'] = (self.cls_tp + self.cls_tn) / (self.n_samples + eps)
        
        # 精确率 (Precision)
        metrics['cls_precision'] = self.cls_tp / (self.cls_tp + self.cls_fp + eps)
        
        # 召回率 (Recall)
        metrics['cls_recall'] = self.cls_tp / (self.cls_tp + self.cls_fn + eps)
        
        # F1 Score
        if metrics['cls_precision'] + metrics['cls_recall'] > 0:
            metrics['cls_f1'] = 2 * metrics['cls_precision'] * metrics['cls_recall'] / (
                metrics['cls_precision'] + metrics['cls_recall']
            )
        else:
            metrics['cls_f1'] = 0.0
            
        # 分割指标
        # IoU
        metrics['seg_iou'] = self.seg_intersection / (self.seg_union + eps)
        
        # Dice
        metrics['seg_dice'] = 2 * self.seg_intersection / (
            self.seg_pred_sum + self.seg_target_sum + eps
        )
        
        return metrics


# =============================================================================
# 学习率调度器
# =============================================================================

class WarmupCosineScheduler(_LRScheduler):
    """带 Warmup 的余弦退火学习率调度器"""
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup 阶段：线性增长 (从 min_lr 开始，避免 lr=0)
            alpha = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [self.min_lr + (base_lr - self.min_lr) * alpha for base_lr in self.base_lrs]
        else:
            # 余弦退火
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]


def create_optimizer(model: nn.Module, config) -> Optimizer:
    """创建优化器"""
    # 分离需要 weight decay 的参数
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'norm' in name or 'bn' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    param_groups = [
        {'params': decay_params, 'weight_decay': config.train.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
    
    if config.train.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=config.train.lr)
    elif config.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_groups, lr=config.train.lr)
    elif config.train.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups, lr=config.train.lr, momentum=0.9
        )
    else:
        raise ValueError(f"未知的优化器: {config.train.optimizer}")
        
    return optimizer


def create_scheduler(optimizer: Optimizer, config) -> _LRScheduler:
    """创建学习率调度器"""
    if config.train.scheduler == 'cosine':
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=config.train.warmup_epochs,
            total_epochs=config.train.epochs,
            min_lr=config.train.min_lr,
        )
    elif config.train.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    elif config.train.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
    else:
        raise ValueError(f"未知的调度器: {config.train.scheduler}")
        
    return scheduler


# =============================================================================
# 检查点管理
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler,  # 可以是 _LRScheduler 或 ReduceLROnPlateau
    epoch: int,
    metrics: Dict,
    save_path: str,
    is_best: bool = False,
    max_retries: int = 3,
    use_temp_file: bool = True,
):
    """
    保存检查点（带错误处理和重试机制）
    
    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前 epoch
        metrics: 指标字典
        save_path: 保存路径
        is_best: 是否为最佳模型
        max_retries: 最大重试次数
        use_temp_file: 是否使用临时文件（先保存到 /tmp，再移动，避免 NFS 写入问题）
    """
    # 安全获取 scheduler state_dict
    scheduler_state = None
    if scheduler is not None:
        try:
            scheduler_state = scheduler.state_dict()
        except Exception:
            pass  # 某些调度器可能没有 state_dict
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler_state,
        'metrics': metrics,
    }
    
    # 保存主 checkpoint
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    success = False
    last_error = None
    
    for attempt in range(max_retries):
        try:
            if use_temp_file:
                # 使用临时文件：先保存到 /tmp，再原子性移动到目标位置
                # 这样可以避免 NFS 写入中断导致文件损坏
                with tempfile.NamedTemporaryFile(
                    mode='wb', 
                    delete=False, 
                    dir='/tmp',
                    suffix='.pth'
                ) as tmp_file:
                    tmp_path = tmp_file.name
                
                try:
                    # 保存到临时文件
                    torch.save(checkpoint, tmp_path)
                    # 原子性移动到目标位置
                    shutil.move(tmp_path, save_path)
                    success = True
                except Exception as e:
                    # 清理临时文件
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except:
                            pass
                    raise e
            else:
                # 直接保存（如果目标目录是本地文件系统）
                torch.save(checkpoint, save_path)
                success = True
            
            break  # 成功则退出重试循环
            
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # 递增等待时间：2s, 4s, 6s
                print(f"警告: 保存 checkpoint 失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                print(f"  {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"错误: 保存 checkpoint 失败，已重试 {max_retries} 次")
                print(f"  路径: {save_path}")
                print(f"  错误: {last_error}")
                # 不抛出异常，让训练继续（但打印警告）
                return False
    
    # 保存最佳模型副本
    if success and is_best:
        best_path = str(save_path_obj.with_suffix('')) + '_best.pth'
        best_success = False
        
        for attempt in range(max_retries):
            try:
                if use_temp_file:
                    with tempfile.NamedTemporaryFile(
                        mode='wb', 
                        delete=False, 
                        dir='/tmp',
                        suffix='.pth'
                    ) as tmp_file:
                        tmp_path = tmp_file.name
                    
                    try:
                        torch.save(checkpoint, tmp_path)
                        shutil.move(tmp_path, best_path)
                        best_success = True
                    except Exception as e:
                        if os.path.exists(tmp_path):
                            try:
                                os.remove(tmp_path)
                            except:
                                pass
                        raise e
                else:
                    torch.save(checkpoint, best_path)
                    best_success = True
                
                break
                
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    wait_time = (attempt + 1) * 2
                    print(f"警告: 保存最佳模型失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    time.sleep(wait_time)
                else:
                    print(f"错误: 保存最佳模型失败，已重试 {max_retries} 次")
                    print(f"  路径: {best_path}")
                    print(f"  错误: {e}")
        
        return best_success if is_best else success
    
    return success
        

def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[Optimizer] = None,
    scheduler = None,  # 可以是 _LRScheduler 或 ReduceLROnPlateau
    device: str = 'cuda',
) -> Dict:
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    if scheduler and checkpoint.get('scheduler_state_dict') is not None:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            print(f"警告: 无法加载调度器状态: {e}")
        
    return checkpoint


# =============================================================================
# 日志工具
# =============================================================================

class AverageMeter:
    """计算和存储平均值和当前值"""
    
    def __init__(self, name: str = ''):
        self.name = name
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ProgressLogger:
    """训练进度日志"""
    
    def __init__(self, total_epochs: int, total_batches: int, log_freq: int = 10):
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.log_freq = log_freq
        
        self.epoch_start_time = None
        self.batch_times = AverageMeter('batch_time')
        
    def epoch_start(self, epoch: int):
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        self.batch_times.reset()
        
    def batch_end(self, batch_idx: int, loss_dict: Dict, lr: float):
        self.batch_times.update(time.time() - self.epoch_start_time if self.epoch_start_time else 0)
        
        if batch_idx % self.log_freq == 0:
            loss_str = ' | '.join([f'{k}: {v:.4f}' for k, v in loss_dict.items()])
            print(
                f'Epoch [{self.current_epoch}/{self.total_epochs}] '
                f'Batch [{batch_idx}/{self.total_batches}] | '
                f'{loss_str} | lr: {lr:.2e}'
            )
            
    def epoch_end(self, train_metrics: Dict, val_metrics: Dict):
        epoch_time = time.time() - self.epoch_start_time
        
        print('\n' + '=' * 80)
        print(f'Epoch {self.current_epoch} 完成 (耗时: {epoch_time:.1f}s)')
        
        print('\n训练指标:')
        for k, v in train_metrics.items():
            print(f'  {k}: {v:.4f}')
            
        print('\n验证指标:')
        for k, v in val_metrics.items():
            print(f'  {k}: {v:.4f}')
            
        print('=' * 80 + '\n')


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
