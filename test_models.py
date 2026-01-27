#!/usr/bin/env python
"""
在test数据集上测试三个最佳模型
- transunet_fusion_tiny_h100
- transunet_seg_from_encoded_tiny_h100
- transunet_cls_only_tiny_h100
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from train.config import Config, get_config
from train.dataset import BoxWorldDataset, get_val_transform
from train.dataset_with_depth import BoxWorldDatasetWithDepth, get_val_transform_with_depth
from train.models.transunet_fusion import TransUNetFusion, TransUNetFusionConfig
from train.models.transunet_seg_from_encoded import TransUNetSegFromEncoded, TransUNetSegFromEncodedConfig
from train.models.transunet_cls_only import TransUNetClsOnly, TransUNetClsOnlyConfig
from train.utils import get_device, MetricCalculator, load_checkpoint


class ClassificationMetricCalculator:
    """仅分类任务的指标计算器"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
    
    def update(self, logits, targets):
        preds = (torch.sigmoid(logits.squeeze(-1)) > 0.5).long()
        targets = targets.long()
        
        self.tp += ((preds == 1) & (targets == 1)).sum().item()
        self.fp += ((preds == 1) & (targets == 0)).sum().item()
        self.tn += ((preds == 0) & (targets == 0)).sum().item()
        self.fn += ((preds == 0) & (targets == 1)).sum().item()
    
    def compute(self):
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn + 1e-8)
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return {
            'cls_accuracy': accuracy,
            'cls_precision': precision,
            'cls_recall': recall,
            'cls_f1': f1,
        }


def detect_model_type(checkpoint_path: str) -> str:
    """根据checkpoint路径或state_dict检测模型类型"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    keys = list(state_dict.keys())
    
    # 检查是否是twostream模型（最优先检查，因为有rgb_encoder和depth_encoder）
    has_rgb_encoder = any('rgb_encoder' in k for k in keys)
    has_depth_encoder = any('depth_encoder' in k for k in keys)
    is_twostream = has_rgb_encoder and has_depth_encoder
    
    # 检查是否有fusion相关键
    has_fusion_mlp = any('fusion_mlp' in k for k in keys)
    has_mask_guided = any('mask_guided' in k for k in keys)
    has_decoder_feat = any('decoder_feat' in k for k in keys)
    has_fusion = has_fusion_mlp or has_mask_guided or has_decoder_feat
    
    # 检查是否有seg_head
    has_seg_head = any('seg_head' in k for k in keys)
    
    # 检查是否有cls_head
    has_cls_head = any('cls_head' in k for k in keys)
    
    # 检查是否有skip connections
    # 方法1: 检查键名中是否有 "skip"
    has_skip_key = any('skip' in k.lower() for k in keys)
    
    # 方法2: 通过 seg_head 权重形状判断是否有 skip connections
    # 原始 TransUNet 的 seg_head.up_blocks.0.conv.0.weight 形状为 [128, 1280, 3, 3]（有 skip）
    # seg_from_encoded 的形状为 [128, 256, 3, 3]（无 skip）
    has_skip_by_shape = False
    if has_seg_head:
        seg_weight_key = 'seg_head.up_blocks.0.conv.0.weight'
        if seg_weight_key in state_dict:
            weight_shape = state_dict[seg_weight_key].shape
            # 如果输入通道 > 512，说明有 skip connections（拼接了 CNN 特征）
            if len(weight_shape) >= 2 and weight_shape[1] > 512:
                has_skip_by_shape = True
    
    has_skip = has_skip_key or has_skip_by_shape
    
    # 判断逻辑（按优先级）
    if is_twostream:
        # twostream模型，根据是否有seg_head进一步判断
        if has_cls_head and not has_seg_head:
            return 'twostream_cls_only'
        elif has_seg_head and not has_skip:
            return 'twostream_seg_from_encoded'
        else:
            return 'twostream_seg_from_encoded'  # 默认
    elif has_fusion:
        return 'fusion'
    elif has_cls_head and not has_seg_head:
        return 'cls_only'
    elif has_seg_head and has_cls_head and has_skip:
        # 同时有 cls_head 和 seg_head，且有 skip connections -> 原始 TransUNet
        return 'transunet'
    elif has_seg_head and not has_skip:
        return 'seg_from_encoded'
    elif has_seg_head and has_skip:
        # 只有 seg_head 有 skip -> seg_only
        return 'seg_only'
    else:
        # 默认返回fusion（因为这是最常见的）
        return 'fusion'


def detect_input_channels(state_dict: dict) -> int:
    """检测模型输入通道数（4或5）"""
    if 'cnn_encoder.conv1.weight' in state_dict:
        return state_dict['cnn_encoder.conv1.weight'].shape[1]
    return 4  # 默认4通道


def detect_model_size(state_dict: dict) -> str:
    """根据state_dict检测模型大小（micro/tiny/small/base）"""
    # 检查hidden_dim
    if 'patch_embed.proj.weight' in state_dict:
        hidden_dim = state_dict['patch_embed.proj.weight'].shape[0]
    elif 'cls_token' in state_dict:
        hidden_dim = state_dict['cls_token'].shape[2]
    else:
        return 'tiny'  # 默认
    
    # 检查transformer层数
    transformer_layers = sum(1 for k in state_dict.keys() if 'transformer.layers' in k and 'norm1.weight' in k)
    
    # 根据hidden_dim和层数判断模型大小
    if hidden_dim == 192:
        return 'micro'
    elif hidden_dim == 384 and transformer_layers == 4:
        return 'tiny'
    elif hidden_dim == 512 and transformer_layers == 6:
        return 'small'
    elif hidden_dim == 768 and transformer_layers == 12:
        return 'base'
    else:
        # 根据hidden_dim推断
        if hidden_dim <= 192:
            return 'micro'
        elif hidden_dim <= 384:
            return 'tiny'
        elif hidden_dim <= 512:
            return 'small'
        else:
            return 'base'


def load_model(checkpoint_path: str, device: torch.device, img_height: int = 480, img_width: int = 640):
    """加载模型（自动检测类型和大小）"""
    model_type = detect_model_type(checkpoint_path)
    print(f"检测到模型类型: {model_type}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    # 检测输入通道数
    in_channels = detect_input_channels(state_dict)
    # twostream 模型总是使用 RGBD（5通道：RGB + Depth + TargetMask）
    if model_type.startswith('twostream'):
        is_rgbd = True
        print(f"检测到 twostream 模型，使用 RGBD 输入（5通道）")
    else:
        is_rgbd = (in_channels == 5)
        print(f"检测到输入通道数: {in_channels} ({'RGBD' if is_rgbd else 'RGB'})")
    
    # 检测模型大小
    model_size = detect_model_size(state_dict)
    print(f"检测到模型大小: {model_size}")
    
    # 根据是否为RGBD导入不同的模型变体
    if is_rgbd:
        # RGBD版本
        if model_type == 'fusion':
            from train.models.transunet_rgbd_fusion import (
                transunet_rgbd_fusion_tiny,
                transunet_rgbd_fusion_small, transunet_rgbd_fusion_base
            )
            model_creators = {
                'micro': transunet_rgbd_fusion_tiny,  # fallback to tiny
                'tiny': transunet_rgbd_fusion_tiny,
                'small': transunet_rgbd_fusion_small,
                'base': transunet_rgbd_fusion_base,
            }
        elif model_type == 'seg_from_encoded':
            from train.models.transunet_seg_from_encoded import (
                transunet_seg_from_encoded_micro, transunet_seg_from_encoded_tiny,
                transunet_seg_from_encoded_small, transunet_seg_from_encoded_base
            )
            model_creators = {
                'micro': transunet_seg_from_encoded_micro,
                'tiny': transunet_seg_from_encoded_tiny,
                'small': transunet_seg_from_encoded_small,
                'base': transunet_seg_from_encoded_base,
            }
        elif model_type == 'cls_only':
            from train.models.transunet_rgbd_cls_only import (
                transunet_rgbd_cls_only_tiny,
                transunet_rgbd_cls_only_small, transunet_rgbd_cls_only_base
            )
            model_creators = {
                'micro': transunet_rgbd_cls_only_tiny,  # fallback to tiny
                'tiny': transunet_rgbd_cls_only_tiny,
                'small': transunet_rgbd_cls_only_small,
                'base': transunet_rgbd_cls_only_base,
            }
        elif model_type == 'transunet' or model_type == 'seg_only':
            # 原始 TransUNet RGBD（同时有 cls_head 和 seg_head，带 skip connections）
            from train.models.transunet_rgbd import (
                transunet_rgbd_tiny, transunet_rgbd_small, transunet_rgbd_base
            )
            model_creators = {
                'micro': transunet_rgbd_tiny,  # fallback to tiny
                'tiny': transunet_rgbd_tiny,
                'small': transunet_rgbd_small,
                'base': transunet_rgbd_base,
            }
        elif model_type == 'twostream_cls_only':
            from train.models.transunet_twostream import (
                twostream_cls_only_micro, twostream_cls_only_tiny,
                twostream_cls_only_small, twostream_cls_only_base
            )
            model_creators = {
                'micro': twostream_cls_only_micro,
                'tiny': twostream_cls_only_tiny,
                'small': twostream_cls_only_small,
                'base': twostream_cls_only_base,
            }
        elif model_type == 'twostream_seg_from_encoded':
            from train.models.transunet_twostream import (
                twostream_seg_from_encoded_micro, twostream_seg_from_encoded_tiny,
                twostream_seg_from_encoded_small, twostream_seg_from_encoded_base
            )
            model_creators = {
                'micro': twostream_seg_from_encoded_micro,
                'tiny': twostream_seg_from_encoded_tiny,
                'small': twostream_seg_from_encoded_small,
                'base': twostream_seg_from_encoded_base,
            }
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
    else:
        # RGB版本
        from train.models.transunet_fusion import (
            transunet_fusion_tiny,
            transunet_fusion_small, transunet_fusion_base
        )
        from train.models.transunet_seg_from_encoded import (
            transunet_seg_from_encoded_micro, transunet_seg_from_encoded_tiny,
            transunet_seg_from_encoded_small, transunet_seg_from_encoded_base
        )
        from train.models.transunet_cls_only import (
            transunet_cls_only_micro, transunet_cls_only_tiny,
            transunet_cls_only_small, transunet_cls_only_base
        )
        from train.models.transunet import (
            transunet_tiny, transunet_small, transunet_base
        )
        
        if model_type == 'fusion':
            model_creators = {
                'micro': transunet_fusion_tiny,  # fallback to tiny
                'tiny': transunet_fusion_tiny,
                'small': transunet_fusion_small,
                'base': transunet_fusion_base,
            }
        elif model_type == 'seg_from_encoded':
            model_creators = {
                'micro': transunet_seg_from_encoded_micro,
                'tiny': transunet_seg_from_encoded_tiny,
                'small': transunet_seg_from_encoded_small,
                'base': transunet_seg_from_encoded_base,
            }
        elif model_type == 'cls_only':
            model_creators = {
                'micro': transunet_cls_only_micro,
                'tiny': transunet_cls_only_tiny,
                'small': transunet_cls_only_small,
                'base': transunet_cls_only_base,
            }
        elif model_type == 'transunet' or model_type == 'seg_only':
            # 原始 TransUNet（同时有 cls_head 和 seg_head，带 skip connections）
            model_creators = {
                'micro': transunet_tiny,  # fallback to tiny
                'tiny': transunet_tiny,
                'small': transunet_small,
                'base': transunet_base,
            }
        elif model_type == 'twostream_cls_only' or model_type == 'twostream_seg_from_encoded':
            # twostream 模型应该在 RGBD 分支处理，这里只是备用
            raise ValueError(f"twostream 模型应该在 RGBD 分支处理，但检测到 is_rgbd=False")
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
    
    # 创建模型
    if model_size not in model_creators:
        print(f"警告: 模型大小 {model_size} 不在支持列表中，使用 tiny 作为默认值")
        model_size = 'tiny'
    
    creator = model_creators[model_size]
    print(f"使用模型创建函数: {creator.__name__}")
    
    # 创建模型，如果是RGBD的seg_from_encoded，需要传入in_channels=5
    create_kwargs = {
        'pretrained': False,
        'img_height': img_height,
        'img_width': img_width,
        'dropout': 0.1,
    }
    if is_rgbd and model_type == 'seg_from_encoded':
        create_kwargs['in_channels'] = 5
    
    model = creator(**create_kwargs)
    
    # 加载权重
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model, model_type, is_rgbd


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    model_type: str,
    dataset,  # BoxWorldDataset or BoxWorldDatasetWithDepth
    device: torch.device,
    batch_size: int = 32,
    output_dir: Optional[Path] = None,
    save_failures: bool = False,
    num_failures: int = 10,
) -> Dict[str, float]:
    """评估模型"""
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    
    if model_type == 'cls_only':
        metric_calc = ClassificationMetricCalculator()
    else:
        metric_calc = MetricCalculator()
    
    all_predictions = []
    failure_samples = []  # 存储失败样本的信息
    
    # ImageNet 反归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    sample_idx = 0
    for batch in tqdm(loader, desc='评估中'):
        inputs = batch['input'].to(device)
        cls_targets = batch['stability_label'].to(device)
        
        # 前向传播
        outputs = model(inputs)
        
        if model_type == 'cls_only':
            cls_logits = outputs
            seg_logits = None
            seg_targets = None
            metric_calc.update(cls_logits, cls_targets)
        else:
            cls_logits, seg_logits = outputs
            seg_targets = batch['affected_mask'].to(device)
            metric_calc.update(cls_logits, seg_logits, cls_targets, seg_targets)
        
        # 收集预测结果
        cls_probs = torch.sigmoid(cls_logits.squeeze(-1)).cpu().numpy()
        
        for i in range(inputs.size(0)):
            cls_pred = int(cls_probs[i] > 0.5)
            cls_target = int(cls_targets[i].item())
            cls_correct = (cls_pred == cls_target)
            
            # 计算分割IoU（如果有分割任务）
            seg_iou = None
            if seg_logits is not None and seg_targets is not None:
                seg_pred_binary = (torch.sigmoid(seg_logits[i]) > 0.5).cpu().numpy()
                seg_target_binary = seg_targets[i].cpu().numpy()
                intersection = np.logical_and(seg_pred_binary, seg_target_binary).sum()
                union = np.logical_or(seg_pred_binary, seg_target_binary).sum()
                seg_iou = float(intersection / (union + 1e-8))
            
            pred_dict = {
                'sample_idx': sample_idx,
                'rgb_path': batch['rgb_path'][i],
                'removal_id': batch['removal_id'][i],
                'cls_prob': float(cls_probs[i]),
                'cls_pred': cls_pred,
                'cls_target': cls_target,
                'cls_correct': cls_correct,
                'seg_iou': seg_iou,
            }
            all_predictions.append(pred_dict)
            
            # 收集失败样本（分类错误或分割IoU低）
            if save_failures:
                is_failure = False
                failure_reason = []
                
                if not cls_correct:
                    is_failure = True
                    failure_reason.append('cls_error')
                
                if seg_iou is not None and seg_iou < 0.3:
                    is_failure = True
                    failure_reason.append(f'low_iou_{seg_iou:.3f}')
                
                if is_failure:
                    # 只保存必要信息，不保存tensor（避免内存溢出）
                    failure_samples.append({
                        'sample_idx': sample_idx,
                        'rgb_path': batch['rgb_path'][i],
                        'removal_id': batch['removal_id'][i],
                        'cls_target': cls_target,
                        'cls_prob': float(cls_probs[i]),
                        'cls_pred': cls_pred,
                        'seg_iou': seg_iou,
                        'failure_reason': failure_reason,
                    })
            
            sample_idx += 1
    
    # 计算指标
    metrics = metric_calc.compute()
    
    # 保存预测结果
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'predictions.json', 'w') as f:
            json.dump(all_predictions, f, indent=2)
    
    # 保存失败样本
    if save_failures and failure_samples:
        # 按失败严重程度排序（优先选择分类错误且IoU低的）
        failure_samples.sort(key=lambda x: (
            0 if 'cls_error' in x['failure_reason'] else 1,  # 分类错误优先
            x['seg_iou'] if x['seg_iou'] is not None else 1.0  # IoU低的优先
        ))
        
        # 选择前num_failures个
        selected_failures = failure_samples[:num_failures]
        
        # 保存失败样本的可视化（重新从dataset加载数据）
        visualize_failures(
            selected_failures,
            dataset,
            model,
            device,
            output_dir / 'failures',
            mean,
            std,
        )
        
        # 保存失败样本信息
        failure_info = [{
            'sample_idx': f['sample_idx'],
            'rgb_path': f['rgb_path'],
            'removal_id': f['removal_id'],
            'cls_pred': f['cls_pred'],
            'cls_target': f['cls_target'],
            'cls_prob': f['cls_prob'],
            'seg_iou': f['seg_iou'],
            'failure_reason': f['failure_reason'],
        } for f in selected_failures]
        
        with open(output_dir / 'failures_info.json', 'w') as f:
            json.dump(failure_info, f, indent=2)
        
        print(f"\n保存了 {len(selected_failures)} 个失败样本到: {output_dir / 'failures'}")
    
    return metrics


@torch.no_grad()
def visualize_failures(
    failure_samples: List[Dict],
    dataset: BoxWorldDataset,
    model: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
    mean: torch.Tensor,
    std: torch.Tensor,
    threshold: float = 0.5,
):
    """可视化失败样本（从dataset重新加载数据，避免内存溢出）"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 尝试设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass
    
    for idx, sample_info in enumerate(tqdm(failure_samples, desc='生成失败样本可视化')):
        sample_idx = sample_info['sample_idx']
        cls_target = sample_info['cls_target']
        cls_pred = sample_info['cls_pred']
        cls_prob = sample_info['cls_prob']
        removal_id = sample_info['removal_id']
        seg_iou = sample_info['seg_iou']
        
        # 从dataset重新加载数据
        dataset_sample = dataset[sample_idx]
        input_tensor = dataset_sample['input'].unsqueeze(0).to(device)
        
        # 重新预测
        outputs = model(input_tensor)
        if len(outputs) == 2:
            cls_logits, seg_logits = outputs
        else:
            cls_logits = outputs
            seg_logits = None
        
        # 反归一化 RGB 图像
        rgb = dataset_sample['input'][:3]
        rgb = rgb * std + mean
        rgb = rgb.clamp(0, 1).permute(1, 2, 0).numpy()
        
        # 被移除箱子掩码
        removed_mask = dataset_sample['input'][3].numpy()
        
        # 创建可视化图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原始 RGB
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title('RGB Image')
        axes[0, 0].axis('off')
        
        # 被移除箱子
        axes[0, 1].imshow(rgb)
        axes[0, 1].imshow(removed_mask, alpha=0.5, cmap='Greens')
        axes[0, 1].set_title('Removed Box (Target)')
        axes[0, 1].axis('off')
        
        # 分类结果
        result_text = f"Pred: {'Stable' if cls_pred else 'Unstable'} ({cls_prob:.3f})\n"
        result_text += f"True: {'Stable' if cls_target else 'Unstable'}"
        correct = '✓' if cls_pred == cls_target else '✗'
        color = 'green' if cls_pred == cls_target else 'red'
        
        axes[0, 2].text(0.5, 0.5, f"{correct}\n{result_text}", 
                        ha='center', va='center', fontsize=16, color=color,
                        transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Classification Result')
        axes[0, 2].axis('off')
        
        # 如果有分割任务
        if seg_logits is not None and 'affected_mask' in dataset_sample:
            seg_pred = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
            seg_target = dataset_sample['affected_mask'].numpy().squeeze()
            
            # 真实受影响区域
            axes[1, 0].imshow(rgb)
            axes[1, 0].imshow(seg_target, alpha=0.5, cmap='Reds')
            axes[1, 0].set_title(f'True Affected Region')
            axes[1, 0].axis('off')
            
            # 预测受影响区域
            axes[1, 1].imshow(rgb)
            axes[1, 1].imshow(seg_pred, alpha=0.5, cmap='Reds')
            axes[1, 1].set_title(f'Pred Affected Region (IoU: {seg_iou:.3f})')
            axes[1, 1].axis('off')
            
            # 分割对比
            seg_pred_binary = (seg_pred > threshold).astype(float)
            overlay = np.zeros((*seg_target.shape, 3))
            overlay[..., 0] = seg_pred_binary  # 红色: 预测
            overlay[..., 1] = seg_target       # 绿色: 真实
            # 黄色: 交集
            
            axes[1, 2].imshow(rgb)
            axes[1, 2].imshow(overlay, alpha=0.5)
            axes[1, 2].set_title('Red=Pred, Green=True, Yellow=Overlap')
            axes[1, 2].axis('off')
        else:
            # 仅分类任务，显示失败原因
            axes[1, 0].axis('off')
            axes[1, 1].axis('off')
            failure_reasons = ', '.join(sample_info['failure_reason'])
            axes[1, 2].text(0.5, 0.5, f"Failure Reasons:\n{failure_reasons}", 
                           ha='center', va='center', fontsize=14, color='red',
                           transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Failure Analysis')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # 保存图像
        iou_str = f"{seg_iou:.3f}" if seg_iou is not None else "N/A"
        save_path = output_dir / f'failure_{idx:02d}_{removal_id}_iou{iou_str}.png'
        # 确保目录存在
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='在test数据集上测试模型')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--test_dir', type=str, default='/DATA/disk0/hs_25/pa/all_dataset/test', help='测试数据目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录（可选）')
    parser.add_argument('--save_failures', action='store_true', help='保存预测失败的样本')
    parser.add_argument('--num_failures', type=int, default=10, help='保存的失败样本数量')
    
    args = parser.parse_args()
    
    # 设备
    device = get_device(args.device)
    
    # 加载模型
    print(f"\n{'='*60}")
    print(f"加载模型: {args.checkpoint}")
    print(f"{'='*60}")
    model, model_type, is_rgbd = load_model(args.checkpoint, device)
    print(f"模型类型: {model_type}")
    print(f"模型参数量: {model.get_num_params() / 1e6:.2f}M")
    
    # 加载测试数据集（根据模型类型选择正确的数据集类）
    print(f"\n{'='*60}")
    print(f"加载测试数据集: {args.test_dir}")
    print(f"{'='*60}")
    if is_rgbd:
        # RGBD模型需要5通道输入（RGB + Depth + Mask）
        transform = get_val_transform_with_depth()
        dataset = BoxWorldDatasetWithDepth([args.test_dir], transform=transform)
    else:
        # RGB模型需要4通道输入（RGB + Mask）
        transform = get_val_transform()
        dataset = BoxWorldDataset([args.test_dir], transform=transform)
    print(f"测试集样本数: {len(dataset)}")
    
    # 评估
    print(f"\n{'='*60}")
    print("开始评估...")
    print(f"{'='*60}")
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is None:
        # 自动生成输出目录
        checkpoint_name = Path(args.checkpoint).parent.name
        output_dir = Path('test_results') / checkpoint_name
        output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = evaluate_model(
        model, model_type, dataset, device,
        batch_size=args.batch_size,
        output_dir=output_dir,
        save_failures=args.save_failures,
        num_failures=args.num_failures,
    )
    
    # 打印结果
    print(f"\n{'='*60}")
    print("评估结果")
    print(f"{'='*60}")
    print(f"分类准确率: {metrics['cls_accuracy']:.4f}")
    print(f"分类精确率: {metrics['cls_precision']:.4f}")
    print(f"分类召回率: {metrics['cls_recall']:.4f}")
    print(f"分类 F1:    {metrics['cls_f1']:.4f}")
    
    if 'seg_iou' in metrics:
        print(f"分割 IoU:   {metrics['seg_iou']:.4f}")
        print(f"分割 Dice:  {metrics['seg_dice']:.4f}")
    
    # 保存指标
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n结果已保存到: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
