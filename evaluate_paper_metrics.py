#!/usr/bin/env python
"""
论文评估指标脚本

评估模型在测试数据集上的性能，使用论文中定义的指标：

## 宏观指标 (Macro-Metrics) - 用于全局稳定性预测
1. Safety Precision (安全精确度) - 核心指标
   定义: TP_safe / (TP_safe + FP_safe)
   目的: 最大限度减少"假安全"预测，避免将危险动作误判为安全
   
2. F1-Score
   综合分类性能指标

## 微观指标 (Micro-Metrics) - 用于受影响区域分割
1. mIoU (Affected Class)
   受影响类别的平均交并比，评估对不稳定区域的空间理解能力
   
2. Boundary F-measure
   不稳定性轮廓的对齐精度，评估边界检测质量

用法:
    # 评估单个模型
    python evaluate_paper_metrics.py --checkpoint /path/to/checkpoint
    
    # 评估所有带有 'best' 检查点的模型（默认行为）
    python evaluate_paper_metrics.py --all
    
    # 评估所有模型（包括只有 epoch checkpoint 的）
    python evaluate_paper_metrics.py --all --include_epoch
    
    # 指定测试数据集
    python evaluate_paper_metrics.py --checkpoint /path/to/checkpoint --test_dir /path/to/test
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import cv2
import numpy as np
import torch
from scipy import ndimage
from tqdm import tqdm

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from train.dataset import BoxWorldDataset, get_val_transform
from train.dataset_with_depth import BoxWorldDatasetWithDepth, get_val_transform_with_depth
from train.utils import get_device

# 从 test_models.py 导入模型加载函数
from test_models import load_model, detect_model_type


class PaperMetricsCalculator:
    """
    论文中定义的评估指标计算器
    
    宏观指标:
    - Safety Precision: 安全精确度，核心指标
    - F1-Score: 综合分类性能
    
    微观指标:
    - mIoU (Affected Class): 受影响区域的IoU
    - Boundary F-measure: 边界对齐精度
    """
    
    def __init__(self, threshold: float = 0.5, boundary_tolerance: int = 2):
        """
        Args:
            threshold: 二值化阈值
            boundary_tolerance: 边界匹配容差（像素）
        """
        self.threshold = threshold
        self.boundary_tolerance = boundary_tolerance
        self.reset()
        
    def reset(self):
        """重置所有计数器"""
        # 分类混淆矩阵
        # Safe = 1 (稳定), Unsafe = 0 (不稳定)
        self.tp_safe = 0  # True Positive for Safe: 预测安全，实际安全
        self.fp_safe = 0  # False Positive for Safe: 预测安全，实际不安全 (危险！)
        self.tn_safe = 0  # True Negative for Safe: 预测不安全，实际不安全
        self.fn_safe = 0  # False Negative for Safe: 预测不安全，实际安全
        
        # 分割指标 (只统计不稳定样本)
        self.seg_intersection = 0
        self.seg_union = 0
        self.seg_pred_sum = 0
        self.seg_target_sum = 0
        
        # 边界指标
        self.boundary_tp = 0
        self.boundary_pred_count = 0
        self.boundary_gt_count = 0
        
        # 样本计数
        self.n_samples = 0
        self.n_unstable_samples = 0
        
        # 逐样本IoU (用于计算mIoU)
        self.sample_ious = []
        
        # 逐样本边界F-measure
        self.sample_boundary_f = []
        
    def _extract_boundary(self, mask: np.ndarray) -> np.ndarray:
        """
        提取掩码的边界
        
        Args:
            mask: 二值掩码 (H, W)
            
        Returns:
            boundary: 边界掩码 (H, W)
        """
        # 使用形态学操作提取边界
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        boundary = mask.astype(np.uint8) - eroded
        return boundary
    
    def _compute_boundary_f_measure(
        self, 
        pred_mask: np.ndarray, 
        gt_mask: np.ndarray
    ) -> Tuple[float, int, int, int]:
        """
        计算边界 F-measure
        
        使用距离变换来计算边界匹配精度
        
        Args:
            pred_mask: 预测掩码 (H, W)
            gt_mask: 真实掩码 (H, W)
            
        Returns:
            f_measure: 边界F-measure
            matched: 匹配的边界点数
            pred_boundary_count: 预测边界点数
            gt_boundary_count: 真实边界点数
        """
        # 提取边界
        pred_boundary = self._extract_boundary(pred_mask)
        gt_boundary = self._extract_boundary(gt_mask)
        
        pred_boundary_points = np.sum(pred_boundary > 0)
        gt_boundary_points = np.sum(gt_boundary > 0)
        
        if pred_boundary_points == 0 and gt_boundary_points == 0:
            return 1.0, 0, 0, 0
        
        if pred_boundary_points == 0 or gt_boundary_points == 0:
            return 0.0, 0, pred_boundary_points, gt_boundary_points
        
        # 计算距离变换
        # 对于预测边界的每个点，找到最近的真实边界点
        gt_distance = ndimage.distance_transform_edt(1 - gt_boundary)
        pred_distance = ndimage.distance_transform_edt(1 - pred_boundary)
        
        # 计算 Precision: 预测边界有多少在容差范围内匹配真实边界
        pred_matched = np.sum(
            (pred_boundary > 0) & (gt_distance <= self.boundary_tolerance)
        )
        precision = pred_matched / pred_boundary_points if pred_boundary_points > 0 else 0
        
        # 计算 Recall: 真实边界有多少被预测边界覆盖
        gt_matched = np.sum(
            (gt_boundary > 0) & (pred_distance <= self.boundary_tolerance)
        )
        recall = gt_matched / gt_boundary_points if gt_boundary_points > 0 else 0
        
        # F-measure
        if precision + recall > 0:
            f_measure = 2 * precision * recall / (precision + recall)
        else:
            f_measure = 0.0
            
        return f_measure, pred_matched, pred_boundary_points, gt_boundary_points
    
    def update(
        self,
        cls_logits: torch.Tensor,
        seg_logits: Optional[torch.Tensor],
        cls_targets: torch.Tensor,
        seg_targets: Optional[torch.Tensor],
    ):
        """
        更新指标
        
        Args:
            cls_logits: (B, 1) 分类 logits
            seg_logits: (B, 1, H, W) 分割 logits，可以为 None (cls_only 模型)
            cls_targets: (B,) 分类标签 - 0=不稳定, 1=稳定
            seg_targets: (B, 1, H, W) 分割掩码，可以为 None
        """
        with torch.no_grad():
            batch_size = cls_targets.size(0)
            
            # 分类预测
            cls_probs = torch.sigmoid(cls_logits.squeeze(-1))
            cls_preds = (cls_probs > self.threshold).long()
            cls_targets_long = cls_targets.long()
            
            # 更新分类混淆矩阵 (Safe=1 为正类)
            for i in range(batch_size):
                pred = cls_preds[i].item()
                target = cls_targets_long[i].item()
                
                if pred == 1 and target == 1:
                    self.tp_safe += 1
                elif pred == 1 and target == 0:
                    self.fp_safe += 1  # 危险！将不安全误判为安全
                elif pred == 0 and target == 0:
                    self.tn_safe += 1
                else:  # pred == 0 and target == 1
                    self.fn_safe += 1
            
            self.n_samples += batch_size
            
            # 分割指标 - 只对不稳定样本计算
            if seg_logits is not None and seg_targets is not None:
                unstable_mask = (cls_targets == 0)
                n_unstable = unstable_mask.sum().item()
                
                if n_unstable > 0:
                    seg_logits_unstable = seg_logits[unstable_mask]
                    seg_targets_unstable = seg_targets[unstable_mask]
                    
                    seg_probs = torch.sigmoid(seg_logits_unstable)
                    seg_preds = (seg_probs > self.threshold).float()
                    
                    # 全局分割统计
                    intersection = (seg_preds * seg_targets_unstable).sum().item()
                    union = ((seg_preds + seg_targets_unstable) > 0).float().sum().item()
                    
                    self.seg_intersection += intersection
                    self.seg_union += union
                    self.seg_pred_sum += seg_preds.sum().item()
                    self.seg_target_sum += seg_targets_unstable.sum().item()
                    
                    # 逐样本计算 IoU 和 Boundary F-measure
                    for i in range(n_unstable):
                        pred_mask = seg_preds[i].squeeze().cpu().numpy()
                        gt_mask = seg_targets_unstable[i].squeeze().cpu().numpy()
                        
                        # 样本级 IoU
                        sample_intersection = np.logical_and(pred_mask > 0.5, gt_mask > 0.5).sum()
                        sample_union = np.logical_or(pred_mask > 0.5, gt_mask > 0.5).sum()
                        
                        if sample_union > 0:
                            sample_iou = sample_intersection / sample_union
                        else:
                            sample_iou = 1.0  # 两者都为空
                        self.sample_ious.append(sample_iou)
                        
                        # Boundary F-measure
                        pred_binary = (pred_mask > 0.5).astype(np.uint8)
                        gt_binary = (gt_mask > 0.5).astype(np.uint8)
                        
                        boundary_f, matched, pred_count, gt_count = \
                            self._compute_boundary_f_measure(pred_binary, gt_binary)
                        
                        self.sample_boundary_f.append(boundary_f)
                        self.boundary_tp += matched
                        self.boundary_pred_count += pred_count
                        self.boundary_gt_count += gt_count
                    
                    self.n_unstable_samples += n_unstable
    
    def compute(self) -> Dict[str, float]:
        """计算所有指标"""
        metrics = {}
        eps = 1e-7
        
        # =================================================================
        # 宏观指标 (Macro-Metrics) - 全局稳定性预测
        # =================================================================
        
        # Safety Precision (核心指标)
        # 定义: TP_safe / (TP_safe + FP_safe)
        # 高 Safety Precision = 最小化将危险误判为安全的情况
        safety_precision = self.tp_safe / (self.tp_safe + self.fp_safe + eps)
        metrics['safety_precision'] = safety_precision
        
        # Safety Recall (辅助指标)
        # 定义: TP_safe / (TP_safe + FN_safe)
        safety_recall = self.tp_safe / (self.tp_safe + self.fn_safe + eps)
        metrics['safety_recall'] = safety_recall
        
        # F1-Score (综合指标)
        if safety_precision + safety_recall > 0:
            safety_f1 = 2 * safety_precision * safety_recall / (safety_precision + safety_recall)
        else:
            safety_f1 = 0.0
        metrics['f1_score'] = safety_f1
        
        # 额外分类指标 (供参考)
        total = self.tp_safe + self.fp_safe + self.tn_safe + self.fn_safe
        metrics['accuracy'] = (self.tp_safe + self.tn_safe) / (total + eps)
        
        # Unstable Precision (不稳定类别精确度)
        unstable_precision = self.tn_safe / (self.tn_safe + self.fn_safe + eps)
        metrics['unstable_precision'] = unstable_precision
        
        # Unstable Recall (不稳定类别召回率)
        unstable_recall = self.tn_safe / (self.tn_safe + self.fp_safe + eps)
        metrics['unstable_recall'] = unstable_recall
        
        # =================================================================
        # 微观指标 (Micro-Metrics) - 受影响区域分割
        # =================================================================
        
        # mIoU (Affected Class)
        # 只计算不稳定样本中受影响区域的平均IoU
        if len(self.sample_ious) > 0:
            miou_affected = np.mean(self.sample_ious)
        else:
            miou_affected = 0.0
        metrics['miou_affected'] = miou_affected
        
        # 全局 IoU (供参考)
        global_iou = self.seg_intersection / (self.seg_union + eps)
        metrics['global_iou'] = global_iou
        
        # Dice Score (供参考)
        dice = 2 * self.seg_intersection / (self.seg_pred_sum + self.seg_target_sum + eps)
        metrics['dice'] = dice
        
        # Boundary F-measure
        if len(self.sample_boundary_f) > 0:
            boundary_f = np.mean(self.sample_boundary_f)
        else:
            boundary_f = 0.0
        metrics['boundary_f_measure'] = boundary_f
        
        # =================================================================
        # 统计信息
        # =================================================================
        metrics['n_samples'] = self.n_samples
        metrics['n_unstable_samples'] = self.n_unstable_samples
        metrics['n_stable_samples'] = self.n_samples - self.n_unstable_samples
        
        # 混淆矩阵
        metrics['confusion_matrix'] = {
            'tp_safe': self.tp_safe,
            'fp_safe': self.fp_safe,
            'tn_safe': self.tn_safe,
            'fn_safe': self.fn_safe,
        }
        
        return metrics


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    model_type: str,
    dataset,
    device: torch.device,
    batch_size: int = 32,
) -> Dict:
    """
    评估单个模型
    
    Args:
        model: 模型
        model_type: 模型类型 ('fusion', 'cls_only', 'seg_from_encoded', etc.)
        dataset: 测试数据集
        device: 计算设备
        batch_size: 批次大小
        
    Returns:
        metrics: 评估指标字典
    """
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    
    metric_calc = PaperMetricsCalculator()
    
    for batch in tqdm(loader, desc='评估中'):
        inputs = batch['input'].to(device)
        cls_targets = batch['stability_label'].to(device)
        
        # 前向传播
        outputs = model(inputs)
        
        if model_type in ['cls_only', 'twostream_cls_only']:
            # 仅分类模型
            cls_logits = outputs
            seg_logits = None
            seg_targets = None
        else:
            # 分类+分割模型
            cls_logits, seg_logits = outputs
            seg_targets = batch['affected_mask'].to(device)
        
        # 更新指标
        metric_calc.update(cls_logits, seg_logits, cls_targets, seg_targets)
    
    return metric_calc.compute()


def find_best_checkpoint(checkpoint_dir: Path, best_only: bool = False) -> Optional[Path]:
    """
    找到目录中的最佳检查点
    
    Args:
        checkpoint_dir: 检查点目录
        best_only: 是否只查找带有 'best' 的检查点
        
    Returns:
        best_checkpoint: 最佳检查点路径
    """
    # 优先查找 *_best.pth 或 *best*.pth
    best_patterns = ['*_best.pth', '*best*.pth']
    for pattern in best_patterns:
        matches = list(checkpoint_dir.glob(pattern))
        if matches:
            # 返回最新的
            return max(matches, key=lambda p: p.stat().st_mtime)
    
    # 如果设置了 best_only，则不返回普通 epoch checkpoint
    if best_only:
        return None
    
    # 否则返回最新的 epoch checkpoint
    epoch_checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
    if epoch_checkpoints:
        return max(epoch_checkpoints, key=lambda p: p.stat().st_mtime)
    
    return None


def evaluate_all_models(
    checkpoint_base_dir: Path,
    test_dir: str,
    device: torch.device,
    batch_size: int = 32,
    output_dir: Optional[Path] = None,
    best_only: bool = True,
) -> Dict[str, Dict]:
    """
    评估所有模型
    
    Args:
        checkpoint_base_dir: 检查点基础目录
        test_dir: 测试数据目录
        device: 计算设备
        batch_size: 批次大小
        output_dir: 输出目录
        best_only: 是否只评估有 'best' 检查点的模型
        
    Returns:
        all_results: 所有模型的评估结果
    """
    all_results = {}
    
    # 查找所有模型目录
    model_dirs = [d for d in checkpoint_base_dir.iterdir() 
                  if d.is_dir() and d.name.startswith('transunet')]
    
    print(f"\n找到 {len(model_dirs)} 个模型目录")
    if best_only:
        print("模式: 只评估带有 'best' 检查点的模型")
    else:
        print("模式: 评估所有模型（包括只有 epoch checkpoint 的）")
    
    skipped_count = 0
    
    for model_dir in sorted(model_dirs):
        model_name = model_dir.name
        
        # 找到最佳检查点
        best_ckpt = find_best_checkpoint(model_dir, best_only=best_only)
        if best_ckpt is None:
            if best_only:
                skipped_count += 1
            else:
                print(f"\n跳过 {model_name}: 未找到检查点")
            continue
        
        print(f"\n{'='*60}")
        print(f"评估模型: {model_name}")
        print(f"{'='*60}")
        print(f"  使用检查点: {best_ckpt.name}")
        
        try:
            # 加载模型
            model, model_type, is_rgbd = load_model(str(best_ckpt), device)
            print(f"  模型类型: {model_type}, RGBD: {is_rgbd}")
            
            # 加载数据集
            if is_rgbd:
                transform = get_val_transform_with_depth()
                dataset = BoxWorldDatasetWithDepth([test_dir], transform=transform)
            else:
                transform = get_val_transform()
                dataset = BoxWorldDataset([test_dir], transform=transform)
            
            # 评估
            metrics = evaluate_model(model, model_type, dataset, device, batch_size)
            all_results[model_name] = metrics
            
            # 打印关键指标
            print(f"\n  论文指标:")
            print(f"    Safety Precision: {metrics['safety_precision']:.4f}")
            print(f"    F1-Score:         {metrics['f1_score']:.4f}")
            if 'miou_affected' in metrics:
                print(f"    mIoU (Affected):  {metrics['miou_affected']:.4f}")
                print(f"    Boundary F:       {metrics['boundary_f_measure']:.4f}")
            
            # 保存单个模型的结果到 JSON 文件
            if output_dir is not None:
                results_dir = output_dir / 'results'
                results_dir.mkdir(parents=True, exist_ok=True)
                
                model_result_file = results_dir / f'{model_name}.json'
                serializable_metrics = {
                    k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in metrics.items()
                }
                with open(model_result_file, 'w') as f:
                    json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
                print(f"  结果已保存: {model_result_file}")
            
            # 释放模型内存
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if best_only and skipped_count > 0:
        print(f"\n跳过了 {skipped_count} 个没有 'best' 检查点的模型")
    
    return all_results


def print_summary_table(all_results: Dict[str, Dict]):
    """
    打印汇总表格
    """
    print("\n" + "="*100)
    print("评估结果汇总")
    print("="*100)
    
    # 表头
    header = f"{'模型名称':<50} {'Safety Prec':>12} {'F1':>8} {'mIoU':>8} {'Boundary F':>12}"
    print(header)
    print("-"*100)
    
    # 按模型类型分组
    model_groups = {
        'RGB Fusion': [],
        'RGB Cls Only': [],
        'RGB Seg From Encoded': [],
        'RGBD Fusion': [],
        'RGBD Cls Only': [],
        'RGBD Seg From Encoded': [],
        'TwoStream': [],
        'Other': [],
    }
    
    for model_name, metrics in all_results.items():
        if 'twostream' in model_name:
            group = 'TwoStream'
        elif 'rgbd' in model_name:
            if 'fusion' in model_name:
                group = 'RGBD Fusion'
            elif 'cls_only' in model_name:
                group = 'RGBD Cls Only'
            elif 'seg_from_encoded' in model_name:
                group = 'RGBD Seg From Encoded'
            else:
                group = 'Other'
        else:
            if 'fusion' in model_name:
                group = 'RGB Fusion'
            elif 'cls_only' in model_name:
                group = 'RGB Cls Only'
            elif 'seg_from_encoded' in model_name:
                group = 'RGB Seg From Encoded'
            else:
                group = 'Other'
        
        model_groups[group].append((model_name, metrics))
    
    # 打印每个分组
    for group_name, models in model_groups.items():
        if not models:
            continue
            
        print(f"\n[{group_name}]")
        for model_name, metrics in sorted(models):
            safety_prec = metrics.get('safety_precision', 0)
            f1 = metrics.get('f1_score', 0)
            miou = metrics.get('miou_affected', 0)
            boundary_f = metrics.get('boundary_f_measure', 0)
            
            # 如果是 cls_only 模型，分割指标显示 N/A
            if 'cls_only' in model_name:
                row = f"{model_name:<50} {safety_prec:>12.4f} {f1:>8.4f} {'N/A':>8} {'N/A':>12}"
            else:
                row = f"{model_name:<50} {safety_prec:>12.4f} {f1:>8.4f} {miou:>8.4f} {boundary_f:>12.4f}"
            print(row)
    
    print("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser(description='论文评估指标脚本')
    
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='单个模型检查点路径')
    parser.add_argument('--checkpoint_dir', type=str, 
                        default='/DATA/disk0/hs_25/pa/checkpoints',
                        help='检查点基础目录（用于批量评估）')
    parser.add_argument('--all', action='store_true',
                        help='评估所有模型')
    parser.add_argument('--best_only', action='store_true', default=True,
                        help='只评估带有 best 检查点的模型 (默认: True)')
    parser.add_argument('--include_epoch', action='store_true',
                        help='也评估只有 epoch checkpoint 的模型 (覆盖 --best_only)')
    parser.add_argument('--test_dir', type=str, 
                        default='/DATA/disk0/hs_25/pa/all_dataset/test',
                        help='测试数据目录')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备')
    parser.add_argument('--output_dir', type=str, default='paper_eval_results',
                        help='输出目录')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='分类/分割阈值')
    
    args = parser.parse_args()
    
    # 设备
    device = get_device(args.device)
    
    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.all:
        # 评估所有模型
        checkpoint_dir = Path(args.checkpoint_dir)
        if not checkpoint_dir.exists():
            print(f"错误: 检查点目录不存在: {checkpoint_dir}")
            return
        
        # 确定是否只评估 best 检查点
        best_only = args.best_only and not args.include_epoch
        
        all_results = evaluate_all_models(
            checkpoint_dir,
            args.test_dir,
            device,
            args.batch_size,
            output_dir,
            best_only=best_only,
        )
        
        # 打印汇总表格
        print_summary_table(all_results)
        
        # 保存汇总结果
        results_dir = output_dir / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        summary_file = results_dir / 'summary.json'
        # 转换为可序列化格式
        serializable_results = {}
        for model_name, metrics in all_results.items():
            serializable_results[model_name] = {
                k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in metrics.items()
            }
        
        with open(summary_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"结果保存位置:")
        print(f"  汇总文件: {summary_file}")
        print(f"  单独文件: {results_dir}/<模型名>.json")
        print(f"  共 {len(all_results)} 个模型评估结果")
        print(f"{'='*60}")
        
    elif args.checkpoint:
        # 评估单个模型
        checkpoint_path = Path(args.checkpoint)
        
        # 如果是目录，查找最佳检查点
        if checkpoint_path.is_dir():
            best_ckpt = find_best_checkpoint(checkpoint_path)
            if best_ckpt is None:
                print(f"错误: 在 {checkpoint_path} 中未找到检查点")
                return
            checkpoint_path = best_ckpt
        
        if not checkpoint_path.exists():
            print(f"错误: 检查点不存在: {checkpoint_path}")
            return
        
        print(f"\n{'='*60}")
        print(f"评估模型: {checkpoint_path}")
        print(f"{'='*60}")
        
        # 加载模型
        model, model_type, is_rgbd = load_model(str(checkpoint_path), device)
        print(f"模型类型: {model_type}")
        print(f"RGBD 输入: {is_rgbd}")
        print(f"模型参数量: {model.get_num_params() / 1e6:.2f}M")
        
        # 加载数据集
        print(f"\n加载测试数据集: {args.test_dir}")
        if is_rgbd:
            transform = get_val_transform_with_depth()
            dataset = BoxWorldDatasetWithDepth([args.test_dir], transform=transform)
        else:
            transform = get_val_transform()
            dataset = BoxWorldDataset([args.test_dir], transform=transform)
        print(f"测试集样本数: {len(dataset)}")
        
        # 评估
        print(f"\n开始评估...")
        metrics = evaluate_model(model, model_type, dataset, device, args.batch_size)
        
        # 打印结果
        print(f"\n{'='*60}")
        print("论文评估指标")
        print(f"{'='*60}")
        
        print("\n【宏观指标 (Macro-Metrics)】- 全局稳定性预测")
        print("-"*40)
        print(f"  Safety Precision (核心):  {metrics['safety_precision']:.4f}")
        print(f"    - 定义: TP_safe / (TP_safe + FP_safe)")
        print(f"    - 意义: 避免将危险误判为安全")
        print(f"  Safety Recall:            {metrics['safety_recall']:.4f}")
        print(f"  F1-Score:                 {metrics['f1_score']:.4f}")
        print(f"  Accuracy:                 {metrics['accuracy']:.4f}")
        
        if model_type not in ['cls_only', 'twostream_cls_only']:
            print("\n【微观指标 (Micro-Metrics)】- 受影响区域分割")
            print("-"*40)
            print(f"  mIoU (Affected Class):    {metrics['miou_affected']:.4f}")
            print(f"    - 定义: 受影响区域的平均交并比")
            print(f"  Boundary F-measure:       {metrics['boundary_f_measure']:.4f}")
            print(f"    - 定义: 边界对齐精度")
            print(f"  Global IoU:               {metrics['global_iou']:.4f}")
            print(f"  Dice Score:               {metrics['dice']:.4f}")
        
        print("\n【混淆矩阵】")
        print("-"*40)
        cm = metrics['confusion_matrix']
        print(f"  TP_safe (预测安全,实际安全): {cm['tp_safe']}")
        print(f"  FP_safe (预测安全,实际危险): {cm['fp_safe']} ← 危险误判!")
        print(f"  TN_safe (预测危险,实际危险): {cm['tn_safe']}")
        print(f"  FN_safe (预测危险,实际安全): {cm['fn_safe']}")
        
        print("\n【样本统计】")
        print("-"*40)
        print(f"  总样本数:   {metrics['n_samples']}")
        print(f"  稳定样本:   {metrics['n_stable_samples']}")
        print(f"  不稳定样本: {metrics['n_unstable_samples']}")
        
        # 保存结果
        results_dir = output_dir / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        model_name = checkpoint_path.parent.name if checkpoint_path.parent.name.startswith('transunet') else checkpoint_path.stem
        result_file = results_dir / f'{model_name}.json'
        
        # 转换为可序列化格式
        serializable_metrics = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in metrics.items()
        }
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {result_file}")
        
    else:
        parser.print_help()
        print("\n请指定 --checkpoint 或 --all 参数")


if __name__ == '__main__':
    main()
