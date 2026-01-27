"""
推理脚本 - 对test数据集中的round进行推理并保存可视化结果

用法:
    python inference_rounds.py --checkpoint checkpoint_epoch_21_best.pth --test_dir /DATA/disk0/hs_25/pa/all_dataset/test --num_rounds 10
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from test_models import load_model
from train.dataset_with_depth import BoxWorldDatasetWithDepth, get_val_transform_with_depth
from train.utils import get_device


def load_round_data(round_dir: Path) -> Dict:
    """加载round的数据"""
    rgb_path = round_dir / 'rgb.png'
    depth_path = round_dir / 'depth.npy'
    mask_path = round_dir / 'mask.npy'
    visible_boxes_path = round_dir / 'visible_boxes.json'
    summary_path = round_dir / 'summary.json'
    removals_dir = round_dir / 'removals'
    
    if not rgb_path.exists() or not depth_path.exists() or not removals_dir.exists():
        return None
    
    data = {
        'round_dir': round_dir,
        'rgb_path': rgb_path,
        'depth_path': depth_path,
        'mask_path': mask_path if mask_path.exists() else None,
        'visible_boxes': None,
        'summary': None,
        'removals': []
    }
    
    # 加载可见箱子信息
    if visible_boxes_path.exists():
        with open(visible_boxes_path, 'r') as f:
            data['visible_boxes'] = json.load(f)
    
    # 加载summary
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            data['summary'] = json.load(f)
    
    # 加载所有removal数据
    for removal_dir in sorted(removals_dir.iterdir()):
        if not removal_dir.is_dir():
            continue
        
        removed_mask_path = removal_dir / 'removed_mask.png'
        affected_mask_path = removal_dir / 'affected_mask.png'
        result_path = removal_dir / 'result.json'
        
        if not all(p.exists() for p in [removed_mask_path, affected_mask_path, result_path]):
            continue
        
        with open(result_path, 'r') as f:
            result = json.load(f)
        
        data['removals'].append({
            'removal_id': removal_dir.name,
            'removed_mask_path': removed_mask_path,
            'affected_mask_path': affected_mask_path,
            'result': result,
        })
    
    return data


def visualize_mask_colored(mask: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """可视化掩码，每个箱子不同颜色，背景不上色"""
    # 获取所有唯一的ID（排除0，0是背景）
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids != 0]
    
    if len(unique_ids) == 0:
        return rgb.copy()
    
    # 创建彩色掩码
    colored_mask = rgb.copy()
    
    # 为每个ID生成一个颜色
    np.random.seed(42)  # 固定随机种子，确保颜色一致
    colors = {}
    for uid in unique_ids:
        # 生成鲜艳的颜色（避免太暗或太亮）
        hue = (uid * 137.508) % 360  # 使用黄金角度分布
        hsv = np.array([[[hue, 0.8, 0.9]]], dtype=np.uint8)
        rgb_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
        colors[int(uid)] = rgb_color
    
    # 应用颜色
    for uid, color in colors.items():
        mask_region = (mask == uid)
        colored_mask[mask_region] = color * 0.6 + rgb[mask_region] * 0.4  # 混合
    
    return colored_mask


@torch.no_grad()
def predict_single_removal(
    model: torch.nn.Module,
    round_data: Dict,
    removal_data: Dict,
    device: torch.device,
    img_size: Tuple[int, int] = (480, 640),
    threshold: float = 0.5,
    is_rgbd: bool = False,
) -> Dict:
    """对单个removal进行预测"""
    # 加载RGB图像
    rgb_pil = Image.open(round_data['rgb_path']).convert('RGB')
    rgb_pil = rgb_pil.resize((img_size[1], img_size[0]), Image.BILINEAR)
    rgb = np.array(rgb_pil).astype(np.float32) / 255.0
    
    # 加载深度图
    depth = np.load(round_data['depth_path'])
    # 归一化深度图
    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max - depth_min > 1e-6:
        depth = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth = np.zeros_like(depth)
    depth = cv2.resize(depth, (img_size[1], img_size[0]), interpolation=cv2.INTER_LINEAR)
    
    # 加载被移除箱子掩码
    removed_mask_pil = Image.open(removal_data['removed_mask_path']).convert('L')
    removed_mask_pil = removed_mask_pil.resize((img_size[1], img_size[0]), Image.NEAREST)
    removed_mask = np.array(removed_mask_pil).astype(np.float32) / 255.0
    
    # 加载真实受影响区域掩码
    affected_mask_pil = Image.open(removal_data['affected_mask_path']).convert('L')
    affected_mask_pil = affected_mask_pil.resize((img_size[1], img_size[0]), Image.NEAREST)
    affected_mask_gt = np.array(affected_mask_pil).astype(np.float32) / 255.0
    
    # 准备输入
    # ImageNet归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    rgb_normalized = (rgb - mean) / std
    
    # 转换为tensor
    rgb_tensor = torch.from_numpy(rgb_normalized).permute(2, 0, 1).float()
    depth_tensor = torch.from_numpy(depth).unsqueeze(0).float()
    removed_mask_tensor = torch.from_numpy(removed_mask).unsqueeze(0).float()
    
    # 组合输入
    if is_rgbd:
        # RGBD模型: RGB (3) + Depth (1) + Target Mask (1) = 5 channels
        input_tensor = torch.cat([rgb_tensor, depth_tensor, removed_mask_tensor], dim=0).unsqueeze(0).to(device)
    else:
        # RGB模型: RGB (3) + Target Mask (1) = 4 channels
        input_tensor = torch.cat([rgb_tensor, removed_mask_tensor], dim=0).unsqueeze(0).to(device)
    
    # 推理
    cls_logits, seg_logits = model(input_tensor)
    
    # 处理分类结果
    cls_prob = torch.sigmoid(cls_logits.squeeze(-1)).item()
    cls_pred = int(cls_prob > threshold)
    
    # 处理分割结果
    seg_pred = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
    seg_pred_binary = (seg_pred > threshold).astype(np.float32)
    
    # 真实标签
    cls_target = removal_data['result'].get('stability_label', 1 if removal_data['result'].get('is_stable', True) else 0)
    cls_correct = (cls_pred == cls_target)
    
    return {
        'rgb': rgb,
        'depth': depth,
        'removed_mask': removed_mask,
        'affected_mask_gt': affected_mask_gt,
        'affected_mask_pred': seg_pred_binary,
        'cls_pred': cls_pred,
        'cls_prob': cls_prob,
        'cls_target': cls_target,
        'cls_correct': cls_correct,
    }


def visualize_single_removal(
    round_data: Dict,
    removal_data: Dict,
    pred_result: Dict,
    output_dir: Path,
    mask: Optional[np.ndarray] = None,
):
    """可视化单个removal的预测结果，每个子图单独保存"""
    rgb = pred_result['rgb']
    depth = pred_result['depth']
    removed_mask = pred_result['removed_mask']
    affected_mask_gt = pred_result['affected_mask_gt']
    affected_mask_pred = pred_result['affected_mask_pred']
    cls_pred = pred_result['cls_pred']
    cls_prob = pred_result['cls_prob']
    cls_target = pred_result['cls_target']
    cls_correct = pred_result['cls_correct']
    
    removal_id = removal_data['removal_id']
    
    # 1. Original Image
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(rgb)
    ax.set_title('Original Image', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / f"{removal_id}_01_original.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Depth Map
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(depth, cmap='viridis')
    ax.set_title('Depth Map', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / f"{removal_id}_02_depth.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Mask (Different Colors for Each Box)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    if mask is not None:
        colored_mask = visualize_mask_colored(mask, rgb)
        ax.imshow(colored_mask)
    else:
        ax.imshow(rgb)
    ax.set_title('Mask (Colored by Box)', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / f"{removal_id}_03_mask_colored.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Prediction Result (text only)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    stability_text = 'Stable' if cls_pred == 1 else 'Unstable'
    target_text = 'Stable' if cls_target == 1 else 'Unstable'
    correct_text = '✓ Correct' if cls_correct else '✗ Wrong'
    result_text = f"Pred: {stability_text}\nConf: {cls_prob:.2%}\nGT: {target_text}\n{correct_text}"
    color = 'green' if cls_correct else 'red'
    ax.text(0.5, 0.5, result_text, ha='center', va='center', 
             fontsize=16, color=color, transform=ax.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_title('Prediction Result', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / f"{removal_id}_04_prediction_result.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Removed Box
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(rgb)
    ax.imshow(removed_mask, alpha=0.5, cmap='Greens')
    ax.set_title('Removed Box', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / f"{removal_id}_05_removed_box.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. Ground Truth Affected Region
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(rgb)
    ax.imshow(affected_mask_gt, alpha=0.5, cmap='Reds')
    ax.set_title('GT Affected Region', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / f"{removal_id}_06_gt_affected.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 7. Predicted Affected Region
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(rgb)
    ax.imshow(affected_mask_pred, alpha=0.5, cmap='Reds')
    ax.set_title('Pred Affected Region', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / f"{removal_id}_07_pred_affected.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 8. Predicted Mask (Binary)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(affected_mask_pred, cmap='Reds')
    ax.set_title('Predicted Mask (Binary)', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / f"{removal_id}_08_pred_mask_binary.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 9. Overlay: Ground Truth vs Prediction
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    overlay = np.zeros((*affected_mask_gt.shape, 3))
    overlay[..., 0] = affected_mask_pred  # Red: Prediction
    overlay[..., 1] = affected_mask_gt   # Green: Ground Truth
    ax.imshow(rgb)
    ax.imshow(overlay, alpha=0.5)
    ax.set_title('Overlay: Red=Pred, Green=GT, Yellow=Intersection', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / f"{removal_id}_09_overlay_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 10. IoU Visualization
    intersection = np.logical_and(affected_mask_pred > 0.5, affected_mask_gt > 0.5)
    union = np.logical_or(affected_mask_pred > 0.5, affected_mask_gt > 0.5)
    iou = np.sum(intersection) / (np.sum(union) + 1e-8)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(rgb)
    ax.imshow(intersection.astype(float), alpha=0.5, cmap='YlOrBr')
    ax.set_title(f'IoU: {iou:.3f}', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / f"{removal_id}_10_iou_{iou:.3f}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 11. Depth Map + Predicted Mask
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(depth, cmap='viridis')
    ax.imshow(affected_mask_pred, alpha=0.5, cmap='Reds')
    ax.set_title('Depth + Predicted Mask', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / f"{removal_id}_11_depth_pred_mask.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 12. Original Image + Predicted Mask
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(rgb)
    ax.imshow(affected_mask_pred, alpha=0.5, cmap='Reds')
    ax.set_title('Original + Predicted Mask', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / f"{removal_id}_12_original_pred_mask.png", dpi=150, bbox_inches='tight')
    plt.close()


def create_round_summary(
    round_data: Dict,
    all_predictions: List[Dict],
    output_path: Path,
    mask: Optional[np.ndarray] = None,
):
    """创建round的summary图（红色表示不稳定，绿色表示稳定）- 纯图片无文字"""
    rgb_path = round_data['rgb_path']
    rgb = np.array(Image.open(rgb_path).convert('RGB')).astype(np.float32) / 255.0
    
    # 调整尺寸以匹配mask
    if mask is not None:
        h, w = mask.shape[:2]
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 创建overlay
    overlay = rgb.copy()
    
    # 根据预测结果给每个removal的箱子着色
    for pred in all_predictions:
        removal_data = pred['removal_data']
        result = pred['pred_result']
        
        # 获取被移除的箱子信息
        removed_box = removal_data['result'].get('removed_box', {})
        prim_path = removed_box.get('prim_path', '')
        
        # 从visible_boxes中找到对应的箱子
        if round_data['visible_boxes'] and mask is not None:
            for box_info in round_data['visible_boxes']:
                if box_info.get('prim_path') == prim_path:
                    # 找到对应的mask ID
                    # 这里简化处理：如果mask存在，尝试找到对应的区域
                    # 实际应该使用id_to_labels，但测试数据可能没有
                    # 我们使用removed_mask来标记
                    removed_mask_path = removal_data['removed_mask_path']
                    removed_mask = np.array(Image.open(removed_mask_path).convert('L'))
                    if mask is not None:
                        removed_mask = cv2.resize(removed_mask, (mask.shape[1], mask.shape[0]), 
                                                  interpolation=cv2.INTER_NEAREST)
                        removed_region = (removed_mask > 127)
                        
                        # 根据预测结果着色
                        if result['cls_pred'] == 0:  # 不稳定 - 红色
                            color = np.array([1.0, 0.0, 0.0])
                            overlay[removed_region] = color * 0.6 + rgb[removed_region] * 0.4
                        else:  # 稳定 - 绿色
                            color = np.array([0.0, 1.0, 0.0])
                            overlay[removed_region] = color * 0.6 + rgb[removed_region] * 0.4
                    break
    
    # 只保存一张带红绿掩码的图片，无标题无文字
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(overlay)
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='对test数据集中的round进行推理')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径（如：checkpoint_epoch_21_best.pth）')
    parser.add_argument('--test_dir', type=str, 
                        default='/DATA/disk0/hs_25/pa/all_dataset/test',
                        help='测试数据目录')
    parser.add_argument('--num_rounds', type=int, default=10,
                        help='要推理的round数量')
    parser.add_argument('--round_name', type=str, default=None,
                        help='指定要处理的round名称（如round_013_32_southwest），如果指定则忽略num_rounds')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='分类阈值')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 设备
    device = get_device(args.device)
    print(f"使用设备: {device}")
    
    # 加载模型
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        # 尝试在checkpoints目录下查找
        possible_paths = [
            Path('/DATA/disk0/hs_25/pa/checkpoints/transunet_rgbd_seg_from_encoded_tiny_h100') / args.checkpoint,
            Path('/DATA/disk0/hs_25/pa/checkpoints/transunet_seg_from_encoded_tiny_h100') / args.checkpoint,
            Path(args.checkpoint),
        ]
        for p in possible_paths:
            if p.exists():
                checkpoint_path = p
                break
    
    if not checkpoint_path.exists():
        print(f"错误: 找不到检查点文件: {checkpoint_path}")
        return
    
    print(f"加载模型: {checkpoint_path}")
    
    # 使用标准加载函数
    model, model_type, is_rgbd = load_model(str(checkpoint_path), device, img_height=480, img_width=640)
    print(f"模型类型: {model_type}, RGBD: {is_rgbd}")
    
    # 加载测试数据目录
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        print(f"错误: 测试数据目录不存在: {test_dir}")
        return
    
    # 找到所有round目录
    round_dirs = [d for d in sorted(test_dir.iterdir()) 
                  if d.is_dir() and d.name.startswith('round_')]
    print(f"找到 {len(round_dirs)} 个round目录")
    
    # 选择round
    if args.round_name:
        # 指定了特定的round
        selected_rounds = [d for d in round_dirs if d.name == args.round_name]
        if not selected_rounds:
            print(f"错误: 找不到指定的round: {args.round_name}")
            return
        print(f"处理指定的round: {args.round_name}")
    else:
        # 随机选择round
        selected_rounds = random.sample(round_dirs, min(args.num_rounds, len(round_dirs)))
        print(f"随机选择了 {len(selected_rounds)} 个round进行推理")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 对每个round进行推理
    for round_idx, round_dir in enumerate(tqdm(selected_rounds, desc='处理rounds')):
        print(f"\n处理: {round_dir.name}")
        
        # 加载round数据
        round_data = load_round_data(round_dir)
        if round_data is None or len(round_data['removals']) == 0:
            print(f"  跳过: 数据不完整")
            continue
        
        # 加载mask（如果存在）
        mask = None
        if round_data['mask_path'] and round_data['mask_path'].exists():
            mask = np.load(round_data['mask_path'])
        
        # 创建round输出目录
        round_output_dir = output_dir / round_dir.name
        round_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 对每个removal进行推理
        all_predictions = []
        for removal_data in tqdm(round_data['removals'], desc=f'  {round_dir.name}'):
            # 预测
            pred_result = predict_single_removal(
                model, round_data, removal_data, device,
                threshold=args.threshold,
                is_rgbd=is_rgbd
            )
            
            # 为每个removal创建子目录
            removal_output_dir = round_output_dir / removal_data['removal_id']
            removal_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存预测结果（每个子图单独保存）
            visualize_single_removal(
                round_data, removal_data, pred_result,
                removal_output_dir, mask
            )
            
            all_predictions.append({
                'removal_data': removal_data,
                'pred_result': pred_result,
            })
        
        # 创建round summary
        summary_output_path = round_output_dir / 'summary.png'
        create_round_summary(round_data, all_predictions, summary_output_path, mask)
        
        # 保存统计信息
        stats = {
            'round_name': round_dir.name,
            'total_removals': len(all_predictions),
            'predictions': []
        }
        
        for pred in all_predictions:
            stats['predictions'].append({
                'removal_id': pred['removal_data']['removal_id'],
                'cls_pred': pred['pred_result']['cls_pred'],
                'cls_prob': pred['pred_result']['cls_prob'],
                'cls_target': pred['pred_result']['cls_target'],
                'cls_correct': pred['pred_result']['cls_correct'],
            })
        
        with open(round_output_dir / 'stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"  完成: {len(all_predictions)} 个removal")
    
    print(f"\n推理完成！结果保存在: {output_dir}")


if __name__ == '__main__':
    main()
