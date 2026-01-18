"""
模型评估脚本

用法:
    python evaluate.py --checkpoint checkpoints/transunet_small/checkpoint_best.pth
    python evaluate.py --checkpoint best.pth --data_dirs dataset3 dataset_messy_small
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train.config import Config, get_config
from train.dataset import BoxWorldDataset, get_val_transform
from train.models import TransUNet, TransUNetConfig
from train.models.transunet import transunet_base, transunet_small, transunet_tiny
from train.utils import get_device, MetricCalculator, load_checkpoint


def load_model(checkpoint_path: str, device: torch.device) -> TransUNet:
    """从检查点加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 尝试从检查点推断模型配置
    state_dict = checkpoint['model_state_dict']
    
    # 根据 Transformer 层数推断模型变体
    transformer_layers = sum(1 for k in state_dict.keys() if 'transformer.layers' in k and 'norm1.weight' in k)
    hidden_dim = state_dict['patch_embed.proj.weight'].shape[0]
    
    print(f"检测到模型配置: {transformer_layers} 层 Transformer, {hidden_dim} hidden dim")
    
    if transformer_layers == 4 and hidden_dim == 384:
        model = transunet_tiny(pretrained=False)
    elif transformer_layers == 6 and hidden_dim == 512:
        model = transunet_small(pretrained=False)
    elif transformer_layers == 12 and hidden_dim == 768:
        model = transunet_base(pretrained=False)
    else:
        # 创建自定义配置
        config = TransUNetConfig(
            num_layers=transformer_layers,
            hidden_dim=hidden_dim,
            pretrained=False,
        )
        model = TransUNet(config)
        
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def evaluate(
    model: TransUNet,
    dataset: BoxWorldDataset,
    device: torch.device,
    batch_size: int = 8,
    save_predictions: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """评估模型"""
    
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    metric_calc = MetricCalculator()
    all_predictions = []
    
    for batch in tqdm(loader, desc='评估中'):
        inputs = batch['input'].to(device)
        cls_targets = batch['stability_label'].to(device)
        seg_targets = batch['affected_mask'].to(device)
        
        # 前向传播
        cls_logits, seg_logits = model(inputs)
        
        # 更新指标
        metric_calc.update(cls_logits, seg_logits, cls_targets, seg_targets)
        
        # 收集预测结果
        if save_predictions:
            cls_probs = torch.sigmoid(cls_logits.squeeze(-1)).cpu().numpy()
            seg_probs = torch.sigmoid(seg_logits).cpu().numpy()
            
            for i in range(inputs.size(0)):
                all_predictions.append({
                    'rgb_path': batch['rgb_path'][i],
                    'removal_id': batch['removal_id'][i],
                    'cls_prob': float(cls_probs[i]),
                    'cls_pred': int(cls_probs[i] > 0.5),
                    'cls_target': int(cls_targets[i].item()),
                })
                
    # 计算指标
    metrics = metric_calc.compute()
    
    # 保存预测结果
    if save_predictions and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'predictions.json', 'w') as f:
            json.dump(all_predictions, f, indent=2)
            
    return metrics


@torch.no_grad()
def visualize_predictions(
    model: TransUNet,
    dataset: BoxWorldDataset,
    device: torch.device,
    output_dir: Path,
    num_samples: int = 20,
    threshold: float = 0.5,
):
    """可视化模型预测结果"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 尝试设置中文字体
    try:
        import matplotlib
        # Windows 系统
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass  # 如果字体设置失败，使用默认字体
    
    # 随机选择样本
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    # ImageNet 反归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    for idx in tqdm(indices, desc='生成可视化'):
        sample = dataset[int(idx)]
        
        inputs = sample['input'].unsqueeze(0).to(device)
        cls_target = sample['stability_label'].item()
        seg_target = sample['affected_mask'].numpy().squeeze()
        
        # 前向传播
        cls_logits, seg_logits = model(inputs)
        
        cls_prob = torch.sigmoid(cls_logits).item()
        cls_pred = int(cls_prob > threshold)
        seg_pred = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
        
        # 反归一化 RGB 图像
        rgb = sample['input'][:3].cpu()
        rgb = rgb * std + mean
        rgb = rgb.clamp(0, 1).permute(1, 2, 0).numpy()
        
        # 被移除箱子掩码
        removed_mask = sample['input'][3].numpy()
        
        # 创建可视化图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原始 RGB
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title('RGB 图像')
        axes[0, 0].axis('off')
        
        # 被移除箱子
        axes[0, 1].imshow(rgb)
        axes[0, 1].imshow(removed_mask, alpha=0.5, cmap='Greens')
        axes[0, 1].set_title('被移除箱子 (Target)')
        axes[0, 1].axis('off')
        
        # 分类结果
        result_text = f"预测: {'稳定' if cls_pred else '不稳定'} ({cls_prob:.2f})\n"
        result_text += f"真实: {'稳定' if cls_target else '不稳定'}"
        correct = '✓' if cls_pred == cls_target else '✗'
        color = 'green' if cls_pred == cls_target else 'red'
        
        axes[0, 2].text(0.5, 0.5, f"{correct}\n{result_text}", 
                        ha='center', va='center', fontsize=16, color=color,
                        transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('分类结果')
        axes[0, 2].axis('off')
        
        # 真实受影响区域
        axes[1, 0].imshow(rgb)
        axes[1, 0].imshow(seg_target, alpha=0.5, cmap='Reds')
        axes[1, 0].set_title('真实受影响区域')
        axes[1, 0].axis('off')
        
        # 预测受影响区域
        axes[1, 1].imshow(rgb)
        axes[1, 1].imshow(seg_pred, alpha=0.5, cmap='Reds')
        axes[1, 1].set_title('预测受影响区域')
        axes[1, 1].axis('off')
        
        # 分割对比
        seg_pred_binary = (seg_pred > threshold).astype(float)
        overlay = np.zeros((*seg_target.shape, 3))
        overlay[..., 0] = seg_pred_binary  # 红色: 预测
        overlay[..., 1] = seg_target       # 绿色: 真实
        # 黄色: 交集
        
        axes[1, 2].imshow(rgb)
        axes[1, 2].imshow(overlay, alpha=0.5)
        axes[1, 2].set_title('红=预测, 绿=真实, 黄=交集')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # 保存图像
        removal_id = sample['removal_id']
        save_path = output_dir / f'vis_{idx}_{removal_id}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    print(f"可视化结果已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='TransUNet 评估脚本')
    
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--data_dirs', type=str, nargs='+', 
                        default=['dataset_messy_small'], help='测试数据目录')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--output_dir', type=str, default='eval_results', help='输出目录')
    parser.add_argument('--visualize', action='store_true', help='生成可视化')
    parser.add_argument('--num_vis', type=int, default=20, help='可视化样本数')
    
    args = parser.parse_args()
    
    # 设备
    device = get_device(args.device)
    
    # 加载模型
    print(f"\n加载模型: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    print(f"模型参数量: {model.get_num_params() / 1e6:.2f}M")
    
    # 加载数据集
    print(f"\n加载数据集: {args.data_dirs}")
    transform = get_val_transform()
    dataset = BoxWorldDataset(args.data_dirs, transform=transform)
    
    # 评估
    print("\n开始评估...")
    output_dir = Path(args.output_dir)
    metrics = evaluate(
        model, dataset, device,
        batch_size=args.batch_size,
        save_predictions=True,
        output_dir=output_dir,
    )
    
    # 打印结果
    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)
    print(f"分类准确率: {metrics['cls_accuracy']:.4f}")
    print(f"分类精确率: {metrics['cls_precision']:.4f}")
    print(f"分类召回率: {metrics['cls_recall']:.4f}")
    print(f"分类 F1:    {metrics['cls_f1']:.4f}")
    print(f"分割 IoU:   {metrics['seg_iou']:.4f}")
    print(f"分割 Dice:  {metrics['seg_dice']:.4f}")
    
    # 保存指标
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
        
    # 可视化
    if args.visualize:
        print("\n生成可视化...")
        visualize_predictions(
            model, dataset, device,
            output_dir=output_dir / 'visualizations',
            num_samples=args.num_vis,
        )
        
    print(f"\n结果已保存到: {output_dir}")


if __name__ == '__main__':
    main()
