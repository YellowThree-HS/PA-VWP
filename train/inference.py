"""
推理脚本 - 对单张图像进行预测

用法:
    python inference.py --checkpoint best.pth --image path/to/image.png --mask path/to/mask.png
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train.utils import get_device
from train.evaluate import load_model


def preprocess_image(
    rgb_path: str,
    mask_path: str,
    img_size: Tuple[int, int] = (480, 640),
) -> torch.Tensor:
    """预处理输入图像"""
    # 兼容新旧版本 Pillow
    try:
        BILINEAR = Image.Resampling.BILINEAR
        NEAREST = Image.Resampling.NEAREST
    except AttributeError:
        BILINEAR = Image.BILINEAR
        NEAREST = Image.NEAREST
    
    # 加载 RGB 图像
    rgb = Image.open(rgb_path).convert('RGB')
    rgb = rgb.resize((img_size[1], img_size[0]), BILINEAR)
    rgb = np.array(rgb).astype(np.float32) / 255.0
    
    # ImageNet 归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    rgb = (rgb - mean) / std
    
    # 加载掩码
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize((img_size[1], img_size[0]), NEAREST)
    mask = np.array(mask).astype(np.float32)
    mask = (mask > 127).astype(np.float32)
    
    # 组合输入
    rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
    input_tensor = torch.cat([rgb_tensor, mask_tensor], dim=0)
    
    return input_tensor.unsqueeze(0)  # (1, 4, H, W)


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[int, float, np.ndarray]:
    """
    运行推理
    
    Returns:
        cls_pred: 分类预测 (0=不稳定, 1=稳定)
        cls_prob: 稳定性概率
        seg_mask: 受影响区域掩码 (H, W)
    """
    model.eval()
    input_tensor = input_tensor.to(device)
    
    cls_logits, seg_logits = model(input_tensor)
    
    cls_prob = torch.sigmoid(cls_logits).item()
    cls_pred = int(cls_prob > threshold)
    
    seg_mask = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
    
    return cls_pred, cls_prob, seg_mask


def visualize_result(
    rgb_path: str,
    mask_path: str,
    cls_pred: int,
    cls_prob: float,
    seg_mask: np.ndarray,
    output_path: str = None,
    threshold: float = 0.5,
):
    """可视化预测结果"""
    # 尝试设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass
    
    # 兼容新旧版本 Pillow
    try:
        BILINEAR = Image.Resampling.BILINEAR
    except AttributeError:
        BILINEAR = Image.BILINEAR
    
    # 加载原始图像
    rgb = np.array(Image.open(rgb_path).convert('RGB'))
    removed_mask = np.array(Image.open(mask_path).convert('L'))
    removed_mask = (removed_mask > 127).astype(float)
    
    # 调整 seg_mask 尺寸
    seg_mask_resized = np.array(
        Image.fromarray((seg_mask * 255).astype(np.uint8)).resize(
            (rgb.shape[1], rgb.shape[0]), BILINEAR
        )
    ) / 255.0
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 原始图像
    axes[0].imshow(rgb)
    axes[0].set_title('原始图像', fontsize=14)
    axes[0].axis('off')
    
    # 被移除箱子
    axes[1].imshow(rgb)
    axes[1].imshow(removed_mask, alpha=0.5, cmap='Greens')
    axes[1].set_title('被移除箱子', fontsize=14)
    axes[1].axis('off')
    
    # 分类结果
    stability = '稳定' if cls_pred == 1 else '不稳定'
    color = 'green' if cls_pred == 1 else 'red'
    axes[2].text(0.5, 0.5, f'{stability}\n置信度: {cls_prob:.2%}',
                 ha='center', va='center', fontsize=20, color=color,
                 transform=axes[2].transAxes,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[2].set_title('稳定性预测', fontsize=14)
    axes[2].axis('off')
    
    # 受影响区域
    axes[3].imshow(rgb)
    axes[3].imshow(seg_mask_resized, alpha=0.6, cmap='Reds')
    axes[3].set_title('预测受影响区域', fontsize=14)
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"结果已保存到: {output_path}")
    else:
        plt.show()
        
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='TransUNet 推理脚本')
    
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--image', type=str, required=True, help='RGB 图像路径')
    parser.add_argument('--mask', type=str, required=True, help='被移除箱子掩码路径')
    parser.add_argument('--output', type=str, default=None, help='输出图像路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--threshold', type=float, default=0.5, help='分类阈值')
    
    args = parser.parse_args()
    
    # 设备
    device = get_device(args.device)
    
    # 加载模型
    print(f"加载模型: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    
    # 预处理输入
    print(f"处理图像: {args.image}")
    input_tensor = preprocess_image(args.image, args.mask)
    
    # 推理
    cls_pred, cls_prob, seg_mask = predict(model, input_tensor, device, args.threshold)
    
    # 打印结果
    stability = '稳定' if cls_pred == 1 else '不稳定'
    print(f"\n预测结果:")
    print(f"  稳定性: {stability}")
    print(f"  置信度: {cls_prob:.2%}")
    
    # 可视化
    output_path = args.output or 'prediction_result.png'
    visualize_result(
        args.image, args.mask,
        cls_pred, cls_prob, seg_mask,
        output_path, args.threshold
    )


if __name__ == '__main__':
    main()
