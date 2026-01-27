#!/usr/bin/env python
"""
RGBD 单张场景多箱子推理脚本

功能:
- 自动从指定 checkpoint 目录中选择“最新”的 best checkpoint
- 使用 RGBD SegFromEncoded Tiny 模型
- 对一张 RGB 图、对应的 depth.npy，以及每个箱子的掩码逐一进行推理
- 输出每个箱子的稳定性预测和受影响区域，并保存可视化结果

用法示例:
    python inference_rgbd_real.py \\
        --checkpoint_dir /DATA/disk0/hs_25/pa/checkpoints/transunet_rgbd_seg_from_encoded_tiny_h100 \\
        --image_dir /home/hs_25/projs/PA-VWP/real_images \\
        --device cuda

当前默认:
- checkpoint_dir: /DATA/disk0/hs_25/pa/checkpoints/transunet_rgbd_seg_from_encoded_tiny_h100
- image_dir:      /home/hs_25/projs/PA-VWP/real_images
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# 添加项目根目录到路径，便于 import
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from train.utils import get_device  # noqa: E402
from train.dataset_with_depth import ValTransformWithDepth  # noqa: E402
from train.dataset import get_val_transform  # RGB版本的transform  # noqa: E402
from test_models import load_model  # 使用支持 RGBD 的通用加载函数  # noqa: E402


def _normalize_depth_minmax(depth: np.ndarray) -> np.ndarray:
    """
    按训练时的默认方式对深度图做 Min-Max 归一化到 [0, 1]。
    逻辑参考 train/dataset_with_depth.py 中的 _normalize_depth('minmax')。
    """
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    d_min = depth.min()
    d_max = depth.max()
    if d_max - d_min > 1e-6:
        depth = (depth - d_min) / (d_max - d_min)
    else:
        depth = np.zeros_like(depth)
    return depth.astype(np.float32)


def find_latest_best_checkpoint(checkpoint_dir: Path) -> Path:
    """
    在给定目录中查找所有 *best.pth，按照 epoch 号排序，返回 epoch 最大的那个。
    目录中常见命名: checkpoint_epoch_20_best.pth
    """
    ckpts = sorted(checkpoint_dir.glob("*best.pth"))
    if not ckpts:
        raise FileNotFoundError(f"在目录 {checkpoint_dir} 中未找到任何 '*best.pth' checkpoint")

    def extract_epoch(path: Path) -> int:
        name = path.stem  # 去掉 .pth
        parts = name.split("_")
        for i, p in enumerate(parts):
            if p == "epoch" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return 0

    ckpts_with_epoch = [(extract_epoch(p), p) for p in ckpts]
    ckpts_with_epoch.sort(key=lambda x: x[0])
    best_epoch, best_path = ckpts_with_epoch[-1]
    print(f"在 {checkpoint_dir} 中找到 {len(ckpts)} 个 best checkpoint，"
          f"选择 epoch={best_epoch} 的 {best_path.name} 作为最新模型")
    return best_path


def preprocess_single_box(
    rgb_path: Path,
    mask_path: Path,
    img_size: Tuple[int, int] = (480, 640),
    use_depth: bool = False,
    depth_path: Path = None,
) -> torch.Tensor:
    """
    将一张 RGB 图 + (可选)深度图 + 单个箱子掩码 预处理为模型输入张量。
    
    Args:
        rgb_path: RGB图像路径
        mask_path: 掩码路径
        img_size: 图像尺寸 (H, W)
        use_depth: 是否使用深度通道
        depth_path: 深度文件路径（当use_depth=True时必需）
    
    Returns:
        输入张量: (1, 4, H, W) 如果use_depth=False, 或 (1, 5, H, W) 如果use_depth=True
    """
    # 加载 RGB
    rgb_pil = Image.open(rgb_path).convert("RGB")
    
    # 加载掩码
    mask_pil = Image.open(mask_path).convert("L")

    if use_depth:
        # RGBD 模式: RGB(3) + Depth(1) + Mask(1) = 5通道
        if depth_path is None or not depth_path.exists():
            raise FileNotFoundError(f"RGBD模式需要深度文件，但未找到: {depth_path}")
        
        # 加载 & 归一化深度
        depth = np.load(depth_path)
        depth = _normalize_depth_minmax(depth)

        # 使用RGBD变换
        val_transform = ValTransformWithDepth(img_size=img_size)
        rgb_tensor, depth_tensor, [mask_tensor] = val_transform(
            rgb_pil, depth, [mask_pil]
        )

        # 组合输入: (5, H, W)
        input_tensor = torch.cat([rgb_tensor, depth_tensor, mask_tensor], dim=0)
    else:
        # RGB 模式: RGB(3) + Mask(1) = 4通道
        val_transform = get_val_transform(img_size=img_size)
        rgb_tensor, [mask_tensor] = val_transform(rgb_pil, [mask_pil])
        
        # 组合输入: (4, H, W)
        input_tensor = torch.cat([rgb_tensor, mask_tensor], dim=0)
    
    return input_tensor.unsqueeze(0)  # (1, C, H, W)


def save_four_images(
    rgb: np.ndarray,
    removed_mask: np.ndarray,
    seg_mask: np.ndarray = None,
    cls_pred: int = None,
    cls_prob: float = None,
    box_dir: Path = None,
    use_chinese: bool = True,
):
    """
    为单个箱子保存四张独立的小图到指定文件夹。
    
    Args:
        rgb: RGB图像 (H, W, 3), uint8
        removed_mask: 被移除箱子掩码 (H, W), float [0,1]
        seg_mask: 预测受影响区域掩码 (H, W), float [0,1]，如果为None则只显示RGB
        cls_pred: 分类预测 (0=不稳定, 1=稳定)
        cls_prob: 分类概率
        box_dir: 保存图片的文件夹路径
        use_chinese: 是否使用中文标签
    """
    box_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置中文字体
    try:
        import matplotlib.font_manager as fm
        chinese_fonts = []
        for font in fm.fontManager.ttflist:
            font_name = font.name.lower()
            if any(keyword in font_name for keyword in ['simhei', 'simsun', 'yahei', 'microsoft', 'noto', 'wenquanyi', 'droid', 'source han', 'adobe']):
                chinese_fonts.append(font.name)
        if chinese_fonts:
            plt.rcParams['font.sans-serif'] = chinese_fonts[:3] + ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        else:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 原始RGB图
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(rgb)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(box_dir / "01_original_rgb.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 被移除箱子掩码
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(rgb)
    ax.imshow(removed_mask, alpha=0.5, cmap='Greens')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(box_dir / "02_removed_box.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. 分类结果（英文）
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    color = 'green' if cls_pred == 1 else 'red'
    stability_text = 'Stable' if cls_pred == 1 else 'Unstable'
    ax.text(0.5, 0.5, f'{stability_text}\nConfidence: {cls_prob:.2%}',
            ha='center', va='center', fontsize=24, color=color,
            transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(box_dir / "03_classification.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. 预测受影响区域（如果有分割结果）
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(rgb)
    if seg_mask is not None:
        ax.imshow(seg_mask, alpha=0.6, cmap='Reds')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(box_dir / "04_affected_region.png", dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_image(
    rgb: np.ndarray,
    mask_results: List[Dict],
    output_path: Path,
    use_chinese: bool = True,
):
    """
    创建汇总图：把所有箱子的掩码叠加在RGB图上，红色表示不稳定，绿色表示稳定。
    在每个掩码上显示置信度。
    
    Args:
        rgb: RGB图像 (H, W, 3), uint8
        mask_results: 每个箱子的结果列表，每个元素包含:
            - 'mask': 掩码数组 (H, W), float [0,1]
            - 'cls_pred': 分类预测 (0=不稳定, 1=稳定)
            - 'cls_prob': 分类置信度
            - 'box_name': 箱子名称
        output_path: 保存路径
        use_chinese: 是否使用中文标签
    """
    # 设置中文字体
    try:
        import matplotlib.font_manager as fm
        chinese_fonts = []
        for font in fm.fontManager.ttflist:
            font_name = font.name.lower()
            if any(keyword in font_name for keyword in ['simhei', 'simsun', 'yahei', 'microsoft', 'noto', 'wenquanyi', 'droid', 'source han', 'adobe']):
                chinese_fonts.append(font.name)
        if chinese_fonts:
            plt.rcParams['font.sans-serif'] = chinese_fonts[:3] + ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        else:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    # 确保RGB图像是uint8格式，范围0-255
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
    
    # 根据图像实际尺寸设置figsize，保持宽高比
    height, width = rgb.shape[:2]
    aspect_ratio = width / height
    fig_height = 9
    fig_width = fig_height * aspect_ratio
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    
    ax.imshow(rgb)
    
    # 创建颜色叠加层：绿色=稳定，红色=不稳定
    overlay = np.zeros((rgb.shape[0], rgb.shape[1], 3), dtype=np.float32)
    
    for result in mask_results:
        mask = result['mask']
        cls_pred = result['cls_pred']
        cls_prob = result.get('cls_prob', 0.0)
        
        # 根据稳定性选择颜色：绿色(稳定)或红色(不稳定)
        if cls_pred == 1:  # 稳定 - 绿色
            overlay[:, :, 1] = np.maximum(overlay[:, :, 1], mask * 0.5)  # 绿色通道，使用max避免叠加过暗
        else:  # 不稳定 - 红色
            overlay[:, :, 0] = np.maximum(overlay[:, :, 0], mask * 0.5)  # 红色通道
        
        # 计算掩码中心位置用于显示文本
        mask_binary = (mask > 0.5).astype(np.uint8)
        if mask_binary.sum() > 0:
            # 找到掩码区域的中心点
            y_coords, x_coords = np.where(mask_binary > 0)
            center_y = int(y_coords.mean())
            center_x = int(x_coords.mean())
            
            # 在掩码中心显示置信度
            text_color = 'white' if cls_pred == 1 else 'yellow'  # 稳定用白色，不稳定用黄色（更醒目）
            ax.text(center_x, center_y, f'{cls_prob:.2f}',
                   ha='center', va='center',
                   fontsize=12, fontweight='bold',
                   color=text_color,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='none'))
    
    # 限制叠加值不超过1.0
    overlay = np.clip(overlay, 0, 1)
    
    # 叠加颜色层，使用较低的alpha值避免背景变暗
    ax.imshow(overlay, alpha=0.4)
    
    ax.axis('off')
    ax.set_xlim(0, rgb.shape[1])
    ax.set_ylim(rgb.shape[0], 0)  # 注意y轴是反向的
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()


@torch.no_grad()
def infer_for_all_boxes(
    model: torch.nn.Module,
    device: torch.device,
    image_dir: Path,
    output_dir: Path,
    threshold: float = 0.5,
    is_rgbd: bool = True,
) -> None:
    """
    对指定目录中的一张 RGB + 深度 + 多个箱子掩码依次推理。

    期望目录结构:
        image_dir/
          capture_rgb.png
          capture_depth.npy
          box_masks/
            box_01.png
            box_02.png
            ...
    
    输出结构:
        output_dir/
          box_01/
            01_original_rgb.png
            02_removed_box.png
            03_classification.png
            04_affected_region.png
          box_02/
            ...
          summary.png  # 汇总图
          results.txt  # 文本摘要
    """
    rgb_path = image_dir / "capture_rgb.png"
    depth_path = image_dir / "capture_depth.npy"
    mask_dir = image_dir / "box_masks"

    if not rgb_path.exists():
        raise FileNotFoundError(f"未找到 RGB 图像: {rgb_path}")
    if not depth_path.exists():
        raise FileNotFoundError(f"未找到深度文件: {depth_path}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"未找到箱子掩码目录: {mask_dir}")

    mask_paths: List[Path] = sorted(mask_dir.glob("*.png"))
    if not mask_paths:
        raise FileNotFoundError(f"在目录 {mask_dir} 中未找到任何 *.png 掩码文件")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"开始推理，场景目录: {image_dir}")
    print(f"  RGB:   {rgb_path}")
    print(f"  Depth: {depth_path}")
    print(f"  共 {len(mask_paths)} 个箱子掩码")

    # 加载原始RGB图像（用于可视化）
    rgb_original = np.array(Image.open(rgb_path).convert('RGB'))
    
    results = []
    mask_results = []  # 用于汇总图

    for idx, mask_path in enumerate(mask_paths, start=1):
        box_name = mask_path.stem  # 如 box_01
        print(f"\n[{idx}/{len(mask_paths)}] 处理箱子掩码: {box_name}")

        # 预处理（根据模型类型决定是否使用深度）
        input_tensor = preprocess_single_box(
            rgb_path=rgb_path,
            mask_path=mask_path,
            img_size=(480, 640),
            use_depth=is_rgbd,
            depth_path=depth_path if is_rgbd else None,
        ).to(device)

        # 前向
        outputs = model(input_tensor)
        
        # 检测模型输出类型：可能是 (cls_logits, seg_logits) 或只有 cls_logits
        if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
            cls_logits, seg_logits = outputs
            has_seg = True
        else:
            # cls_only 模型只返回分类结果
            cls_logits = outputs
            seg_logits = None
            has_seg = False

        cls_prob = torch.sigmoid(cls_logits.squeeze(-1)).item()
        cls_pred = int(cls_prob > threshold)
        
        if has_seg:
            seg_mask = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
        else:
            seg_mask = None

        stability_label = "稳定(1)" if cls_pred == 1 else "不稳定(0)"
        print(f"  分类结果: {stability_label}, 置信度={cls_prob:.4f}")

        results.append(
            {
                "box": box_name,
                "cls_pred": cls_pred,
                "cls_prob": float(cls_prob),
            }
        )

        # 加载并调整掩码尺寸以匹配原始RGB
        removed_mask_pil = Image.open(mask_path).convert('L')
        removed_mask = np.array(removed_mask_pil.resize(
            (rgb_original.shape[1], rgb_original.shape[0]), Image.NEAREST
        )).astype(np.float32) / 255.0
        
        # 调整seg_mask尺寸（如果有分割结果）
        if seg_mask is not None:
            try:
                BILINEAR = Image.Resampling.BILINEAR
            except AttributeError:
                BILINEAR = Image.BILINEAR
            seg_mask_resized = np.array(
                Image.fromarray((seg_mask * 255).astype(np.uint8)).resize(
                    (rgb_original.shape[1], rgb_original.shape[0]), BILINEAR
                )
            ) / 255.0
        else:
            seg_mask_resized = None

        # 为每个箱子创建文件夹并保存四张图
        box_dir = output_dir / box_name
        save_four_images(
            rgb_original,
            removed_mask,
            seg_mask_resized,
            cls_pred,
            cls_prob,
            box_dir,
            use_chinese=True,
        )
        
        # 保存掩码信息用于汇总图
        mask_results.append({
            'mask': removed_mask,
            'cls_pred': cls_pred,
            'cls_prob': float(cls_prob),
            'box_name': box_name,
        })

    # 创建汇总图
    print(f"\n生成汇总图...")
    summary_path = output_dir / "summary.png"
    create_summary_image(rgb_original, mask_results, summary_path, use_chinese=True)

    # 保存一个简要的结果文本
    summary_text_path = output_dir / "results.txt"
    with open(summary_text_path, "w", encoding="utf-8") as f:
        for r in results:
            line = (
                f"{r['box']}: 分类={r['cls_pred']} "
                f"(稳定=1, 不稳定=0), 置信度={r['cls_prob']:.4f}\n"
            )
            f.write(line)

    print(f"\n全部箱子推理完成！")
    print(f"  - 每个箱子的四张图保存在: {output_dir}/<box_name>/")
    print(f"  - 汇总图: {summary_path}")
    print(f"  - 文本摘要: {summary_text_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="RGBD SegFromEncoded Tiny 模型 - 实机单场景多箱子推理脚本"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/DATA/disk0/hs_25/pa/checkpoints/transunet_rgbd_seg_from_encoded_tiny_h100",
        help="存放 best checkpoint 的目录(会自动选择最新的 best)",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/home/hs_25/projs/PA-VWP/real_images",
        help="包含 capture_rgb.png / capture_depth.npy / box_masks 的目录",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="推理设备 (cuda / cpu)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="分类为“稳定”的概率阈值",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="real_inference_results",
        help="保存可视化结果和预测摘要的目录",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)

    # 设备
    device = get_device(args.device)

    # 选择并加载最新 best checkpoint
    best_ckpt = find_latest_best_checkpoint(checkpoint_dir)
    print(f"\n加载模型权重: {best_ckpt}")
    model, model_type, is_rgbd = load_model(str(best_ckpt), device)
    print(f"模型类型: {model_type}, 使用 RGBD 输入: {is_rgbd}")

    if not is_rgbd:
        print("提示: 检测到当前 checkpoint 模型是 RGB 版本（4通道），将使用 RGB + Mask 输入。")

    # 推理
    infer_for_all_boxes(
        model=model,
        device=device,
        image_dir=image_dir,
        output_dir=output_dir,
        threshold=args.threshold,
        is_rgbd=is_rgbd,
    )


if __name__ == "__main__":
    main()

