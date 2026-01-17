"""
CollapseNet 推理脚本
"""

import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model_collapse import CollapseNet


def load_image(rgb_path, mask_path, image_size=224):
    """加载并预处理图像"""
    # 读取图像
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 保存原始尺寸用于后处理
    orig_h, orig_w = rgb.shape[:2]

    # Resize
    rgb = cv2.resize(rgb, (image_size, image_size))
    mask = cv2.resize(mask, (image_size, image_size))

    # 归一化
    mask = (mask > 127).astype(np.float32)
    rgb_norm = rgb.astype(np.float32) / 255.0

    # 构建4通道输入
    combined = np.concatenate([rgb_norm, mask[:, :, np.newaxis]], axis=2)  # (H, W, 4)

    # 转为tensor并归一化
    x = torch.from_numpy(combined).permute(2, 0, 1).unsqueeze(0)  # (1, 4, H, W)
    mean = torch.tensor([0.485, 0.456, 0.406, 0.0]).view(4, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225, 1.0]).view(4, 1, 1)
    x = (x - mean) / std

    return x, (orig_h, orig_w)


def predict(model, x, device):
    """推理预测"""
    model.eval()

    x = x.to(device)

    with torch.no_grad():
        cls_pred, seg_pred = model(x)

    # 分类概率
    cls_prob = torch.sigmoid(cls_pred).item()

    # 分割掩码 (resize回原始尺寸)
    seg_mask = torch.sigmoid(seg_pred).squeeze().cpu().numpy()

    return cls_prob, seg_mask


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    model = CollapseNet(
        backbone=args.backbone,
        pretrained=False
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from {args.checkpoint}")

    # 推理模式
    if os.path.isfile(args.input):
        # 单张图像
        rgb_path = os.path.join(args.input, "initial", "rgb.png")
        mask_path = os.path.join(
            args.input, "removals", str(args.sample_idx), "mask.png"
        )

        x, orig_size = load_image(rgb_path, mask_path, args.image_size)

        cls_prob, seg_mask = predict(model, x, device)

        print(f"\n=== Result ===")
        print(f"Stability: {'Stable' if cls_prob < 0.5 else 'Unstable'}")
        print(f"Collapse probability: {cls_prob:.4f}")

        # 保存分割掩码
        seg_mask_resized = cv2.resize(seg_mask, orig_size[::-1])
        seg_mask_uint8 = (seg_mask_resized * 255).astype(np.uint8)

        output_path = os.path.join(args.output, "affected_mask.png")
        cv2.imwrite(output_path, seg_mask_uint8)
        print(f"Segmentation mask saved to {output_path}")

    else:
        # 批量推理
        os.makedirs(args.output, exist_ok=True)

        for round_dir in tqdm(sorted(os.listdir(args.input))):
            round_path = os.path.join(args.input, round_dir)
            if not os.path.isdir(round_path):
                continue

            rgb_path = os.path.join(round_path, "initial", "rgb.png")
            if not os.path.exists(rgb_path):
                continue

            removals_dir = os.path.join(round_path, "removals")
            if not os.path.exists(removals_dir):
                continue

            for sample_idx in sorted(os.listdir(removals_dir)):
                sample_dir = os.path.join(removals_dir, sample_idx)
                if not os.path.isdir(sample_dir):
                    continue

                mask_path = os.path.join(sample_dir, "mask.png")
                if not os.path.exists(mask_path):
                    continue

                try:
                    x, _ = load_image(rgb_path, mask_path, args.image_size)
                    cls_prob, seg_mask = predict(model, x, device)

                    # 保存结果
                    round_output = os.path.join(args.output, round_dir, sample_idx)
                    os.makedirs(round_output, exist_ok=True)

                    # 保存掩码
                    seg_mask_resized = cv2.resize(seg_mask, (224, 224))
                    seg_mask_uint8 = (seg_mask_resized * 255).astype(np.uint8)
                    cv2.imwrite(
                        os.path.join(round_output, "predicted_mask.png"),
                        seg_mask_uint8
                    )

                    # 保存文本结果
                    with open(os.path.join(round_output, "prediction.json"), "w") as f:
                        import json
                        json.dump({
                            "stability_prob": float(cls_prob),
                            "prediction": "unstable" if cls_prob > 0.5 else "stable"
                        }, f, indent=2)

                except Exception as e:
                    print(f"Error processing {sample_dir}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CollapseNet inference")
    parser.add_argument("--input", type=str, required=True, help="Input dataset or image dir")
    parser.add_argument("--output", type=str, default="predictions", help="Output dir")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--sample_idx", type=int, default=0)
    args = parser.parse_args()

    main(args)
