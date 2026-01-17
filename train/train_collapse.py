"""
CollapseNet 多任务训练脚本
同时训练稳定性分类和受影响区域分割
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset_collapse import CollapseDataset
from model_collapse import CollapseNet


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        return 1.0 - dice


class MultiTaskLoss(nn.Module):
    """多任务损失函数

    结合分类损失和分割损失，支持任务权重调整
    """

    def __init__(self, cls_weight=1.0, seg_weight=1.0, use_dice=False):
        super().__init__()
        self.cls_weight = cls_weight
        self.seg_weight = seg_weight
        self.use_dice = use_dice

        self.cls_criterion = nn.BCEWithLogitsLoss()
        self.seg_criterion = nn.BCEWithLogitsLoss()
        if use_dice:
            self.dice_criterion = DiceLoss()

    def forward(self, cls_pred, seg_pred, cls_target, seg_target):
        # 分类损失
        cls_loss = self.cls_criterion(cls_pred, cls_target)

        # 分割损失
        seg_loss = self.seg_criterion(seg_pred, seg_target)
        if self.use_dice:
            dice_loss = self.dice_criterion(seg_pred, seg_target)
            seg_loss = seg_loss + dice_loss

        # 加权总损失
        total_loss = self.cls_weight * cls_loss + self.seg_weight * seg_loss

        return total_loss, cls_loss, seg_loss


def train_epoch(
    model, loader, criterion, optimizer, device, use_segmentation=True
):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_seg_loss = 0

    cls_correct = 0
    cls_total = 0

    pbar = tqdm(loader, desc="Training")

    for batch in pbar:
        if use_segmentation:
            inputs, cls_labels, seg_labels = batch
            seg_labels = seg_labels.to(device)
        else:
            inputs, cls_labels = batch

        inputs = inputs.to(device)
        cls_labels = cls_labels.to(device).unsqueeze(1)

        optimizer.zero_grad()

        # 前向传播
        cls_pred, seg_pred = model(inputs)

        # 计算损失
        if use_segmentation:
            loss, cls_loss, seg_loss = criterion(
                cls_pred, seg_pred, cls_labels, seg_labels
            )
        else:
            loss = criterion(cls_pred, cls_labels)
            cls_loss = loss
            seg_loss = torch.tensor(0.0, device=device)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_seg_loss += seg_loss.item() if isinstance(seg_loss, torch.Tensor) else seg_loss

        # 分类准确率
        cls_preds = (torch.sigmoid(cls_pred) > 0.5).float()
        cls_correct += (cls_preds == cls_labels).sum().item()
        cls_total += cls_labels.size(0)

        pbar.set_postfix(
            loss=loss.item(),
            cls_loss=cls_loss.item(),
            seg_loss=seg_loss.item() if isinstance(seg_loss, torch.Tensor) else seg_loss,
            acc=cls_correct / cls_total
        )

    n = len(loader)
    return {
        'loss': total_loss / n,
        'cls_loss': total_cls_loss / n,
        'seg_loss': total_seg_loss / n,
        'acc': cls_correct / cls_total
    }


def validate(
    model, loader, criterion, device, use_segmentation=True, iou_thresh=0.5
):
    """验证模型"""
    model.eval()

    total_loss = 0
    total_cls_loss = 0
    total_seg_loss = 0

    cls_correct = 0
    cls_total = 0

    total_iou = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            if use_segmentation:
                inputs, cls_labels, seg_labels = batch
                seg_labels = seg_labels.to(device)
            else:
                inputs, cls_labels = batch

            inputs = inputs.to(device)
            cls_labels = cls_labels.to(device).unsqueeze(1)

            cls_pred, seg_pred = model(inputs)

            if use_segmentation:
                loss, cls_loss, seg_loss = criterion(
                    cls_pred, seg_pred, cls_labels, seg_labels
                )
            else:
                loss = criterion(cls_pred, cls_labels)
                cls_loss = loss
                seg_loss = torch.tensor(0.0, device=device)

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_seg_loss += seg_loss.item() if isinstance(seg_loss, torch.Tensor) else seg_loss

            # 分类准确率
            cls_preds = (torch.sigmoid(cls_pred) > 0.5).float()
            cls_correct += (cls_preds == cls_labels).sum().item()
            cls_total += cls_labels.size(0)

            # IoU for segmentation
            if use_segmentation:
                seg_probs = torch.sigmoid(seg_pred) > 0.5
                seg_masks = seg_labels > 0.5

                # 计算 IoU
                intersection = (seg_probs & seg_masks).float().sum((1, 2))
                union = (seg_probs | seg_masks).float().sum((1, 2))
                iou = (intersection + 1e-6) / (union + 1e-6)
                total_iou += iou.sum().item()
                total_samples += cls_labels.size(0)

    n = len(loader)
    results = {
        'loss': total_loss / n,
        'cls_loss': total_cls_loss / n,
        'seg_loss': total_seg_loss / n,
        'acc': cls_correct / cls_total
    }

    if use_segmentation and total_samples > 0:
        results['iou'] = total_iou / total_samples

    return results


def main(args):
    """主函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === 1. 创建数据集 ===
    dataset = CollapseDataset(
        data_dir=args.data_dir,
        image_size=args.image_size,
        use_segmentation=args.use_segmentation
    )

    if len(dataset) == 0:
        print("Error: No samples found in dataset!")
        return

    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # === 2. 创建模型 ===
    model = CollapseNet(
        backbone=args.backbone,
        pretrained=True,
        decoder_channels=args.decoder_channels,
        dropout=args.dropout
    ).to(device)

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # === 3. 损失函数和优化器 ===
    criterion = MultiTaskLoss(
        cls_weight=args.cls_weight,
        seg_weight=args.seg_weight,
        use_dice=args.use_dice
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr_min
    )

    # === 4. 保存目录 ===
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)

    best_metric = 0  # 用于保存最佳模型

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    # === 5. 训练循环 ===
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # 训练
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            use_segmentation=args.use_segmentation
        )

        # 验证
        val_metrics = validate(
            model, val_loader, criterion, device,
            use_segmentation=args.use_segmentation
        )

        scheduler.step()

        # 打印指标
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"Cls Loss: {train_metrics['cls_loss']:.4f}, "
              f"Seg Loss: {train_metrics['seg_loss']:.4f}, "
              f"Acc: {train_metrics['acc']:.4f}")

        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Cls Loss: {val_metrics['cls_loss']:.4f}, "
              f"Seg Loss: {val_metrics['seg_loss']:.4f}, "
              f"Acc: {val_metrics['acc']:.4f}",
              end="")

        if 'iou' in val_metrics:
            print(f", IoU: {val_metrics['iou']:.4f}")
        else:
            print()

        # TensorBoard 记录
        writer.add_scalar("Loss/train", train_metrics['loss'], epoch)
        writer.add_scalar("Loss/val", val_metrics['loss'], epoch)
        writer.add_scalar("Cls_Loss/train", train_metrics['cls_loss'], epoch)
        writer.add_scalar("Cls_Loss/val", val_metrics['cls_loss'], epoch)
        writer.add_scalar("Seg_Loss/train", train_metrics['seg_loss'], epoch)
        writer.add_scalar("Seg_Loss/val", val_metrics['seg_loss'], epoch)
        writer.add_scalar("Accuracy/train", train_metrics['acc'], epoch)
        writer.add_scalar("Accuracy/val", val_metrics['acc'], epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        if 'iou' in val_metrics:
            writer.add_scalar("IoU/val", val_metrics['iou'], epoch)

        # 保存最佳模型 (基于验证准确率 + IoU)
        if args.use_segmentation:
            current_metric = val_metrics['acc'] + val_metrics.get('iou', 0)
        else:
            current_metric = val_metrics['acc']

        if current_metric > best_metric:
            best_metric = current_metric
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best.pth"))
            print(f"  -> Saved best model (metric: {best_metric:.4f})")

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final.pth"))
    writer.close()

    print(f"\nTraining complete!")
    print(f"Best metric: {best_metric:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CollapseNet")

    # 数据参数
    parser.add_argument("--data_dir", type=str, default="../dataset", help="Dataset dir")
    parser.add_argument("--output_dir", type=str, default="checkpoints_collapse", help="Output dir")
    parser.add_argument("--image_size", type=int, default=224, help="Input image size")
    parser.add_argument("--use_segmentation", action="store_true", default=True,
                        help="Use segmentation task")

    # 模型参数
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet50", "resnet101"], help="Backbone network")
    parser.add_argument("--decoder_channels", type=int, nargs="+",
                        default=[1024, 512, 256, 64],
                        help="Decoder channel sizes")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_min", type=float, default=1e-6, help="Min learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")

    # 损失权重
    parser.add_argument("--cls_weight", type=float, default=1.0,
                        help="Classification loss weight")
    parser.add_argument("--seg_weight", type=float, default=1.0,
                        help="Segmentation loss weight")
    parser.add_argument("--use_dice", action="store_true", default=False,
                        help="Use Dice loss in addition to BCE")

    args = parser.parse_args()

    main(args)
