"""
BoxWorld ViT 双流网络训练脚本
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset_vit_twostream import ViTTwoStreamDataset
from model_vit_twostream import ViTTwoStreamStabilityPredictor


def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for global_input, local_input, labels in pbar:
        global_input = global_input.to(device)
        local_input = local_input.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()

        # 混合精度训练
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(global_input, local_input)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(global_input, local_input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=loss.item(), acc=correct/total)

    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for global_input, local_input, labels in tqdm(loader, desc="Validating"):
            global_input = global_input.to(device)
            local_input = local_input.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(global_input, local_input)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


def main(args):
    """主函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建数据集
    train_dataset = ViTTwoStreamDataset(
        args.data_dir,
        image_size=args.image_size,
        padding_ratio=args.padding_ratio,
        augment=True
    )
    val_dataset = ViTTwoStreamDataset(
        args.data_dir,
        image_size=args.image_size,
        padding_ratio=args.padding_ratio,
        augment=False
    )

    # 划分训练集和验证集
    total = len(train_dataset)
    train_size = int(0.8 * total)

    indices = torch.randperm(total).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    print(f"Train: {len(train_subset)}, Val: {len(val_subset)}")

    # DataLoader
    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # 创建模型
    model = ViTTwoStreamStabilityPredictor(
        model_size=args.model_size,
        pretrained=args.pretrained,
        drop_rate=args.drop_rate
    ).to(device)
    print(f"Model: ViT-{args.model_size} Two-Stream")

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")

    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == 'cuda' else None

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    best_acc = 0

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    # 训练循环
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # TensorBoard 记录
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, "best.pth"))
            print(f"Saved best model with acc: {best_acc:.4f}")

    # 保存最终模型
    torch.save(model.state_dict(),
               os.path.join(args.output_dir, "final.pth"))
    writer.close()
    print(f"\nTraining complete! Best acc: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ViT Two-Stream stability predictor"
    )
    parser.add_argument("--data_dir", type=str, default="../dataset")
    parser.add_argument("--output_dir", type=str, default="checkpoints_vit")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--padding_ratio", type=float, default=1.5)
    parser.add_argument("--model_size", type=str, default="small",
                        choices=["tiny", "small", "base"])
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true", help="Use AMP")

    args = parser.parse_args()
    main(args)
