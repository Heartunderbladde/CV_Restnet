"""
Phase 5: 训练 + 验证流程
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
from config import *
from model import build_model
from dataset import get_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, epoch):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch:2d} [Train]")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)

    return avg_loss, acc, f1


@torch.no_grad()
def validate(model, loader, criterion, split_name="Val"):
    """验证/测试"""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    all_probs = []

    pbar = tqdm(loader, desc=f"         [{split_name}]")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, acc, f1, auc, cm


def main():
    print("=" * 60)
    print("Phase 5: 训练 + 验证")
    print("=" * 60)

    # 准备数据
    print("\n[1/4] 加载数据...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # 构建模型
    print("\n[2/4] 构建模型...")
    model = build_model()

    # 损失函数（处理类别不平衡）
    # 统计训练集类别权重
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.numpy())
    n_up = sum(train_labels)
    n_down = len(train_labels) - n_up
    print(f"  训练集: 涨={n_up}, 跌={n_down}, 涨/跌={n_up/n_down:.2f}" if n_down > 0 else "")

    # 对少数类给更大权重
    if n_down > 0 and n_up > 0:
        pos_weight = n_down / n_up
        class_weights = torch.tensor([1.0, pos_weight], device=DEVICE, dtype=torch.float32)
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 优化器和调度器
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # 训练循环
    print(f"\n[3/4] 开始训练 (设备: {DEVICE})...")
    print(f"  Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print("-" * 60)

    best_val_f1 = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    history = {"train_loss": [], "train_acc": [], "train_f1": [],
               "val_loss": [], "val_acc": [], "val_f1": [], "val_auc": []}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch
        )
        val_loss, val_acc, val_f1, val_auc, val_cm = validate(
            model, val_loader, criterion, "Val"
        )

        scheduler.step()

        # 记录
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_auc"].append(val_auc)

        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        print(f"  Val   CM: TN={val_cm[0][0]}, FP={val_cm[0][1]}, FN={val_cm[1][0]}, TP={val_cm[1][1]}")

        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "val_auc": val_auc,
            }, best_model_path)
            print(f"  *** 最佳模型已保存 (F1={best_val_f1:.4f}) ***")
        print()

    # 保存训练历史
    print("[4/4] 保存训练记录...")
    with open(os.path.join(OUTPUT_DIR, "training_history.pkl"), "wb") as f:
        pickle.dump(history, f)

    print(f"\n  训练完成! 最佳 Val F1: {best_val_f1:.4f}")
    print(f"  最佳模型: {best_model_path}")


if __name__ == "__main__":
    main()
