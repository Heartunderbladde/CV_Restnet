"""
Phase 6: 测试集评估 + 结果分析
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report, roc_curve)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *
from model import build_model
from dataset import get_dataloaders, eval_transform, KLineDataset


@torch.no_grad()
def evaluate_test(model, loader):
    """在测试集上全面评估"""
    model.eval()
    all_preds, all_labels = [], []
    all_probs = []

    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    return all_labels, all_preds, all_probs


def plot_confusion_matrix(cm, save_path):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]),
                    ha="center", va="center",
                    fontsize=20, fontweight="bold",
                    color="white" if cm[i][j] > cm.max() / 2 else "black")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["跌 (Down)", "涨 (Up)"], fontsize=12)
    ax.set_yticklabels(["跌 (Down)", "涨 (Up)"], fontsize=12)
    ax.set_xlabel("预测标签", fontsize=12)
    ax.set_ylabel("真实标签", fontsize=12)
    ax.set_title("混淆矩阵 (Confusion Matrix)", fontsize=14)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  混淆矩阵已保存: {save_path}")


def plot_roc_curve(labels, probs, save_path):
    """绘制 ROC 曲线"""
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC 曲线", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC 曲线已保存: {save_path}")


def plot_training_curve(history, save_path):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-", label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], "r-", label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # F1
    axes[2].plot(epochs, history["train_f1"], "b-", label="Train F1")
    axes[2].plot(epochs, history["val_f1"], "r-", label="Val F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score")
    axes[2].set_title("F1 Curve")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  训练曲线已保存: {save_path}")


def main():
    print("=" * 60)
    print("Phase 6: 测试集评估")
    print("=" * 60)

    # 加载数据
    print("\n[1/4] 加载数据...")
    _, _, test_loader = get_dataloaders()

    # 构建模型并加载最佳权重
    print("\n[2/4] 加载最佳模型...")
    model = build_model()
    best_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  已加载 Epoch {checkpoint['epoch']}, Val F1={checkpoint['val_f1']:.4f}")

    # 评估
    print("\n[3/4] 在测试集上评估...")
    labels, preds, probs = evaluate_test(model, test_loader)

    # 计算指标
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, probs)
    cm = confusion_matrix(labels, preds)

    print("\n" + "=" * 50)
    print("  测试集结果")
    print("=" * 50)
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Precision:   {prec:.4f}")
    print(f"  Recall:      {rec:.4f}")
    print(f"  F1 Score:    {f1:.4f}")
    print(f"  AUC-ROC:     {auc:.4f}")
    print(f"  混淆矩阵:")
    print(f"    TN={cm[0][0]:5d}  FP={cm[0][1]:5d}")
    print(f"    FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")
    print(f"\n  分类报告:")
    print(classification_report(labels, preds,
                                target_names=["跌 (Down)", "涨 (Up)"],
                                zero_division=0))

    # 保存文本结果
    results_txt = f"""基于 ResNet 的 K 线图涨跌趋势分类研究 — 实验结果
{'='*60}

数据集:
  - 股票池: 沪深300成分股
  - 时间范围: 2020-2025
  - 训练集: 2020-2022, 验证集: 2023, 测试集: 2024-2025
  - 每张图: 60根日K线 + 成交量
  - 标签: 未来10日涨跌 (二分类)

模型: ResNet-18 (ImageNet 预训练)

测试集结果:
  Accuracy:    {acc:.4f}
  Precision:   {prec:.4f}
  Recall:      {rec:.4f}
  F1 Score:    {f1:.4f}
  AUC-ROC:     {auc:.4f}

混淆矩阵:
             预测跌    预测涨
  真实跌    {cm[0][0]:5d}    {cm[0][1]:5d}
  真实涨    {cm[1][0]:5d}    {cm[1][1]:5d}

最佳模型信息:
  Epoch: {checkpoint['epoch']}
  Val F1: {checkpoint['val_f1']:.4f}
  Val Acc: {checkpoint['val_acc']:.4f}
  Val AUC: {checkpoint['val_auc']:.4f}
"""

    results_path = os.path.join(OUTPUT_DIR, "results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(results_txt)
    print(f"  文本结果已保存: {results_path}")

    # 保存数值结果
    metrics = {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "auc": auc, "confusion_matrix": cm.tolist(),
        "labels": labels.tolist(), "preds": preds.tolist(), "probs": probs.tolist(),
    }
    with open(os.path.join(OUTPUT_DIR, "test_metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)

    # 绘制图表
    print("\n[4/4] 绘制结果图表...")
    plot_confusion_matrix(cm, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plot_roc_curve(labels, probs, os.path.join(OUTPUT_DIR, "roc_curve.png"))

    # 加载训练历史并绘图
    history_path = os.path.join(OUTPUT_DIR, "training_history.pkl")
    if os.path.exists(history_path):
        with open(history_path, "rb") as f:
            history = pickle.load(f)
        plot_training_curve(history, os.path.join(OUTPUT_DIR, "training_curves.png"))

    print("\n" + "=" * 60)
    print("  实验完成!")
    print(f"  所有输出保存在: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
