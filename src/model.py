"""
Phase 4: ResNet-18 模型搭建 (ImageNet 预训练 + 迁移学习)
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from config import *


def build_model():
    """
    构建 ResNet-18 模型

    - 加载 ImageNet 预训练权重
    - 替换最后的全连接层为 2 分类
    - 可选：冻结 backbone 前几层
    """
    print(f"  加载 ResNet-18 (pretrained={PRETRAINED})...")

    if PRETRAINED:
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
    else:
        model = resnet18(weights=None)

    # 替换最后的全连接层
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES),
    )

    # 全部参数可训练 — K线图与ImageNet差异大，需要全面finetune
    # 不再冻结任何层

    # 统计可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    model = model.to(DEVICE)
    return model


if __name__ == "__main__":
    print("Phase 4: 测试模型构建...")
    model = build_model()
    print(f"  模型: {model.__class__.__name__}")
    print(f"  设备: {DEVICE}")

    # 测试前向传播
    dummy_input = torch.randn(2, 3, 224, 224).to(DEVICE)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"  输入形状: {dummy_input.shape}")
    print(f"  输出形状: {output.shape}")  # [2, 2]
    print("  模型构建测试通过!")
