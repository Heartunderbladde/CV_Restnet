"""
Phase 3: PyTorch Dataset & DataLoader
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from config import *


# 训练集的数据增强
# K线图特殊性：不翻转(时间不可逆)、不旋转(破坏形态)、只做轻度颜色抖动
train_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.05, contrast=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 验证/测试集只做标准化
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class KLineDataset(Dataset):
    """
    K线图数据集

    目录结构:
        data/images/train/up/xxx.png
        data/images/train/down/xxx.png
        data/images/val/up/xxx.png
        data/images/val/down/xxx.png
        data/images/test/up/xxx.png
        data/images/test/down/xxx.png
    """

    def __init__(self, split="train", transform=None):
        """
        split: "train" | "val" | "test"
        """
        self.split = split
        self.transform = transform if transform else eval_transform

        self.samples = []

        for label_name, label_val in [("up", 1), ("down", 0)]:
            label_dir = os.path.join(IMAGE_DIR, split, label_name)
            if not os.path.exists(label_dir):
                continue
            for fname in os.listdir(label_dir):
                if fname.endswith(".png"):
                    self.samples.append({
                        "path": os.path.join(label_dir, fname),
                        "label": label_val,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["path"]).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(sample["label"], dtype=torch.long)
        return img, label


def get_dataloaders():
    """创建 train/val/test 的 DataLoader"""
    train_dataset = KLineDataset(split="train", transform=train_transform)
    val_dataset = KLineDataset(split="val", transform=eval_transform)
    test_dataset = KLineDataset(split="test", transform=eval_transform)

    print(f"  训练集: {len(train_dataset)} 张")
    print(f"  验证集: {len(val_dataset)} 张")
    print(f"  测试集: {len(test_dataset)} 张")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Phase 3: 测试 DataLoader...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # 测试取一个 batch
    imgs, labels = next(iter(train_loader))
    print(f"  Batch shape: {imgs.shape}")   # [B, 3, 224, 224]
    print(f"  Label shape: {labels.shape}")  # [B]
    print(f"  Labels: {labels[:10]}")
    print("  DataLoader 测试通过!")
