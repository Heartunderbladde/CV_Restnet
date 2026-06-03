"""
全局配置文件
基于 ResNet 的 K 线图涨跌趋势分类研究

适配环境: ModelScope (魔搭社区) GPU
  - CUDA 12.8, PyTorch 2.10.0, Python 3.12
  - 24GB 显存, 8核32GB
"""

import os

# ============ 项目路径 ============
# 魔搭环境下 BASE_DIR 设为 /mnt/workspace
BASE_DIR = os.environ.get("MODELSCOPE_BASE_DIR",
             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

for d in [RAW_DATA_DIR, IMAGE_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ============ 数据参数 ============
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
CANDLE_COUNT = 60          # 每张图 60 根日K线
FORECAST_DAYS = 10         # 预测未来10日涨跌
UP_THRESHOLD = 0.03        # 涨≥3% 才标"涨"，过滤震荡
DOWN_THRESHOLD = -0.03     # 跌≤-3% 才标"跌"，过滤震荡

# ============ 数据集划分（按年份） ============
TRAIN_START = "2020-01-01"
TRAIN_END = "2022-12-31"
VAL_START = "2023-01-01"
VAL_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2025-12-31"

# ============ 图片参数 ============
IMG_SIZE = 224              # ResNet 标准输入尺寸
CHART_COLOR_UP = "red"     # 红涨
CHART_COLOR_DOWN = "green"  # 绿跌

# ============ 训练参数 (GPU 24GB) ============
BATCH_SIZE = 128
NUM_EPOCHS = 20             # 配合早停，不需要太多轮
LEARNING_RATE = 1e-5        # 全量 finetune + 标签平滑，用极低 LR
WEIGHT_DECAY = 5e-4         # 适度正则化
LABEL_SMOOTHING = 0.1       # 标签平滑防止过拟合
NUM_WORKERS = 4

# ============ 模型参数 ============
MODEL_NAME = "resnet18"
PRETRAINED = True
NUM_CLASSES = 2

# ============ 设备 ============
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
