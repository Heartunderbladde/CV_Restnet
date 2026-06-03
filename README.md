# 基于 ResNet 的 K 线图涨跌趋势分类研究

计算机视觉课程论文实验代码。使用 ResNet-18 对沪深 300 成分股 K 线图进行涨跌趋势二分类。

## 实验设计

| 项目 | 说明 |
|------|------|
| 数据 | 沪深300成分股, 2020-2025, AKShare(腾讯源) |
| 输入 | 60日K线 + 成交量, 224×224, 红涨绿跌 |
| 标签 | 未来10日涨跌, 二分类 |
| 模型 | ResNet-18, ImageNet预训练 |
| 划分 | 2020-22训练 / 2023验证 / 2024-25测试 |

## 魔搭社区运行

1. 上传 `all_ohlc_data.pkl` → `/mnt/workspace/data/raw/`
2. 上传 `src/` 全部 `.py` 文件 → `/mnt/workspace/src/`
3. 上传 `notebooks/experiment.ipynb` → `/mnt/workspace/`
4. 按顺序运行 Notebook 每个 Cell

## 本地运行

```bash
pip install mplfinance scikit-learn tqdm akshare
cd src
python 01_fetch_data.py    # 获取数据
python 02_render_charts.py # 渲染K线图
python train.py            # 训练
python evaluate.py         # 评估
```

## 文件结构

```
├── src/
│   ├── config.py          # 全局配置
│   ├── 01_fetch_data.py   # 数据获取
│   ├── 02_render_charts.py # K线图渲染
│   ├── dataset.py         # PyTorch Dataset
│   ├── model.py           # ResNet-18 模型
│   ├── train.py           # 训练脚本
│   └── evaluate.py        # 评估脚本
├── notebooks/
│   └── experiment.ipynb   # 魔搭一键运行 Notebook
├── data/                  # 数据目录 (gitignore)
└── outputs/               # 结果输出 (gitignore)
```
