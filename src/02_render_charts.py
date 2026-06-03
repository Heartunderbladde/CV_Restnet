"""
Phase 2: 将 OHLC 数据渲染为 K 线图图片
- 滑动窗口步长 STRIDE=15 天（GPU 训练，数据量可放开）
- 多进程并行渲染（Linux fork 模式）
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from config import *

# 滑动步长：每隔 STRIDE 天取一个窗口
STRIDE = 45  # 降低窗口重叠(25%)，避免数据冗余导致过拟合

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def render_single_chart_from_row(args):
    """多进程 worker: 渲染一张 K 线图 (Linux fork 模式无需额外初始化)"""
    ohlc_data, save_path = args

    try:
        df = pd.DataFrame(ohlc_data)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index("trade_date")
        df = df[["open", "high", "low", "close", "vol"]]

        # mplfinance 需要首字母大写列名
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "vol": "Volume",
        })

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna()

        if len(df) < CANDLE_COUNT // 2:
            return False

        mc = mpf.make_marketcolors(
            up=CHART_COLOR_UP, down=CHART_COLOR_DOWN,
            edge="inherit", wick="inherit",
            volume={"up": CHART_COLOR_UP, "down": CHART_COLOR_DOWN},
        )
        style = mpf.make_mpf_style(
            marketcolors=mc, gridstyle=":", gridcolor="gray",
            facecolor="white", figcolor="white",
        )

        fig, axes = mpf.plot(
            df, type="candle", volume=True, style=style,
            figsize=(2.24, 2.24), tight_layout=True,
            returnfig=True, panel_ratios=(3, 1),
        )

        for ax in axes:
            ax.set_axis_off()

        # 直接保存到目标路径
        tmp = save_path + ".tmp.png"
        fig.savefig(tmp, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        img = Image.open(tmp).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        img.save(save_path, "PNG")

        if os.path.exists(tmp):
            os.remove(tmp)
        return True

    except Exception:
        return False


def prepare_render_tasks(all_data):
    """
    遍历所有股票，生成渲染任务列表
    返回: [(ohlc_dict, save_path), ...]
    """
    tasks = []

    for ts_code, df in all_data.items():
        df = df.reset_index(drop=True)
        n = len(df)

        # 滑动窗口，步长 STRIDE
        for i in range(0, n - CANDLE_COUNT - FORECAST_DAYS + 1, STRIDE):
            window_end = i + CANDLE_COUNT
            label_date_idx = window_end + FORECAST_DAYS - 1

            if label_date_idx >= n:
                break

            row = df.iloc[label_date_idx]
            label = row.get("label")
            if pd.isna(label):
                continue

            label = int(label)
            label_date = df.iloc[window_end - 1]["trade_date"]

            # 时间划分
            ld = pd.Timestamp(label_date)
            if ld <= pd.Timestamp(TRAIN_END):
                split = "train"
            elif ld <= pd.Timestamp(VAL_END):
                split = "val"
            else:
                split = "test"

            label_dir = "up" if label == 1 else "down"
            safe_code = ts_code.replace(".", "_")
            img_name = f"{safe_code}_{i:06d}.png"
            save_path = os.path.join(IMAGE_DIR, split, label_dir, img_name)

            ohlc_slice = df.iloc[i:window_end][
                ["trade_date", "open", "high", "low", "close", "vol"]
            ]
            # 转为 Python 原生类型确保可 pickle
            records = []
            for _, row in ohlc_slice.iterrows():
                records.append({
                    "trade_date": str(row["trade_date"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "vol": float(row["vol"]),
                })

            tasks.append((records, save_path))

    return tasks


def main():
    print("=" * 60)
    print("Phase 2: 渲染 K 线图图片")
    print(f"  步长 STRIDE = {STRIDE} 天")
    print("=" * 60)

    # 加载数据
    data_path = os.path.join(RAW_DATA_DIR, "all_ohlc_data.pkl")
    if not os.path.exists(data_path):
        print(f"ERROR: 数据文件不存在 {data_path}")
        return

    with open(data_path, "rb") as f:
        all_data = pickle.load(f)

    print(f"  加载数据: {len(all_data)} 只股票")

    # 创建目录
    for split in ["train", "val", "test"]:
        for label_dir in ["up", "down"]:
            os.makedirs(os.path.join(IMAGE_DIR, split, label_dir), exist_ok=True)

    # 准备任务
    print("  准备渲染任务...")
    tasks = prepare_render_tasks(all_data)
    print(f"  总任务数: {len(tasks)}")

    # 统计各 split 分布
    splits = {"train": 0, "val": 0, "test": 0}
    labels = {"up": 0, "down": 0}
    for _, path in tasks:
        for s in splits:
            if f"/{s}/" in path.replace("\\", "/"):
                splits[s] += 1
                break
        for l in labels:
            if f"/{l}/" in path.replace("\\", "/"):
                labels[l] += 1
                break

    print(f"  Train: {splits['train']}, Val: {splits['val']}, Test: {splits['test']}")
    print(f"  Up: {labels['up']}, Down: {labels['down']}")

    # 多进程渲染
    n_workers = max(1, cpu_count() - 2)
    print(f"\n  开始渲染 ({n_workers} 个进程)...")

    with Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(render_single_chart_from_row, tasks, chunksize=20),
            total=len(tasks),
            desc="渲染进度",
        ))

    n_ok = sum(1 for r in results if r)
    n_fail = len(results) - n_ok
    print(f"\n  完成! 成功: {n_ok}, 失败: {n_fail}")

    # 保存统计
    stats = {"splits": splits, "labels": labels, "n_ok": n_ok, "n_fail": n_fail,
             "stride": STRIDE}
    with open(os.path.join(IMAGE_DIR, "render_stats.pkl"), "wb") as f:
        pickle.dump(stats, f)

    print(f"  图片已保存到 {IMAGE_DIR}")


if __name__ == "__main__":
    main()
