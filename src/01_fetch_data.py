"""
Phase 1: 获取沪深300成分股 OHLC 数据 + 标签 (AKShare - 腾讯数据源)
"""

import os
import pickle
import time
import numpy as np
import pandas as pd

# 屏蔽系统代理 + 中国网络兼容
import requests.sessions as _rs
_orig_req = _rs.Session.request
def _patched(self, method, url, *args, **kwargs):
    self.trust_env = False
    kwargs.setdefault("timeout", 15)
    return _orig_req(self, method, url, *args, **kwargs)
_rs.Session.request = _patched

import akshare as ak
from tqdm import tqdm
from config import *


def get_csi300_stocks():
    """获取沪深300成分股列表"""
    print("[1/5] 获取沪深300成分股列表...")

    # 直接用 AKShare 的 stock_zh_index_spot_em 获取沪深300实时行情来提取成分股
    try:
        df = ak.index_stock_cons_weight_csindex("000300")
        # 按位置: 列4 = 成分券代码
        stock_col_idx = 4
        stocks = df.iloc[:, stock_col_idx].unique().tolist()
        print(f"  沪深300 成分股数量: {len(stocks)}")
        return stocks
    except Exception as e:
        print(f"  接口1失败: {e}")

    # 备用: 东方财富
    try:
        df = ak.index_stock_cons("000300")
        stocks = df["品种代码"].unique().tolist()
        print(f"  沪深300 成分股数量(备用): {len(stocks)}")
        return stocks
    except Exception as e2:
        print(f"  备用也失败: {e2}")
        raise RuntimeError("无法获取沪深300成分股")


def to_tx_symbol(code):
    """将 000001 / 000001.SZ 转为 Tencent 格式 sz000001 或 sh600001"""
    code = code.replace(".SZ", "").replace(".SH", "").replace(".BJ", "")
    if code.startswith(("0", "3")):
        return f"sz{code}"
    elif code.startswith(("6", "9")):
        return f"sh{code}"
    return code


def fetch_stock_data(tx_symbol, start, end):
    """
    用腾讯数据源获取单只股票日线数据
    tx_symbol: 如 "sz000001"
    腾讯数据源优势: 在中国大陆网络下比东方财富更稳定
    """
    try:
        df = ak.stock_zh_a_hist_tx(
            symbol=tx_symbol,
            start_date=start,
            end_date=end,
            adjust="qfq",
        )
        if df is None or len(df) == 0:
            return None

        # 腾讯源列: date, open, close, high, low, amount(实为成交量/手)
        df = df.rename(columns={
            "date": "trade_date",
            "open": "open",
            "close": "close",
            "high": "high",
            "low": "low",
            "amount": "vol",  # 腾讯源中 amount 实际是成交量(手)
        })
        df = df[["trade_date", "open", "high", "low", "close", "vol"]]
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values("trade_date").reset_index(drop=True)

        # 成交量 × 100 转为股数(每手100股)
        df["vol"] = pd.to_numeric(df["vol"], errors="coerce") * 100
        return df

    except Exception as e:
        return None


def compute_label(df, forecast_days=FORECAST_DAYS):
    """
    标签: 未来 forecast_days 日的累计收益率
    >0 → label=1 (涨), <=0 → label=0 (跌)
    """
    df = df.copy()
    df["future_close"] = df["close"].shift(-forecast_days)
    df["return"] = (df["future_close"] - df["close"]) / df["close"]
    df["label"] = (df["return"] > THRESHOLD).astype(int)
    df["label"] = df["label"].where(df["future_close"].notna(), np.nan)
    return df


def get_list_date(tx_symbol):
    """获取股票上市日期"""
    try:
        code = tx_symbol[2:]  # sz000001 → 000001
        info = ak.stock_individual_info_em(symbol=code)
        row = info[info["item"] == "上市时间"]
        if len(row) > 0:
            date_str = str(row.iloc[0]["value"])
            return date_str[:10]
    except Exception:
        pass
    return "19000101"


def main():
    print("=" * 60)
    print("Phase 1: 获取数据 + 标签生成 (AKShare - 腾讯源)")
    print("=" * 60)

    # 获取股票列表
    stock_codes = get_csi300_stocks()

    # 标准化并转腾讯格式
    tx_symbols = []
    for code in stock_codes:
        tx = to_tx_symbol(code)
        if len(tx) == 8:  # sz000001 / sh600001
            tx_symbols.append(tx)

    # 去重
    tx_symbols = list(set(tx_symbols))
    print(f"  腾讯格式代码: {len(tx_symbols)} 个")

    # 过滤 2020 年后上市的
    print(f"\n[2/5] 过滤股票（排除2020年后上市）...")
    valid_stocks = []
    for tx in tqdm(tx_symbols, desc="过滤上市时间"):
        list_date = get_list_date(tx)
        if list_date < "20200101":
            valid_stocks.append(tx)
        time.sleep(0.15)

    print(f"  排除后: {len(tx_symbols)} → {len(valid_stocks)}")

    # 获取日线数据
    print(f"\n[3/5] 获取日线数据 ({START_DATE[:4]}-{END_DATE[:4]})...")
    all_data = {}
    failed = []

    for tx in tqdm(valid_stocks, desc="拉取数据"):
        df = fetch_stock_data(
            tx,
            START_DATE.replace("-", ""),
            END_DATE.replace("-", ""),
        )
        if df is not None and len(df) >= CANDLE_COUNT + FORECAST_DAYS + 10:
            all_data[tx] = df
        else:
            failed.append(tx)

        time.sleep(0.12)

    print(f"  成功: {len(all_data)} 只, 失败/数据不足: {len(failed)} 只")

    # 计算标签
    print(f"\n[4/5] 计算标签（未来{FORECAST_DAYS}日涨跌, 二分类）...")
    for code in tqdm(all_data, desc="标签计算"):
        all_data[code] = compute_label(all_data[code])

    # 统计
    all_labels = []
    for df in all_data.values():
        all_labels.extend(df["label"].dropna().tolist())
    n_up = sum(all_labels)
    n_down = len(all_labels) - n_up
    print(f"  总样本数: {len(all_labels)}")
    print(f"  涨(1): {n_up} ({n_up / len(all_labels) * 100:.1f}%)")
    print(f"  跌(0): {n_down} ({n_down / len(all_labels) * 100:.1f}%)")

    # 保存
    print(f"\n[5/5] 保存到 {RAW_DATA_DIR}...")
    with open(os.path.join(RAW_DATA_DIR, "all_ohlc_data.pkl"), "wb") as f:
        pickle.dump(all_data, f)

    meta = {
        "valid_stocks": valid_stocks,
        "failed": failed,
        "n_stocks": len(all_data),
        "n_samples": len(all_labels),
        "label_dist": {"up": n_up, "down": n_down},
        "candle_count": CANDLE_COUNT,
        "forecast_days": FORECAST_DAYS,
        "date_range": f"{START_DATE} ~ {END_DATE}",
        "data_source": "AKShare (Tencent)",
    }
    with open(os.path.join(RAW_DATA_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"\n  完成! 股票数: {len(all_data)}, 样本数: {len(all_labels)}")
    if n_down > 0:
        print(f"  涨跌比: {n_up / n_down:.2f}")


if __name__ == "__main__":
    main()
