# -*- coding: utf-8 -*-
"""
对比测试：cl_open vs cl_pyarmor

对同一组行情数据，分别用两个引擎计算，逐项对比输出。
使用方法：
    cd /path/to/chanlun-pro
    .venv/bin/python tests/test_cl_open_vs_pyarmor.py

需要网络访问 Binance API 获取行情数据（或使用缓存数据）。
"""
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd

# 确保 src 在 path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import ccxt

from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor
from chanlun.cl_interface import Config


# ---- 数据获取 ----

CACHE_DIR = os.path.join(os.path.dirname(__file__), "test_data")
os.makedirs(CACHE_DIR, exist_ok=True)

FREQ_MAP = {
    "w": "1w",
    "d": "1d",
    "4h": "4h",
    "60m": "1h",
    "30m": "30m",
    "15m": "15m",
    "10m": "5m",
    "5m": "5m",
    "1m": "1m",
}


def fetch_klines(
    symbol: str, frequency: str, limit: int = 1000, use_cache: bool = True
) -> pd.DataFrame:
    """获取K线数据，支持本地缓存"""
    cache_key = f"{symbol.replace('/', '_')}_{frequency}_{limit}"
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.parquet")

    if use_cache and os.path.exists(cache_file):
        df = pd.read_parquet(cache_file)
        print(f"  [缓存] {cache_file} ({len(df)} 行)")
        return df

    ccxt_freq = FREQ_MAP.get(frequency, frequency)
    exchange = ccxt.binance({"options": {"defaultType": "future"}})

    all_ohlcv = []
    end_time = None
    remaining = limit

    while remaining > 0:
        batch_size = min(remaining, 1000)
        params = {}
        if end_time is not None:
            params["endTime"] = end_time

        ohlcv = exchange.fetch_ohlcv(
            symbol, ccxt_freq, limit=batch_size, params=params
        )
        if len(ohlcv) == 0:
            break

        all_ohlcv = ohlcv + all_ohlcv
        end_time = ohlcv[0][0] - 1
        remaining -= len(ohlcv)

        if len(ohlcv) < batch_size:
            break

    df = pd.DataFrame(
        all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["code"] = symbol

    # 缓存
    df.to_parquet(cache_file, index=False)
    print(f"  [下载] {symbol} {frequency} → {len(df)} 行, 缓存到 {cache_file}")

    return df


# ---- 对比逻辑 ----


def compare_counts(label: str, open_val, pyarmor_val) -> dict:
    match = open_val == pyarmor_val
    status = "✅" if match else "❌"
    print(f"  {label:12s}: open={open_val:<6} pyarmor={pyarmor_val:<6} {status}")
    return {"label": label, "open": open_val, "pyarmor": pyarmor_val, "match": match}


def compare_line_details(label: str, open_lines, pyarmor_lines, max_show=5):
    """对比两个引擎的线（笔/段）的详细差异"""
    diffs = []
    min_len = min(len(open_lines), len(pyarmor_lines))

    for i in range(min_len):
        o_line = open_lines[i]
        p_line = pyarmor_lines[i]

        issues = []
        if o_line.type != p_line.type:
            issues.append(f"type: {o_line.type} vs {p_line.type}")
        if abs(o_line.high - p_line.high) > 1e-6:
            issues.append(f"high: {o_line.high} vs {p_line.high}")
        if abs(o_line.low - p_line.low) > 1e-6:
            issues.append(f"low: {o_line.low} vs {p_line.low}")

        if issues:
            diffs.append((i, issues))

    if len(open_lines) != len(pyarmor_lines):
        diffs.append(
            ("count", [f"数量不同: {len(open_lines)} vs {len(pyarmor_lines)}"])
        )

    if diffs:
        shown = 0
        for idx, issues in diffs:
            if shown >= max_show:
                print(f"  ... 还有 {len(diffs) - shown} 个差异")
                break
            print(f"  [{label}#{idx}] {'; '.join(issues)}")
            shown += 1
    return diffs


def compare_mmds(label: str, open_lines, pyarmor_lines, zs_type="|", max_show=5):
    """对比买卖点"""
    diffs = []
    min_len = min(len(open_lines), len(pyarmor_lines))

    for i in range(min_len):
        o_mmds = sorted(open_lines[i].line_mmds(zs_type))
        p_mmds = sorted(pyarmor_lines[i].line_mmds(zs_type))
        if o_mmds != p_mmds:
            diffs.append((i, o_mmds, p_mmds))

    if diffs:
        shown = 0
        for idx, o_mmds, p_mmds in diffs:
            if shown >= max_show:
                print(f"  ... 还有 {len(diffs) - shown} 个差异")
                break
            print(f"  [{label}#{idx}] open={o_mmds} pyarmor={p_mmds}")
            shown += 1
    else:
        print(f"  {label} 买卖点: 全部一致 ✅")
    return diffs


def compare_bcs(label: str, open_lines, pyarmor_lines, zs_type="|", max_show=5):
    """对比背驰"""
    diffs = []
    min_len = min(len(open_lines), len(pyarmor_lines))

    for i in range(min_len):
        o_bcs = sorted(open_lines[i].line_bcs(zs_type))
        p_bcs = sorted(pyarmor_lines[i].line_bcs(zs_type))
        if o_bcs != p_bcs:
            diffs.append((i, o_bcs, p_bcs))

    if diffs:
        shown = 0
        for idx, o_bcs, p_bcs in diffs:
            if shown >= max_show:
                print(f"  ... 还有 {len(diffs) - shown} 个差异")
                break
            print(f"  [{label}#{idx}] open={o_bcs} pyarmor={p_bcs}")
            shown += 1
    else:
        print(f"  {label} 背驰: 全部一致 ✅")
    return diffs


def run_comparison(
    symbol: str,
    frequency: str,
    config: dict = None,
    limit: int = 1000,
    label: str = "",
):
    """运行一次对比测试"""
    if config is None:
        config = {}

    if not label:
        label = f"{symbol} {frequency}"

    print(f"\n{'='*60}")
    print(f"  {label} (config: {config or '默认'})")
    print(f"{'='*60}")

    # 获取数据
    df = fetch_klines(symbol, frequency, limit=limit)

    # 分别用两个引擎计算
    cd_open = CL_Open(symbol, frequency, config=config)
    cd_open.process_klines(df)

    cd_pyarmor = CL_Pyarmor(symbol, frequency, config=config)
    cd_pyarmor.process_klines(df)

    results = {}

    # 1. 缠论K线
    results["cl_klines"] = compare_counts(
        "缠论K线", len(cd_open.get_cl_klines()), len(cd_pyarmor.get_cl_klines())
    )

    # 2. 分型
    results["fxs"] = compare_counts(
        "分型", len(cd_open.get_fxs()), len(cd_pyarmor.get_fxs())
    )

    # 3. 笔
    results["bis"] = compare_counts(
        "笔", len(cd_open.get_bis()), len(cd_pyarmor.get_bis())
    )

    # 4. 线段
    results["xds"] = compare_counts(
        "线段", len(cd_open.get_xds()), len(cd_pyarmor.get_xds())
    )

    # 5. 走势段
    results["zsds"] = compare_counts(
        "走势段", len(cd_open.get_zsds()), len(cd_pyarmor.get_zsds())
    )

    # 6. 笔中枢
    results["bi_zss"] = compare_counts(
        "笔中枢", len(cd_open.get_bi_zss()), len(cd_pyarmor.get_bi_zss())
    )

    # 7. 线段中枢
    results["xd_zss"] = compare_counts(
        "线段中枢", len(cd_open.get_xd_zss()), len(cd_pyarmor.get_xd_zss())
    )

    # 详细对比
    print(f"\n--- 笔详细对比 ---")
    bi_diffs = compare_line_details("笔", cd_open.get_bis(), cd_pyarmor.get_bis())

    print(f"\n--- 线段详细对比 ---")
    xd_diffs = compare_line_details("线段", cd_open.get_xds(), cd_pyarmor.get_xds())

    # 买卖点 / 背驰对比
    if len(cd_open.get_bis()) > 0 and len(cd_pyarmor.get_bis()) > 0:
        print(f"\n--- 笔买卖点对比 ---")
        compare_mmds("笔MMD", cd_open.get_bis(), cd_pyarmor.get_bis())
        print(f"\n--- 笔背驰对比 ---")
        compare_bcs("笔BC", cd_open.get_bis(), cd_pyarmor.get_bis())

    if len(cd_open.get_xds()) > 0 and len(cd_pyarmor.get_xds()) > 0:
        print(f"\n--- 线段买卖点对比 ---")
        compare_mmds("线段MMD", cd_open.get_xds(), cd_pyarmor.get_xds())
        print(f"\n--- 线段背驰对比 ---")
        compare_bcs("线段BC", cd_open.get_xds(), cd_pyarmor.get_xds())

    return results


# ---- 主测试入口 ----


def run_default_tests():
    """运行默认配置下的全部测试"""
    test_cases = [
        ("BTC/USDT", "60m", 1000),
        ("BTC/USDT", "5m", 1000),
        ("BTC/USDT", "d", 500),
        ("ETH/USDT", "60m", 1000),
        ("ETH/USDT", "5m", 1000),
    ]

    all_results = {}
    for symbol, freq, limit in test_cases:
        label = f"{symbol} {freq}"
        try:
            results = run_comparison(symbol, freq, limit=limit, label=label)
            all_results[label] = results
        except Exception as e:
            print(f"\n❌ {label} 测试失败: {e}")
            import traceback
            traceback.print_exc()

    # 汇总
    print(f"\n\n{'='*60}")
    print("  汇总")
    print(f"{'='*60}")
    for label, results in all_results.items():
        matches = sum(1 for r in results.values() if r.get("match"))
        total = len(results)
        print(f"  {label}: {matches}/{total} 项一致")


def run_single_test():
    """运行单个快速测试"""
    run_comparison("BTC/USDT", "60m", limit=500)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        run_default_tests()
    else:
        run_single_test()
