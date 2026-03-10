# -*- coding: utf-8 -*-
"""Detailed split candidate analysis - show ALL candidates with validation info"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor
from chanlun.cl_interface import BI

DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
LIMITS = {
    "BTC/USDT_d": 500, "BTC/USDT_60m": 1000, "BTC/USDT_5m": 1000,
    "ETH/USDT_60m": 1000, "ETH/USDT_5m": 1000,
}


def load(symbol, freq):
    limit = LIMITS.get(f"{symbol}_{freq}", 1000)
    cache_key = f"{symbol.replace('/', '_')}_{freq}_{limit}"
    return pd.read_parquet(os.path.join(DATA_DIR, f"{cache_key}.parquet"))


def analyze_split(cd_o, cd_p, parent_type, parent_start, parent_end, p_di_idx, p_ding_idx):
    """Analyze a split by showing all candidates with gap info"""
    fxs = cd_o.fxs
    internal = [fx for fx in fxs if parent_start < fx.k.index < parent_end]

    # Find parent start/end FX
    start_fx = next(fx for fx in fxs if fx.k.index == parent_start)
    end_fx = next(fx for fx in fxs if fx.k.index == parent_end)

    if parent_type == "down":
        # Need: di FX for first split, ding FX for second split
        di_fxs = [fx for fx in internal if fx.type == "di"]
        ding_fxs = [fx for fx in internal if fx.type == "ding"]

        print(f"  Start: ding[{parent_start}] val={start_fx.val:.2f}")
        print(f"  End:   di[{parent_end}] val={end_fx.val:.2f}")

        print(f"\n  DI candidates (first split point):")
        for fx in sorted(di_fxs, key=lambda x: x.k.index):
            gap_from_start = fx.k.index - parent_start
            chosen = "<<<" if fx.k.index == p_di_idx else ""
            # Check: start.high > fx.low for down sub-BI 1
            valid_strict = start_fx.val > fx.val
            print(f"    di[{fx.k.index}] val={fx.val:.2f} gap_from_start={gap_from_start} "
                  f"start.val>fx.val={valid_strict} {chosen}")

        print(f"\n  DING candidates (second split point):")
        for fx in sorted(ding_fxs, key=lambda x: x.k.index):
            gap_to_end = parent_end - fx.k.index
            chosen = "<<<" if fx.k.index == p_ding_idx else ""
            # Check: fx.high > end.low for down sub-BI 3
            valid_strict = fx.val > end_fx.val
            print(f"    ding[{fx.k.index}] val={fx.val:.2f} gap_to_end={gap_to_end} "
                  f"fx.val>end.val={valid_strict} {chosen}")

        # For each di candidate, show valid ding partners
        print(f"\n  Valid pairs (di_fx, ding_fx where ding_idx > di_idx and ding.val > di.val):")
        for di_fx in sorted(di_fxs, key=lambda x: x.k.index):
            for ding_fx in sorted(ding_fxs, key=lambda x: x.k.index):
                if ding_fx.k.index <= di_fx.k.index:
                    continue
                gap_di_ding = ding_fx.k.index - di_fx.k.index
                ding_above_di = ding_fx.val > di_fx.val
                gap_start = di_fx.k.index - parent_start
                gap_end = parent_end - ding_fx.k.index

                is_pyarmor = (di_fx.k.index == p_di_idx and ding_fx.k.index == p_ding_idx)
                marker = " <<< PYARMOR" if is_pyarmor else ""

                print(f"    di[{di_fx.k.index}]→ding[{ding_fx.k.index}] "
                      f"gap_start={gap_start} gap_mid={gap_di_ding} gap_end={gap_end} "
                      f"ding>di={ding_above_di} "
                      f"start.val>di.val={start_fx.val > di_fx.val} "
                      f"ding.val>end.val={ding_fx.val > end_fx.val}"
                      f"{marker}")
    else:
        # up: need ding FX first, then di FX
        ding_fxs = [fx for fx in internal if fx.type == "ding"]
        di_fxs = [fx for fx in internal if fx.type == "di"]

        print(f"  Start: di[{parent_start}] val={start_fx.val:.2f}")
        print(f"  End:   ding[{parent_end}] val={end_fx.val:.2f}")

        print(f"\n  DING candidates (first split point):")
        for fx in sorted(ding_fxs, key=lambda x: x.k.index):
            gap_from_start = fx.k.index - parent_start
            chosen = "<<<" if fx.k.index == p_ding_idx else ""
            valid_strict = fx.val > start_fx.val
            print(f"    ding[{fx.k.index}] val={fx.val:.2f} gap_from_start={gap_from_start} "
                  f"fx.val>start.val={valid_strict} {chosen}")

        print(f"\n  DI candidates (second split point):")
        for fx in sorted(di_fxs, key=lambda x: x.k.index):
            gap_to_end = parent_end - fx.k.index
            chosen = "<<<" if fx.k.index == p_di_idx else ""
            valid_strict = end_fx.val > fx.val
            print(f"    di[{fx.k.index}] val={fx.val:.2f} gap_to_end={gap_to_end} "
                  f"fx.val<end.val={valid_strict} {chosen}")

        print(f"\n  Valid pairs (ding_fx, di_fx where di_idx > ding_idx and ding.val > di.val):")
        for ding_fx in sorted(ding_fxs, key=lambda x: x.k.index):
            for di_fx in sorted(di_fxs, key=lambda x: x.k.index):
                if di_fx.k.index <= ding_fx.k.index:
                    continue
                gap_ding_di = di_fx.k.index - ding_fx.k.index
                ding_above_di = ding_fx.val > di_fx.val
                gap_start = ding_fx.k.index - parent_start
                gap_end = parent_end - di_fx.k.index

                is_pyarmor = (di_fx.k.index == p_di_idx and ding_fx.k.index == p_ding_idx)
                marker = " <<< PYARMOR" if is_pyarmor else ""

                print(f"    ding[{ding_fx.k.index}]→di[{di_fx.k.index}] "
                      f"gap_start={gap_start} gap_mid={gap_ding_di} gap_end={gap_end} "
                      f"ding>di={ding_above_di} "
                      f"ding.val>start.val={ding_fx.val > start_fx.val} "
                      f"end.val>di.val={end_fx.val > di_fx.val}"
                      f"{marker}")


for symbol, freq in [("BTC/USDT", "60m"), ("BTC/USDT", "5m"), ("BTC/USDT", "d"),
                      ("ETH/USDT", "60m"), ("ETH/USDT", "5m")]:
    df = load(symbol, freq)
    cd_o = CL_Open(symbol, freq, config={})
    cd_o.process_klines(df)
    cd_p = CL_Pyarmor(symbol, freq, config={})
    cd_p.process_klines(df)

    p_bis = list(cd_p.bis)
    i = 0
    splits_found = []
    while i < len(p_bis):
        bi = p_bis[i]
        if getattr(bi, 'is_split', False) and i + 2 < len(p_bis):
            parent_type = bi.type
            parent_start = bi.start.k.index
            parent_end = p_bis[i + 2].end.k.index

            if parent_type == "down":
                p_di_idx = bi.end.k.index
                p_ding_idx = p_bis[i + 1].end.k.index
            else:
                p_ding_idx = bi.end.k.index
                p_di_idx = p_bis[i + 1].end.k.index

            splits_found.append((parent_type, parent_start, parent_end, p_di_idx, p_ding_idx))
            i += 3
        else:
            i += 1

    if splits_found:
        print(f"\n{'='*80}")
        print(f"{symbol} {freq}: {len(splits_found)} splits")
        print(f"{'='*80}")
        for parent_type, parent_start, parent_end, p_di_idx, p_ding_idx in splits_found:
            print(f"\n  Parent: {parent_type}[{parent_start}→{parent_end}]")
            analyze_split(cd_o, cd_p, parent_type, parent_start, parent_end, p_di_idx, p_ding_idx)
