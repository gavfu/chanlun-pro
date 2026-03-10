# -*- coding: utf-8 -*-
"""
Diagnose remaining v3 diffs for ETH5m & BTC5m
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor
from chanlun.cl_interface import FX, BI, Config

DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")

def load(symbol, freq, limit):
    cache_key = f"{symbol.replace('/', '_')}_{freq}_{limit}"
    return pd.read_parquet(os.path.join(DATA_DIR, f"{cache_key}.parquet"))


def diagnose_diff(symbol, freq, limit, diff_start_idx, diff_pyarmor_end, diff_v3_end):
    """Diagnose why v3 and pyarmor disagree about an endpoint"""
    df = load(symbol, freq, limit)
    cd = CL_Open(symbol, freq, config={})
    cd.process_klines(df)
    fxs = cd.get_fxs()

    qj, qy = cd.fx_qj, cd.fx_qy

    # Find the start FX
    start_fx = next(fx for fx in fxs if fx.k.index == diff_start_idx)
    start_pos = next(i for i, fx in enumerate(fxs) if fx.k.index == diff_start_idx)

    print(f"\n{'='*70}")
    print(f"  {symbol} {freq}: BI starting at FX({diff_start_idx},{start_fx.type})")
    print(f"  v3 ends at {diff_v3_end}, pyarmor ends at {diff_pyarmor_end}")
    print(f"{'='*70}")

    # Print all FXs from start to max(v3_end, pyarmor_end) + 5
    max_end = max(diff_v3_end, diff_pyarmor_end) + 10
    print(f"\n  FXs from {diff_start_idx} to {max_end}:")
    for i, fx in enumerate(fxs):
        if fx.k.index < diff_start_idx or fx.k.index > max_end:
            continue
        marker = ""
        if fx.k.index == diff_start_idx:
            marker = " ← START"
        elif fx.k.index == diff_v3_end:
            marker = " ← V3_END"
        elif fx.k.index == diff_pyarmor_end:
            marker = " ← PYARMOR_END"
        print(f"    fxs[{i:>3}]: FX({fx.k.index:>3},{fx.type:>4}) val={fx.val:.2f} "
              f"k_idx={fx.k.k_index:>3} "
              f"h={fx.high(qj,qy):.2f} l={fx.low(qj,qy):.2f}{marker}")

    # Simulate v3 logic step by step
    print(f"\n  Simulating v3 logic from FX({diff_start_idx}):")
    end_fx = None
    end_idx = -1

    for i in range(start_pos + 1, min(start_pos + 30, len(fxs))):
        cur_fx = fxs[i]
        if cur_fx.k.index > max_end:
            break

        if end_fx is None:
            if cur_fx.type != start_fx.type:
                valid = cd._bi_fx_valid(start_fx, cur_fx)
                print(f"    [{i}] FX({cur_fx.k.index},{cur_fx.type}): opposite, _bi_fx_valid={valid}")
                if valid:
                    end_fx = cur_fx
                    end_idx = i
                    print(f"         → SET end_fx")
            else:
                print(f"    [{i}] FX({cur_fx.k.index},{cur_fx.type}): same type, skip (v3 no replace)")
        else:
            if cur_fx.type == end_fx.type:
                if (end_fx.type == "di" and cur_fx.val <= end_fx.val) or \
                   (end_fx.type == "ding" and cur_fx.val >= end_fx.val):
                    ext_valid = cd._bi_fx_valid(start_fx, cur_fx)
                    print(f"    [{i}] FX({cur_fx.k.index},{cur_fx.type}): extend candidate "
                          f"(val={cur_fx.val:.2f} vs {end_fx.val:.2f}), valid={ext_valid}")
                    if ext_valid:
                        end_fx = cur_fx
                        end_idx = i
                        print(f"         → EXTEND end_fx")
                else:
                    print(f"    [{i}] FX({cur_fx.k.index},{cur_fx.type}): same type, not better "
                          f"(val={cur_fx.val:.2f} vs {end_fx.val:.2f})")
            else:
                confirm = cd._bi_fx_valid(end_fx, cur_fx)
                cl_gap = cur_fx.k.index - end_fx.k.index
                k_gap = cur_fx.k.k_index - end_fx.k.k_index
                print(f"    [{i}] FX({cur_fx.k.index},{cur_fx.type}): CONFIRM? "
                      f"_bi_fx_valid(end={end_fx.k.index},cur={cur_fx.k.index})={confirm} "
                      f"cl_gap={cl_gap} k_gap={k_gap}")
                if confirm:
                    print(f"         → CONFIRM BI [{start_fx.k.index}→{end_fx.k.index}]")
                    break


if __name__ == "__main__":
    # ETH5m diffs:
    # #24: v3=('down', 240, 253) vs pyarmor=('down', 240, 244)
    diagnose_diff("ETH/USDT", "5m", 1000, 240, 244, 253)
    # #34: v3=('down', 346, 357) vs pyarmor=('down', 346, 363)
    diagnose_diff("ETH/USDT", "5m", 1000, 346, 363, 357)
    # #43: v3=('up', 448, 454) vs pyarmor=('up', 448, 461)
    diagnose_diff("ETH/USDT", "5m", 1000, 448, 461, 454)
    # BTC5m diff:
    # #59: v3=('up', 638, 642) vs pyarmor=('up', 638, 650)
    diagnose_diff("BTC/USDT", "5m", 1000, 638, 650, 642)
