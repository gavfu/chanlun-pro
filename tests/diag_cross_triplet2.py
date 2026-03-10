# -*- coding: utf-8 -*-
"""Check which cross-counting triplet triggers and compare to split points"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor

DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
LIMITS = {
    "BTC/USDT_d": 500, "BTC/USDT_60m": 1000, "BTC/USDT_5m": 1000,
    "ETH/USDT_60m": 1000, "ETH/USDT_5m": 1000,
}


def load(symbol, freq):
    limit = LIMITS.get(f"{symbol}_{freq}", 1000)
    cache_key = f"{symbol.replace('/', '_')}_{freq}_{limit}"
    return pd.read_parquet(os.path.join(DATA_DIR, f"{cache_key}.parquet"))


for symbol, freq in [("BTC/USDT", "60m"), ("BTC/USDT", "5m"), ("BTC/USDT", "d"),
                      ("ETH/USDT", "60m"), ("ETH/USDT", "5m")]:
    df = load(symbol, freq)
    cd_o = CL_Open(symbol, freq, config={})
    cd_o.process_klines(df)
    cd_p = CL_Pyarmor(symbol, freq, config={})
    cd_p.process_klines(df)

    qj, qy = cd_o.fx_qj, cd_o.fx_qy
    threshold = cd_o.bi_split_k_cross_nums
    tolerance = cd_o.bi_split_k_cross_tolerance

    # Build FX index map
    fxs_by_idx = {fx.k.index: fx for fx in cd_o.fxs}

    p_bis = list(cd_p.bis)
    i = 0
    while i < len(p_bis):
        bi = p_bis[i]
        if getattr(bi, 'is_split', False) and i + 2 < len(p_bis):
            parent_type = bi.type
            parent_start = bi.start.k.index
            parent_end = p_bis[i + 2].end.k.index

            if parent_type == "down":
                split1_idx = bi.end.k.index  # di
                split2_idx = p_bis[i + 1].end.k.index  # ding
            else:
                split1_idx = bi.end.k.index  # ding
                split2_idx = p_bis[i + 1].end.k.index  # di

            # Get FXs from cl_open
            start_fx = fxs_by_idx.get(parent_start)
            end_fx = fxs_by_idx.get(parent_end)
            if not start_fx or not end_fx:
                i += 3
                continue

            end_ki = end_fx.k.k_index
            internal_fxs = [fx for fx in cd_o.fxs
                           if parent_start < fx.k.index < parent_end]

            if len(internal_fxs) < 3:
                i += 3
                continue

            # Find ALL triplets that pass threshold
            passing_triplets = []
            first_triggered = None
            for ti in range(len(internal_fxs) - 2):
                fx1 = internal_fxs[ti]
                fx2 = internal_fxs[ti + 1]
                fx3 = internal_fxs[ti + 2]
                h1, l1 = fx1.high(qj, qy), fx1.low(qj, qy)
                h2, l2 = fx2.high(qj, qy), fx2.low(qj, qy)
                h3, l3 = fx3.high(qj, qy), fx3.low(qj, qy)
                hit_count = 0
                miss_count = 0
                for ki in range(fx1.k.k_index, end_ki):
                    k = cd_o.src_klines[ki]
                    if (k.h >= l1 and k.l <= h1
                            and k.h >= l2 and k.l <= h2
                            and k.h >= l3 and k.l <= h3):
                        hit_count += 1
                        miss_count = 0
                    else:
                        miss_count += 1
                    if miss_count > tolerance:
                        break
                if hit_count >= threshold:
                    passing_triplets.append((ti, fx1, fx2, fx3, hit_count))
                    if first_triggered is None:
                        first_triggered = (ti, fx1, fx2, fx3, hit_count)

            print(f"\n{symbol} {freq}: {parent_type}[{parent_start}→{parent_end}]")
            print(f"  Pyarmor splits: [{split1_idx}] [{split2_idx}]")

            if first_triggered:
                ti, fx1, fx2, fx3, hits = first_triggered
                print(f"  FIRST triggered triplet (ti={ti}): "
                      f"{fx1.type}[{fx1.k.index}] {fx2.type}[{fx2.k.index}] {fx3.type}[{fx3.k.index}] "
                      f"hits={hits}")

            if passing_triplets:
                # Find the LAST triggered triplet
                last = passing_triplets[-1]
                print(f"  LAST triggered triplet (ti={last[0]}): "
                      f"{last[1].type}[{last[1].k.index}] {last[2].type}[{last[2].k.index}] "
                      f"{last[3].type}[{last[3].k.index}] hits={last[4]}")
                print(f"  Total passing triplets: {len(passing_triplets)}")

                # Show all passing triplets
                for tj, fx1, fx2, fx3, hits in passing_triplets:
                    # Check if split points are in this triplet
                    contains_split1 = split1_idx in (fx1.k.index, fx2.k.index, fx3.k.index)
                    contains_split2 = split2_idx in (fx1.k.index, fx2.k.index, fx3.k.index)
                    marker = ""
                    if contains_split1:
                        marker += " [has split1]"
                    if contains_split2:
                        marker += " [has split2]"
                    print(f"    ti={tj}: {fx1.type}[{fx1.k.index}] "
                          f"{fx2.type}[{fx2.k.index}] {fx3.type}[{fx3.k.index}] "
                          f"hits={hits}{marker}")

            i += 3
        else:
            i += 1
