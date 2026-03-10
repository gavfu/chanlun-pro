# -*- coding: utf-8 -*-
"""Verify split2 selection: is it highest-value or first-by-position among valid candidates?"""
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

    p_bis = list(cd_p.bis)
    fxs_by_idx = {fx.k.index: fx for fx in cd_o.fxs}

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
                split1_type, split2_type = "di", "ding"
            else:
                split1_idx = bi.end.k.index  # ding
                split2_idx = p_bis[i + 1].end.k.index  # di
                split1_type, split2_type = "ding", "di"

            split1_fx = fxs_by_idx.get(split1_idx)
            internal = [fx for fx in cd_o.fxs if parent_start < fx.k.index < parent_end]

            # Get all split2-type candidates after split1
            candidates2 = sorted(
                [fx for fx in internal if fx.type == split2_type and fx.k.index > split1_idx],
                key=lambda f: f.k.index
            )

            print(f"\n{symbol} {freq}: {parent_type}[{parent_start}→{parent_end}]")
            print(f"  Split1: {split1_type}[{split1_idx}] val={split1_fx.val:.2f}")
            print(f"  Split2: {split2_type}[{split2_idx}] (pyarmor)")

            if not candidates2:
                i += 3
                continue

            # Check each candidate
            first_valid = None
            highest_valid = None
            for fx in candidates2:
                cl_gap_mid = fx.k.index - split1_idx
                if parent_type == "down":
                    ding_gt_di = fx.val > split1_fx.val
                else:
                    ding_gt_di = split1_fx.val > fx.val

                marker = " <<<" if fx.k.index == split2_idx else ""
                print(f"    {split2_type}[{fx.k.index}] val={fx.val:.2f} cl_gap_mid={cl_gap_mid} "
                      f"valid_dir={ding_gt_di}{marker}")

                if ding_gt_di:
                    if first_valid is None:
                        first_valid = fx
                    if highest_valid is None or \
                       (parent_type == "down" and fx.val > highest_valid.val) or \
                       (parent_type == "up" and fx.val < highest_valid.val):
                        highest_valid = fx

            if first_valid:
                print(f"  First valid: {split2_type}[{first_valid.k.index}] val={first_valid.val:.2f}")
            if highest_valid:
                print(f"  Most extreme: {split2_type}[{highest_valid.k.index}] val={highest_valid.val:.2f}")

            # Check: which one matches pyarmor?
            if first_valid and first_valid.k.index == split2_idx:
                print(f"  → FIRST VALID matches pyarmor")
            elif highest_valid and highest_valid.k.index == split2_idx:
                print(f"  → MOST EXTREME matches pyarmor")
            else:
                print(f"  → NEITHER matches pyarmor!")

            i += 3
        else:
            i += 1
