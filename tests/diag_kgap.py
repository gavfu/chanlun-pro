# -*- coding: utf-8 -*-
"""Check if pyarmor uses k_index (original bar) gaps for split selection"""
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
    i = 0
    while i < len(p_bis):
        bi = p_bis[i]
        if getattr(bi, 'is_split', False) and i + 2 < len(p_bis):
            parent_type = bi.type
            parent_start_idx = bi.start.k.index
            parent_end_idx = p_bis[i + 2].end.k.index
            parent_start_k = bi.start.k.k_index
            parent_end_k = p_bis[i + 2].end.k.k_index

            if parent_type == "down":
                split1_idx = bi.end.k.index
                split1_k = bi.end.k.k_index
                split2_idx = p_bis[i + 1].end.k.index
                split2_k = p_bis[i + 1].end.k.k_index
                split1_type = "di"
                split2_type = "ding"
            else:
                split1_idx = bi.end.k.index
                split1_k = bi.end.k.k_index
                split2_idx = p_bis[i + 1].end.k.index
                split2_k = p_bis[i + 1].end.k.k_index
                split1_type = "ding"
                split2_type = "di"

            cl_gap_start = split1_idx - parent_start_idx
            cl_gap_mid = split2_idx - split1_idx
            cl_gap_end = parent_end_idx - split2_idx
            k_gap_start = split1_k - parent_start_k
            k_gap_mid = split2_k - split1_k
            k_gap_end = parent_end_k - split2_k

            print(f"{symbol} {freq}: {parent_type}[{parent_start_idx}→{parent_end_idx}]")
            print(f"  Splits: {split1_type}[{split1_idx}] → {split2_type}[{split2_idx}]")
            print(f"  cl_gaps: start={cl_gap_start} mid={cl_gap_mid} end={cl_gap_end}")
            print(f"  k_gaps:  start={k_gap_start} mid={k_gap_mid} end={k_gap_end}")

            # Show ALL internal FXs with both gap types
            internal = [fx for fx in cd_o.fxs if parent_start_idx < fx.k.index < parent_end_idx]
            if parent_type == "down":
                first_type_fxs = sorted([fx for fx in internal if fx.type == "di"], key=lambda f: f.k.index)
            else:
                first_type_fxs = sorted([fx for fx in internal if fx.type == "ding"], key=lambda f: f.k.index)

            print(f"  First-type ({split1_type}) candidates by position:")
            for fx in first_type_fxs:
                cl_g = fx.k.index - parent_start_idx
                k_g = fx.k.k_index - parent_start_k
                chosen = " <<<" if fx.k.index == split1_idx else ""
                print(f"    {split1_type}[{fx.k.index}] val={fx.val:.2f} cl_gap_start={cl_g} k_gap_start={k_g}{chosen}")

            i += 3
        else:
            i += 1
