# -*- coding: utf-8 -*-
"""Print first N BIs side-by-side for open pre-split vs pyarmor"""
import os, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor

DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")

def load(symbol, freq, limit):
    cache_key = f"{symbol.replace('/', '_')}_{freq}_{limit}"
    return pd.read_parquet(os.path.join(DATA_DIR, f"{cache_key}.parquet"))


def print_side_by_side(symbol, freq, limit, n=30):
    df = load(symbol, freq, limit)

    # cl_open: capture pre-split BIs
    cd_o = CL_Open(symbol, freq, config={})
    pre_split = []

    orig_fn = cd_o._bi_special_bi_split.__func__
    def capture(self, bis):
        pre_split.extend(bis)
        return orig_fn(self, bis)
    cd_o._bi_special_bi_split = types.MethodType(capture, cd_o)
    cd_o.process_klines(df)

    # cl_pyarmor
    cd_p = CL_Pyarmor(symbol, freq, config={})
    cd_p.process_klines(df)
    p_bis = cd_p.get_bis()

    print(f"\n{'='*80}")
    print(f"  {symbol} {freq}  (open_pre={len(pre_split)}, pyarmor={len(p_bis)})")
    print(f"{'='*80}")
    print(f"{'#':>3} {'OPEN (pre-split)':^30} {'':^5} {'PYARMOR':^30} {'':^5}")
    print(f"{'':>3} {'type  start→end':^30} {'':^5} {'type  start→end':^30} {'match':^5}")
    print(f"{'-'*78}")

    max_i = min(n, max(len(pre_split), len(p_bis)))
    for i in range(max_i):
        o_str = ""
        p_str = ""
        if i < len(pre_split):
            b = pre_split[i]
            o_str = f"{b.type:>4} {b.start.k.index:>4}→{b.end.k.index:<4}"
        if i < len(p_bis):
            b = p_bis[i]
            sp = "*" if b.is_split else " "
            p_str = f"{b.type:>4} {b.start.k.index:>4}→{b.end.k.index:<4}{sp}"

        match = ""
        if i < len(pre_split) and i < len(p_bis):
            o = pre_split[i]
            p = p_bis[i]
            if o.type == p.type and o.start.k.index == p.start.k.index and o.end.k.index == p.end.k.index:
                match = "  ✅"
            else:
                match = "  ❌"

        print(f"{i:>3} {o_str:^30} {'|':^5} {p_str:^30} {match}")


if __name__ == "__main__":
    print_side_by_side("BTC/USDT", "60m", 1000, n=30)
    print_side_by_side("ETH/USDT", "60m", 1000, n=30)
    print_side_by_side("ETH/USDT", "5m", 1000, n=30)
