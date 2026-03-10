# -*- coding: utf-8 -*-
"""Deep comparison of pre-split BI counts between cl_open and cl_pyarmor"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor
from chanlun.cl_interface import BI

DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")

def load(symbol, freq, limit):
    cache_key = f"{symbol.replace('/', '_')}_{freq}_{limit}"
    return pd.read_parquet(os.path.join(DATA_DIR, f"{cache_key}.parquet"))


def analyze(symbol, freq, limit):
    df = load(symbol, freq, limit)

    # cl_open: capture pre-split BIs
    cd_o = CL_Open(symbol, freq, config={})
    pre_split_open = []

    orig_split = cd_o._bi_special_bi_split.__func__
    def capture_split(self, bis):
        pre_split_open.extend([(b.index, b.type, b.start.k.index, b.end.k.index) for b in bis])
        return orig_split(self, bis)
    import types
    cd_o._bi_special_bi_split = types.MethodType(capture_split, cd_o)

    cd_o.process_klines(df)
    o_bis = cd_o.get_bis()

    # cl_pyarmor
    cd_p = CL_Pyarmor(symbol, freq, config={})
    cd_p.process_klines(df)
    p_bis = cd_p.get_bis()

    print(f"\n{'='*60}")
    print(f"  {symbol} {freq}")
    print(f"{'='*60}")
    print(f"  Open pre-split: {len(pre_split_open)}, final: {len(o_bis)}")
    print(f"  Pyarmor final: {len(p_bis)}")

    # Check is_split across ALL pyarmor BIs
    split_true = [b for b in p_bis if b.is_split]
    split_false = [b for b in p_bis if not b.is_split]
    print(f"  Pyarmor is_split=True: {len(split_true)}, is_split=False: {len(split_false)}")

    if split_true:
        print(f"  Split BIs:")
        for b in split_true:
            print(f"    #{b.index} {b.type} [{b.start.k.index}→{b.end.k.index}] val:{b.start.val:.2f}→{b.end.val:.2f}")

    # Compare first divergence in pre-split open vs pyarmor BIs
    print(f"\n  First BI differences (pre-split open vs pyarmor):")
    p_bi_list = [(b.index, b.type, b.start.k.index, b.end.k.index) for b in p_bis]
    shown = 0
    for i in range(min(len(pre_split_open), len(p_bi_list))):
        o = pre_split_open[i]
        p = p_bi_list[i]
        if o[2] != p[2] or o[3] != p[3] or o[1] != p[1]:
            print(f"    #{i}: open=({o[1]} {o[2]}→{o[3]}) pyarmor=({p[1]} {p[2]}→{p[3]})")
            shown += 1
            if shown >= 10:
                print(f"    ... (more diffs)")
                break

    if len(pre_split_open) != len(p_bi_list):
        print(f"    Count diff: open_presplit={len(pre_split_open)} vs pyarmor_final={len(p_bi_list)}")


if __name__ == "__main__":
    cases = [
        ("BTC/USDT", "60m", 1000),
        ("BTC/USDT", "5m", 1000),
        ("BTC/USDT", "d", 500),
        ("ETH/USDT", "60m", 1000),
        ("ETH/USDT", "5m", 1000),
    ]
    for sym, freq, lim in cases:
        analyze(sym, freq, lim)
