# -*- coding: utf-8 -*-
"""Test v3 + threshold change: k_gap <= fx_check_k_nums instead of <"""
import sys, os, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor
from chanlun.cl_interface import Config, FX, BI

DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
LIMITS = {
    "BTC/USDT_d": 500, "BTC/USDT_60m": 1000, "BTC/USDT_5m": 1000,
    "ETH/USDT_60m": 1000, "ETH/USDT_5m": 1000,
}


def load(symbol, freq):
    limit = LIMITS.get(f"{symbol}_{freq}", 1000)
    cache_key = f"{symbol.replace('/', '_')}_{freq}_{limit}"
    return pd.read_parquet(os.path.join(DATA_DIR, f"{cache_key}.parquet"))


def bi_fx_valid_lte(self, start_fx, end_fx):
    """Same as _bi_fx_valid but with <= instead of < for threshold"""
    if start_fx.type == end_fx.type:
        return False

    cl_gap = end_fx.k.index - start_fx.k.index
    k_gap = end_fx.k.k_index - start_fx.k.k_index

    if self.bi_type == Config.BI_TYPE_DD.value:
        if cl_gap < 1:
            return False
    elif self.bi_type == Config.BI_TYPE_JDB.value:
        if k_gap < 4:
            return False
    else:
        if cl_gap < 4:
            return False

    # CHANGE: <= instead of <
    if k_gap <= self.fx_check_k_nums:
        if self.allow_bi_fx_strict:
            qj = self.fx_qj
            qy = self.fx_qy
            if start_fx.type == "ding" and end_fx.type == "di":
                if start_fx.low(qj, qy) < end_fx.low(qj, qy):
                    return False
                if end_fx.high(qj, qy) > start_fx.high(qj, qy):
                    return False
            elif start_fx.type == "di" and end_fx.type == "ding":
                if start_fx.high(qj, qy) > end_fx.high(qj, qy):
                    return False
                if end_fx.low(qj, qy) < start_fx.low(qj, qy):
                    return False

    return True


# Test all 5 cases with the threshold change
for symbol, freq in [("BTC/USDT", "60m"), ("BTC/USDT", "5m"), ("BTC/USDT", "d"),
                      ("ETH/USDT", "60m"), ("ETH/USDT", "5m")]:
    df = load(symbol, freq)

    cd = CL_Open(symbol, freq, config={})
    cd._bi_fx_valid = types.MethodType(bi_fx_valid_lte, cd)

    # Capture pre-split
    pre_split = []
    orig_split = cd._bi_special_bi_split.__func__
    def cap(self, bis, ps=pre_split):
        ps.extend(bis)
        return orig_split(self, bis)
    cd._bi_special_bi_split = types.MethodType(cap, cd)
    cd.process_klines(df)

    cd_py = CL_Pyarmor(symbol, freq, config={})
    cd_py.process_klines(df)

    # Reconstruct pyarmor pre-split
    py_bis = list(cd_py.bis)
    py_presplit = []
    j = 0
    while j < len(py_bis):
        bi = py_bis[j]
        if getattr(bi, 'is_split', False):
            merged = BI(start=bi.start, end=py_bis[j + 2].end, _type=bi.type,
                        index=len(py_presplit), default_zs_type=bi.default_zs_type)
            py_presplit.append(merged)
            j += 3
        else:
            py_presplit.append(bi)
            j += 1

    v_count = len(pre_split)
    py_ps_count = len(py_presplit)
    diffs = sum(1 for k in range(min(v_count, py_ps_count))
                if pre_split[k].start.k.index != py_presplit[k].start.k.index
                or pre_split[k].end.k.index != py_presplit[k].end.k.index)

    status = "✅" if len(cd.bis) == len(cd_py.bis) else "❌"
    print(f"{symbol:>12} {freq:>4}: open={len(cd.bis):>3} pyarmor={len(cd_py.bis):>3} {status}  "
          f"pre-split: {v_count:>3} vs {py_ps_count:>3} ({diffs} diffs)")

    if diffs > 0:
        for k in range(min(v_count, py_ps_count)):
            o = pre_split[k]
            p = py_presplit[k]
            if o.start.k.index != p.start.k.index or o.end.k.index != p.end.k.index:
                print(f"    #{k}: open={o.type[0]}[{o.start.k.index}→{o.end.k.index}] "
                      f"py={p.type[0]}[{p.start.k.index}→{p.end.k.index}]")
