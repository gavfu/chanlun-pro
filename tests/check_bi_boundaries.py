# -*- coding: utf-8 -*-
"""Check if BI start/end FX indices match for all 4 passing cases"""
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

for symbol, freq in [("BTC/USDT", "60m"), ("BTC/USDT", "5m"), ("BTC/USDT", "d"),
                      ("ETH/USDT", "60m"), ("ETH/USDT", "5m")]:
    limit = LIMITS.get(f"{symbol}_{freq}", 1000)
    cache_key = f"{symbol.replace('/', '_')}_{freq}_{limit}"
    df = pd.read_parquet(os.path.join(DATA_DIR, f"{cache_key}.parquet"))

    cd_o = CL_Open(symbol, freq, config={})
    cd_o.process_klines(df)
    cd_p = CL_Pyarmor(symbol, freq, config={})
    cd_p.process_klines(df)

    o_bis = list(cd_o.bis)
    p_bis = list(cd_p.bis)

    # Compare BI boundaries and high/low
    boundary_diffs = 0
    highlow_diffs = 0
    n = min(len(o_bis), len(p_bis))
    for i in range(n):
        ob, pb = o_bis[i], p_bis[i]
        b_match = (ob.start.k.index == pb.start.k.index and
                   ob.end.k.index == pb.end.k.index and
                   ob.type == pb.type)
        if not b_match:
            boundary_diffs += 1
        elif abs(ob.high - pb.high) > 0.01 or abs(ob.low - pb.low) > 0.01:
            highlow_diffs += 1

    status = "✅" if len(o_bis) == len(p_bis) and boundary_diffs == 0 and highlow_diffs == 0 else ""
    print(f"{symbol:>12} {freq:>4}: bis={len(o_bis):>3}/{len(p_bis):>3}  "
          f"boundary_diffs={boundary_diffs}  highlow_diffs={highlow_diffs}  {status}")

    if boundary_diffs > 0:
        for i in range(n):
            ob, pb = o_bis[i], p_bis[i]
            if (ob.start.k.index != pb.start.k.index or
                ob.end.k.index != pb.end.k.index or ob.type != pb.type):
                split_o = getattr(ob, 'is_split', False)
                split_p = getattr(pb, 'is_split', False)
                print(f"    #{i}: open={ob.type[0]}[{ob.start.k.index}→{ob.end.k.index}] "
                      f"py={pb.type[0]}[{pb.start.k.index}→{pb.end.k.index}] "
                      f"split_o={split_o} split_p={split_p}")
    if highlow_diffs > 0:
        for i in range(n):
            ob, pb = o_bis[i], p_bis[i]
            if (ob.start.k.index == pb.start.k.index and
                ob.end.k.index == pb.end.k.index):
                if abs(ob.high - pb.high) > 0.01 or abs(ob.low - pb.low) > 0.01:
                    split_o = getattr(ob, 'is_split', False)
                    split_p = getattr(pb, 'is_split', False)
                    print(f"    #{i}: {ob.type[0]}[{ob.start.k.index}→{ob.end.k.index}] "
                          f"h={ob.high:.2f}/{pb.high:.2f} l={ob.low:.2f}/{pb.low:.2f} "
                          f"split_o={split_o} split_p={split_p}")
