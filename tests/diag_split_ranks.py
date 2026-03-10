# -*- coding: utf-8 -*-
"""Compare split sub-BI boundaries between cl_open and pyarmor to deduce selection logic"""
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


for symbol, freq in [("BTC/USDT", "60m"), ("BTC/USDT", "5m"), ("BTC/USDT", "d"),
                      ("ETH/USDT", "60m"), ("ETH/USDT", "5m")]:
    df = load(symbol, freq)
    cd_o = CL_Open(symbol, freq, config={})
    cd_o.process_klines(df)
    cd_p = CL_Pyarmor(symbol, freq, config={})
    cd_p.process_klines(df)

    qj, qy = cd_o.fx_qj, cd_o.fx_qy
    o_bis = list(cd_o.bis)
    p_bis = list(cd_p.bis)

    # Find pyarmor's split groups
    splits = []
    i = 0
    while i < len(p_bis):
        bi = p_bis[i]
        if getattr(bi, 'is_split', False) and i + 2 < len(p_bis):
            parent_type = bi.type
            parent_start = bi.start.k.index
            parent_end = p_bis[i + 2].end.k.index
            split1 = (bi.start.k.index, bi.end.k.index)
            split2 = (p_bis[i + 1].start.k.index, p_bis[i + 1].end.k.index)
            split3 = (p_bis[i + 2].start.k.index, p_bis[i + 2].end.k.index)

            # Get the split FXs
            if parent_type == "down":
                # start(ding)→di_fx→ding_fx→end(di)
                di_fx_idx = bi.end.k.index  # first sub-BI end
                ding_fx_idx = p_bis[i + 1].end.k.index  # second sub-BI end
            else:
                # start(di)→ding_fx→di_fx→end(ding)
                ding_fx_idx = bi.end.k.index
                di_fx_idx = p_bis[i + 1].end.k.index

            # Get internal FXs
            internal = [(fx.k.index, fx.type, fx.val)
                        for fx in cd_o.fxs
                        if parent_start < fx.k.index < parent_end]

            if parent_type == "down":
                di_fxs = [(idx, val) for idx, t, val in internal if t == "di"]
                ding_fxs = [(idx, val) for idx, t, val in internal if t == "ding"]
                # Sort by pyarmor's expected order
                di_fxs_sorted = sorted(di_fxs, key=lambda x: x[1])  # lowest first
                ding_fxs_sorted = sorted(ding_fxs, key=lambda x: x[1], reverse=True)

                # Find rank of pyarmor's chosen split point
                di_rank = next((r for r, (idx, v) in enumerate(di_fxs_sorted) if idx == di_fx_idx), -1)
                ding_rank = next((r for r, (idx, v) in enumerate(ding_fxs_sorted) if idx == ding_fx_idx), -1)
            else:
                di_fxs = [(idx, val) for idx, t, val in internal if t == "di"]
                ding_fxs = [(idx, val) for idx, t, val in internal if t == "ding"]
                ding_fxs_sorted = sorted(ding_fxs, key=lambda x: x[1], reverse=True)
                di_fxs_sorted = sorted(di_fxs, key=lambda x: x[1])

                ding_rank = next((r for r, (idx, v) in enumerate(ding_fxs_sorted) if idx == ding_fx_idx), -1)
                di_rank = next((r for r, (idx, v) in enumerate(di_fxs_sorted) if idx == di_fx_idx), -1)

            splits.append({
                'parent': f"{parent_type}[{parent_start}→{parent_end}]",
                'subs': f"[{split1[0]}→{split1[1]}] [{split2[0]}→{split2[1]}] [{split3[0]}→{split3[1]}]",
                'di_fx': di_fx_idx, 'ding_fx': ding_fx_idx,
                'di_rank': di_rank, 'ding_rank': ding_rank,
                'n_di': len(di_fxs), 'n_ding': len(ding_fxs),
            })
            i += 3
        else:
            i += 1

    if splits:
        print(f"\n{symbol} {freq}: {len(splits)} splits")
        for s in splits:
            print(f"  Parent: {s['parent']}")
            print(f"    Sub-BIs: {s['subs']}")
            print(f"    di_fx={s['di_fx']} (rank {s['di_rank']}/{s['n_di']}), "
                  f"ding_fx={s['ding_fx']} (rank {s['ding_rank']}/{s['n_ding']})")
