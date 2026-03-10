# -*- coding: utf-8 -*-
"""诊断：对比 cl_open 笔拆分行为 vs cl_pyarmor"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor
from chanlun.cl_interface import FX, BI

DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")

def load(symbol, freq, limit):
    cache_key = f"{symbol.replace('/', '_')}_{freq}_{limit}"
    return pd.read_parquet(os.path.join(DATA_DIR, f"{cache_key}.parquet"))


def inspect_splits(symbol, freq, limit):
    """Run cl_open with split logging, compare with pyarmor"""
    df = load(symbol, freq, limit)

    # Run pyarmor as reference
    cd_p = CL_Pyarmor(symbol, freq, config={})
    cd_p.process_klines(df)
    p_bis = cd_p.get_bis()

    # Run open - but first get BIs BEFORE split
    cd_o = CL_Open(symbol, freq, config={})

    # Monkey-patch to capture pre-split BIs and split decisions
    orig_split = cd_o._bi_special_bi_split
    pre_split_bis = []
    split_decisions = []

    def patched_split(bis):
        pre_split_bis.extend(bis)
        qj = cd_o.fx_qj
        qy = cd_o.fx_qy
        threshold = cd_o.bi_split_k_cross_nums
        tolerance = cd_o.bi_split_k_cross_tolerance

        result = []
        for bi in bis:
            # Check if this BI triggers
            start_idx = bi.start.k.index
            end_idx = bi.end.k.index
            internal_fxs = [fx for fx in cd_o.fxs
                            if start_idx < fx.k.index < end_idx]

            triggered = False
            trigger_triplet = None
            trigger_hits = 0

            if len(internal_fxs) >= 3:
                end_ki = bi.end.k.k_index
                for ti in range(len(internal_fxs) - 2):
                    fx1, fx2, fx3 = internal_fxs[ti], internal_fxs[ti+1], internal_fxs[ti+2]
                    h1, l1 = fx1.high(qj, qy), fx1.low(qj, qy)
                    h2, l2 = fx2.high(qj, qy), fx2.low(qj, qy)
                    h3, l3 = fx3.high(qj, qy), fx3.low(qj, qy)

                    hit, miss = 0, 0
                    for ki in range(fx1.k.k_index, end_ki):
                        k = cd_o.src_klines[ki]
                        if (k.h >= l1 and k.l <= h1
                                and k.h >= l2 and k.l <= h2
                                and k.h >= l3 and k.l <= h3):
                            hit += 1
                            miss = 0
                        else:
                            miss += 1
                        if miss > tolerance:
                            break

                    if hit >= threshold:
                        triggered = True
                        trigger_triplet = ti
                        trigger_hits = hit
                        break

            if triggered:
                split_result = cd_o._try_split_bi(bi, qj, qy, threshold, tolerance)
                split_decisions.append({
                    'bi_idx': bi.index,
                    'bi_type': bi.type,
                    'start': bi.start.k.index,
                    'end': bi.end.k.index,
                    'internal_fxs': len(internal_fxs),
                    'triplet': trigger_triplet,
                    'hits': trigger_hits,
                    'split_count': len(split_result),
                    'split_details': [(b.start.k.index, b.end.k.index, b.type) for b in split_result]
                })
                result.extend(split_result)
            else:
                result.extend([bi])

        for i, b in enumerate(result):
            b.index = i
        return result

    cd_o._bi_special_bi_split = patched_split
    cd_o.process_klines(df)
    o_bis = cd_o.get_bis()

    print(f"\n{'='*60}")
    print(f"  {symbol} {freq}: open_bis={len(o_bis)}, pyarmor_bis={len(p_bis)}")
    print(f"  Pre-split: {len(pre_split_bis)} BIs")
    print(f"{'='*60}")

    if split_decisions:
        print(f"\n  Splits triggered ({len(split_decisions)}):")
        for d in split_decisions:
            print(f"    BI#{d['bi_idx']} {d['bi_type']} [{d['start']}→{d['end']}] "
                  f"ifx={d['internal_fxs']} triplet={d['triplet']} hits={d['hits']} "
                  f"→ {d['split_details']}")
    else:
        print(f"\n  No splits triggered")

    # Check which pyarmor BIs have is_split set
    p_split_bis = [b for b in p_bis if getattr(b, 'is_split', False)]
    if p_split_bis:
        print(f"\n  Pyarmor split BIs ({len(p_split_bis)}):")
        for b in p_split_bis:
            print(f"    BI#{b.index} {b.type} [{b.start.k.index}→{b.end.k.index}]")

    return o_bis, p_bis


if __name__ == "__main__":
    cases = [
        ("BTC/USDT", "60m", 1000),
        ("ETH/USDT", "60m", 1000),
        ("ETH/USDT", "5m", 1000),
    ]
    for sym, freq, lim in cases:
        inspect_splits(sym, freq, lim)
