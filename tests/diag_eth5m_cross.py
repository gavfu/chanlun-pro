# -*- coding: utf-8 -*-
"""Check if cl_open's d[346→357] triggers split vs pyarmor's d[346→363]"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_Open

DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')

df = pd.read_parquet(os.path.join(DATA_DIR, "ETH_USDT_5m_1000.parquet"))
cd = CL_Open("ETH/USDT", "5m", config={})
cd.process_klines(df)

qj, qy = cd.fx_qj, cd.fx_qy
threshold = cd.bi_split_k_cross_nums
tolerance = cd.bi_split_k_cross_tolerance

print(f"Config: threshold={threshold}, tolerance={tolerance}")

# Test cross-counting for both BI ranges
for end_idx in [357, 363]:
    start_idx = 346
    start_fx = next(fx for fx in cd.fxs if fx.k.index == start_idx)
    end_fx = next(fx for fx in cd.fxs if fx.k.index == end_idx)
    end_ki = end_fx.k.k_index

    internal = [fx for fx in cd.fxs if start_idx < fx.k.index < end_idx]
    print(f"\nBI d[{start_idx}→{end_idx}]: {len(internal)} internal FXs")
    for fx in internal:
        print(f"  {fx.type}[{fx.k.index}] val={fx.val:.2f}")

    max_hits = 0
    best_triplet = None
    for ti in range(len(internal) - 2):
        fx1, fx2, fx3 = internal[ti], internal[ti+1], internal[ti+2]
        h1, l1 = fx1.high(qj, qy), fx1.low(qj, qy)
        h2, l2 = fx2.high(qj, qy), fx2.low(qj, qy)
        h3, l3 = fx3.high(qj, qy), fx3.low(qj, qy)
        hit_count = 0
        miss_count = 0
        for ki in range(fx1.k.k_index, end_ki):
            k = cd.src_klines[ki]
            if (k.h >= l1 and k.l <= h1
                    and k.h >= l2 and k.l <= h2
                    and k.h >= l3 and k.l <= h3):
                hit_count += 1
                miss_count = 0
            else:
                miss_count += 1
            if miss_count > tolerance:
                break
        print(f"  triplet ti={ti}: {fx1.type}[{fx1.k.index}] {fx2.type}[{fx2.k.index}] "
              f"{fx3.type}[{fx3.k.index}] hits={hit_count}")
        if hit_count > max_hits:
            max_hits = hit_count
            best_triplet = ti

    triggered = max_hits >= threshold
    print(f"  Max hits: {max_hits}, triggered: {triggered}")
