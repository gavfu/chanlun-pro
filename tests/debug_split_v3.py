# -*- coding: utf-8 -*-
"""Debug split for specific BIs across remaining failing cases"""
import os, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_interface import FX, BI

DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")

def load(symbol, freq, limit):
    cache_key = f"{symbol.replace('/', '_')}_{freq}_{limit}"
    return pd.read_parquet(os.path.join(DATA_DIR, f"{cache_key}.parquet"))


def debug_bi_split(symbol, freq, limit, bi_start_idx, bi_end_idx):
    """Debug why a specific BI doesn't get split"""
    df = load(symbol, freq, limit)
    cd = CL_Open(symbol, freq, config={})

    # Capture pre-split BIs
    pre_split = []
    orig_fn = cd._bi_special_bi_split.__func__
    def cap(self, bis):
        pre_split.extend(bis)
        return orig_fn(self, bis)
    cd._bi_special_bi_split = types.MethodType(cap, cd)
    cd.process_klines(df)

    qj, qy = cd.fx_qj, cd.fx_qy
    threshold = cd.bi_split_k_cross_nums
    tolerance = cd.bi_split_k_cross_tolerance

    # Find the target BI
    target_bi = None
    for bi in pre_split:
        if bi.start.k.index == bi_start_idx and bi.end.k.index == bi_end_idx:
            target_bi = bi
            break

    if target_bi is None:
        print(f"\n{symbol} {freq}: BI [{bi_start_idx}→{bi_end_idx}] NOT FOUND in pre-split BIs")
        print(f"  Pre-split BIs: {[(b.start.k.index, b.end.k.index) for b in pre_split[-10:]]}")
        return

    bi = target_bi
    print(f"\n{'='*70}")
    print(f"  {symbol} {freq}: Debug split for BI [{bi.start.k.index}→{bi.end.k.index}] type={bi.type}")
    print(f"  threshold={threshold}, tolerance={tolerance}")
    print(f"{'='*70}")

    # Find internal FXs
    start_idx = bi.start.k.index
    end_idx = bi.end.k.index
    internal_fxs = [fx for fx in cd.fxs if start_idx < fx.k.index < end_idx]

    print(f"\n  Internal FXs ({len(internal_fxs)}):")
    for fx in internal_fxs:
        print(f"    FX({fx.k.index},{fx.type}) val={fx.val:.2f} k_index={fx.k.k_index} "
              f"h={fx.high(qj,qy):.2f} l={fx.low(qj,qy):.2f}")

    if len(internal_fxs) < 3:
        print(f"\n  < 3 internal FXs, no split possible")
        return

    # Phase 1: Cross-counting
    end_ki = bi.end.k.k_index
    print(f"\n  Cross-counting (end_ki={end_ki}):")
    for ti in range(len(internal_fxs) - 2):
        fx1, fx2, fx3 = internal_fxs[ti], internal_fxs[ti+1], internal_fxs[ti+2]
        h1, l1 = fx1.high(qj, qy), fx1.low(qj, qy)
        h2, l2 = fx2.high(qj, qy), fx2.low(qj, qy)
        h3, l3 = fx3.high(qj, qy), fx3.low(qj, qy)

        hit, miss = 0, 0
        for ki in range(fx1.k.k_index, end_ki):
            k = cd.src_klines[ki]
            if (k.h >= l1 and k.l <= h1 and k.h >= l2 and k.l <= h2 and k.h >= l3 and k.l <= h3):
                hit += 1; miss = 0
            else:
                miss += 1
            if miss > tolerance:
                break

        triggered = hit >= threshold
        print(f"    Triplet {ti}: FX({fx1.k.index},{fx2.k.index},{fx3.k.index}) "
              f"hits={hit} {'→ TRIGGERED!' if triggered else ''}")
        if triggered:
            break

    if not triggered:
        print(f"\n  No triplet triggered → NO SPLIT")
    else:
        # Phase 2: Selection
        result = cd._try_split_bi(bi, qj, qy, threshold, tolerance)
        print(f"\n  Split result: {len(result)} BIs")
        for b in result:
            print(f"    {b.type} [{b.start.k.index}→{b.end.k.index}]")


if __name__ == "__main__":
    # BTCd: pyarmor splits up[330→352]
    debug_bi_split("BTC/USDT", "d", 500, 330, 352)

    # BTC5m: pyarmor splits down[429→445] and down[650→672]
    debug_bi_split("BTC/USDT", "5m", 1000, 429, 445)
    debug_bi_split("BTC/USDT", "5m", 1000, 650, 672)

    # ETH5m: pyarmor splits down[191→213], down[346→363], down[617→641]
    debug_bi_split("ETH/USDT", "5m", 1000, 191, 213)
    debug_bi_split("ETH/USDT", "5m", 1000, 346, 363)
    debug_bi_split("ETH/USDT", "5m", 1000, 617, 641)
