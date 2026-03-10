# -*- coding: utf-8 -*-
"""
Compare _bi_fx_valid results between cl_open and cl_pyarmor
for the specific FX pairs where remaining diffs exist.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor

DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")

def load(symbol, freq, limit):
    cache_key = f"{symbol.replace('/', '_')}_{freq}_{limit}"
    return pd.read_parquet(os.path.join(DATA_DIR, f"{cache_key}.parquet"))


def compare_fx_valid(symbol, freq, limit, pairs_desc):
    """
    pairs_desc: list of (start_index, end_index, description)
    """
    df = load(symbol, freq, limit)

    cd_o = CL_Open(symbol, freq, config={})
    cd_o.process_klines(df)
    o_fxs = cd_o.get_fxs()

    cd_p = CL_Pyarmor(symbol, freq, config={})
    cd_p.process_klines(df)
    p_fxs = cd_p.get_fxs()

    print(f"\n{'='*70}")
    print(f"  {symbol} {freq}: _bi_fx_valid comparison")
    print(f"{'='*70}")

    qj_o, qy_o = cd_o.fx_qj, cd_o.fx_qy
    # Pyarmor stores config differently
    qj_p = getattr(cd_p, 'fx_qj', cd_p.config.get('fx_qj', 'fx_qj_k') if hasattr(cd_p, 'config') else 'fx_qj_k')
    qy_p = getattr(cd_p, 'fx_qy', cd_p.config.get('fx_qy', 'fx_qy_k') if hasattr(cd_p, 'config') else 'fx_qy_k')
    print(f"  Open: qj={qj_o}, qy={qy_o}")
    print(f"  Pyarmor: qj={qj_p}, qy={qy_p}")

    for s_idx, e_idx, desc in pairs_desc:
        o_start = next((fx for fx in o_fxs if fx.k.index == s_idx), None)
        o_end = next((fx for fx in o_fxs if fx.k.index == e_idx), None)
        p_start = next((fx for fx in p_fxs if fx.k.index == s_idx), None)
        p_end = next((fx for fx in p_fxs if fx.k.index == e_idx), None)

        if not (o_start and o_end and p_start and p_end):
            print(f"\n  [{desc}] FX({s_idx})→FX({e_idx}): Missing FX!")
            continue

        o_result = cd_o._bi_fx_valid(o_start, o_end)
        # Pyarmor uses _bi_check_bi_fx_ok
        try:
            p_result = cd_p._bi_check_bi_fx_ok(p_start, p_end)
        except Exception as e:
            p_result = f"ERROR: {e}"

        match = "✅" if o_result == p_result else "❌ MISMATCH"
        print(f"\n  [{desc}] FX({s_idx},{o_start.type})→FX({e_idx},{o_end.type}): "
              f"open={o_result}, pyarmor={p_result} {match}")

        if o_result != p_result:
            cl_gap = o_end.k.index - o_start.k.index
            k_gap = o_end.k.k_index - o_start.k.k_index
            print(f"    cl_gap={cl_gap}, k_gap={k_gap}, fx_check_k_nums={cd_o.fx_check_k_nums}")
            print(f"    Open  start: h={o_start.high(qj_o,qy_o):.2f} l={o_start.low(qj_o,qy_o):.2f}")
            print(f"    Open  end:   h={o_end.high(qj_o,qy_o):.2f} l={o_end.low(qj_o,qy_o):.2f}")
            print(f"    Pyarm start: h={p_start.high(qj_p,qy_p):.2f} l={p_start.low(qj_p,qy_p):.2f}")
            print(f"    Pyarm end:   h={p_end.high(qj_p,qy_p):.2f} l={p_end.low(qj_p,qy_p):.2f}")


if __name__ == "__main__":
    # BTC5m: v3 confirms at 642→648, pyarmor doesn't
    compare_fx_valid("BTC/USDT", "5m", 1000, [
        (642, 645, "confirm 642→645"),
        (642, 648, "confirm 642→648"),
        (642, 650, "extend 638→650 from 642"),
        (638, 642, "set end 638→642"),
        (638, 650, "set end 638→650"),
    ])

    # ETH5m: v3 extends to 253 (eq val), pyarmor doesn't
    compare_fx_valid("ETH/USDT", "5m", 1000, [
        (244, 245, "confirm 244→245"),
        (244, 247, "confirm 244→247"),
        (244, 250, "confirm 244→250"),
        (244, 256, "confirm 244→256"),
        (240, 244, "set end 240→244"),
        (240, 253, "extend 240→253"),
        # Case 2: 346→363 vs 346→357
        (352, 354, "confirm 352→354"),
        (357, 359, "confirm 357→359"),
        (357, 361, "confirm 357→361"),
        (346, 352, "set end 346→352"),
        (346, 357, "extend 346→357"),
        (346, 363, "extend 346→363"),
        # Case 3: 448→461 vs 448→454
        (454, 456, "confirm 454→456"),
        (454, 458, "confirm 454→458"),
        (448, 452, "_bi_fx_valid 448→452"),
        (448, 454, "set end 448→454"),
        (448, 461, "extend 448→461"),
    ])
