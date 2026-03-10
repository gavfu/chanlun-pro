# -*- coding: utf-8 -*-
"""Detailed trace of _build_bis around ETH5m position 346-363"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor

DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')

df = pd.read_parquet(os.path.join(DATA_DIR, "ETH_USDT_5m_1000.parquet"))
cd_o = CL_Open("ETH/USDT", "5m", config={})
cd_o.process_klines(df)
cd_p = CL_Pyarmor("ETH/USDT", "5m", config={})
cd_p.process_klines(df)

# Show FXs around 340-370 from cl_open
print("=== FXs around 340-370 ===")
for fx in cd_o.fxs:
    if 340 <= fx.k.index <= 370:
        print(f"  {fx.type}[{fx.k.index}] val={fx.val:.2f} "
              f"k_idx={fx.k.k_index} "
              f"high={fx.high(cd_o.fx_qj, cd_o.fx_qy):.2f} "
              f"low={fx.low(cd_o.fx_qj, cd_o.fx_qy):.2f}")

# Show BIs around this area from both engines
print("\n=== CL_Open BIs around 340-380 ===")
for bi in cd_o.bis:
    if 340 <= bi.start.k.index <= 380 or 340 <= bi.end.k.index <= 380:
        is_split = getattr(bi, 'is_split', False)
        print(f"  {bi.type}[{bi.start.k.index}→{bi.end.k.index}] is_split={is_split}")

print("\n=== CL_Pyarmor BIs around 340-380 ===")
for bi in cd_p.bis:
    if 340 <= bi.start.k.index <= 380 or 340 <= bi.end.k.index <= 380:
        is_split = getattr(bi, 'is_split', False)
        print(f"  {bi.type}[{bi.start.k.index}→{bi.end.k.index}] is_split={is_split}")

# Now trace the _build_bis logic step by step for position 346 onward
# The BI starts at ding[346] and should build a down BI
print("\n=== Tracing _build_bis for down BI starting at ding[346] ===")
start_fx_idx = 346
start_fx = next(fx for fx in cd_o.fxs if fx.k.index == start_fx_idx)
print(f"Start: {start_fx.type}[{start_fx.k.index}] val={start_fx.val:.2f}")

# Get subsequent FXs
subsequent = [fx for fx in cd_o.fxs if fx.k.index > start_fx_idx and fx.k.index <= 370]
end_fx = None
for fx in subsequent:
    valid = cd_o._bi_fx_valid(start_fx, fx)
    if end_fx:
        valid_confirm = cd_o._bi_fx_valid(end_fx, fx)
    else:
        valid_confirm = None

    # Check same-type extension
    if end_fx and fx.type == end_fx.type:
        if fx.type == "di":
            ext = fx.val <= end_fx.val
        else:
            ext = fx.val >= end_fx.val
        ext_valid = cd_o._bi_fx_valid(start_fx, fx) if ext else None
    else:
        ext = None
        ext_valid = None

    cl_gap_from_start = fx.k.index - start_fx.k.index
    cl_gap_from_end = (fx.k.index - end_fx.k.index) if end_fx else None

    print(f"  FX: {fx.type}[{fx.k.index}] val={fx.val:.2f} "
          f"cl_gap_start={cl_gap_from_start} "
          f"cl_gap_end={cl_gap_from_end} "
          f"valid_from_start={valid} "
          f"valid_confirm={valid_confirm} "
          f"ext={ext} ext_valid={ext_valid}")
    if end_fx is None and valid and fx.type != start_fx.type:
        print(f"    → Set end_fx = {fx.type}[{fx.k.index}]")
        end_fx = fx
    elif end_fx and fx.type == end_fx.type and ext and ext_valid:
        print(f"    → Extend end_fx to {fx.type}[{fx.k.index}]")
        end_fx = fx
    elif end_fx and fx.type == start_fx.type and valid_confirm:
        print(f"    → CONFIRM BI: {start_fx.type}[{start_fx.k.index}]→{end_fx.type}[{end_fx.k.index}]")
        break

# Check what pyarmor's _bi_check_bi_fx_ok says for the same pairs
print("\n=== Pyarmor _bi_check_bi_fx_ok comparison ===")
p_fxs = {fx.k.index: fx for fx in cd_p.fxs}
p_start = p_fxs.get(346)
key_pairs = [
    (346, 349), (346, 352), (346, 357), (346, 360),
    (349, 350), (352, 354), (357, 359), (357, 361),
    (360, 361), (346, 363),
]
for a, b in key_pairs:
    fa, fb = p_fxs.get(a), p_fxs.get(b)
    if fa and fb:
        try:
            v_p = cd_p._bi_check_bi_fx_ok(fa, fb)
        except:
            v_p = "ERROR"
        v_o = cd_o._bi_fx_valid(
            next(fx for fx in cd_o.fxs if fx.k.index == a),
            next(fx for fx in cd_o.fxs if fx.k.index == b))
        match = "✅" if v_p == v_o else "❌"
        print(f"  ({a},{b}): open={v_o} pyarmor={v_p} {match}")
