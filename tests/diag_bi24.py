# -*- coding: utf-8 -*-
"""
Debug why bi[24] extends to 370 instead of stopping at 347.
Expected: down 315->347, up 347->355, down 355->370
Actual: down 315->370
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from chanlun.cl_open import CL
from chanlun.cl_interface import Config

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "test_data", "BTC_USDT_60m_500.parquet"))
cd = CL("BTC/USDT", "60m")
cd.process_klines(df)

# Find all fractals in the 310-375 area
print("=== Fractals in area 310-375 ===")
for fx in cd.fxs:
    if 310 <= fx.k.index <= 375:
        qj = cd.fx_qj
        qy = cd.fx_qy
        h = fx.high(qj, qy)
        l = fx.low(qj, qy)
        k_idx = fx.k.k_index
        print(f"  FX[{fx.k.index:3d}] {fx.type:4s} val={fx.val:.1f}  high({qy})={h:.1f}  low({qy})={l:.1f}  k_index={k_idx}")

# Now check _bi_fx_valid for ding@315 -> each di candidate
print("\n=== Checking _bi_fx_valid for ding@315 -> di candidates ===")
start_fx = None
for fx in cd.fxs:
    if fx.k.index == 315:
        start_fx = fx
        break

if start_fx:
    for fx in cd.fxs:
        if fx.k.index > 315 and fx.type == "di":
            valid = cd._bi_fx_valid(start_fx, fx)
            cl_gap = fx.k.index - start_fx.k.index
            k_gap = fx.k.k_index - start_fx.k.k_index
            
            qj = cd.fx_qj
            qy = cd.fx_qy
            sh = start_fx.high(qj, qy)
            sl = start_fx.low(qj, qy)
            eh = fx.high(qj, qy)
            el = fx.low(qj, qy)
            
            # Check conditions
            check1 = sl < el  # DOWN: start.low < end.low -> block
            check2 = eh > sh  # DOWN: end.high > start.high -> block
            
            strict_skip = k_gap >= cd.fx_check_k_nums
            
            print(f"  di@{fx.k.index:3d} valid={valid} cl_gap={cl_gap} k_gap={k_gap} strict_skip={strict_skip}")
            print(f"    sh={sh:.1f} sl={sl:.1f} eh={eh:.1f} el={el:.1f}")
            print(f"    check1(sl<el)={check1} check2(eh>sh)={check2}")
            if fx.k.index >= 370:
                break

# Also check pyarmor's fractals
print("\n=== Pyarmor fractals in area 310-375 ===")
from chanlun.cl_pyarmor import CL as CL_P
cd_p = CL_P("BTC/USDT", "60m")
cd_p.process_klines(df)

for fx in cd_p.fxs:
    if 310 <= fx.k.index <= 375:
        qj = "fx_qj_k"
        qy = "fx_qy_three"
        h = fx.high(qj, qy)
        l = fx.low(qj, qy)
        print(f"  FX[{fx.k.index:3d}] {fx.type:4s} val={fx.val:.1f}  high({qy})={h:.1f}  low({qy})={l:.1f}")
