# -*- coding: utf-8 -*-
"""Check all possible high/low values for key fractals to find the right combination"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from chanlun.cl_open import CL as CL_Open

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "test_data", "BTC_USDT_60m_500.parquet"))
cd_o = CL_Open("BTC/USDT", "60m")
cd_o.process_klines(df)

fxs = cd_o.get_fxs()

# For key fractals, show ALL possible high/low values
key_cks = [262, 267, 276, 347, 355, 25, 29, 65, 74]
fx_map = {fx.k.index: fx for fx in fxs}

qj_options = ['fx_qj_k', 'fx_qj_ck']
qy_options = ['fx_qy_middle', 'fx_qy_three']

print("=== All high/low values for key fractals ===")
for ck in key_cks:
    if ck not in fx_map:
        continue
    fx = fx_map[ck]
    print(f"\nck={ck:3d} {fx.type:4s} val={fx.val:8.1f}  k.h={fx.k.h:8.1f} k.l={fx.k.l:8.1f}")
    
    # Also show the raw CLKline elements
    if hasattr(fx, 'elements') and fx.elements:
        for j, elem in enumerate(fx.elements):
            print(f"  elem[{j}]: index={elem.index} h={elem.h:8.1f} l={elem.l:8.1f} "
                  f"k_index={elem.k_index}")
    # Show the 3 CLKlines that make up this fractal
    print(f"  klines: ", end="")
    for ck in fx.klines:
        print(f"ck[{ck.index}] h={ck.h:.1f} l={ck.l:.1f}  ", end="")
    print()
    
    for qj in qj_options:
        for qy in qy_options:
            h = fx.high(qj, qy)
            l = fx.low(qj, qy)
            print(f"  {qj:10s} {qy:14s}: h={h:8.1f} l={l:8.1f}")

# Now for the critical pair: di 262 -> ding 267 (should be REJECTED)
# And valid pair: di 347 -> ding 355 (should be ACCEPTED but h3>h3 rejects it)
print("\n\n=== Critical comparisons ===")

# di 262 -> ding 267 (INVALID, pyarmor rejects)
s = fx_map[262]
e = fx_map[267]
print(f"\ndi 262 -> ding 267 (should REJECT):")
for qj in qj_options:
    for qy in qy_options:
        sh = s.high(qj, qy)
        sl = s.low(qj, qy)
        eh = e.high(qj, qy)
        el = e.low(qj, qy)
        # For up stroke: check s.high > e.high → block
        check1 = sh > eh
        # Also check s.low > e.low → block  
        check2 = sl > el
        # Also check e.low > s.high → block (no overlap)
        check3 = el > sh
        # s.h > e.l → block (start contains end)
        check4 = sh > el
        print(f"  {qj:10s} {qy:14s}: sh={sh:8.1f} sl={sl:8.1f} eh={eh:8.1f} el={el:8.1f} | "
              f"sh>eh={check1} sl>el={check2} sh>el={check4}")

# up 347 -> 355 (VALID, pyarmor accepts)
s = fx_map[347]
e = fx_map[355]
print(f"\ndi 347 -> ding 355 (should ACCEPT):")
for qj in qj_options:
    for qy in qy_options:
        sh = s.high(qj, qy)
        sl = s.low(qj, qy)
        eh = e.high(qj, qy)
        el = e.low(qj, qy)
        check1 = sh > eh
        check2 = sl > el
        check4 = sh > el
        print(f"  {qj:10s} {qy:14s}: sh={sh:8.1f} sl={sl:8.1f} eh={eh:8.1f} el={el:8.1f} | "
              f"sh>eh={check1} sl>el={check2} sh>el={check4}")

# up 25 -> 29 (VALID, pyarmor accepts, k_gap=5)
s = fx_map[25]
e = fx_map[29]
print(f"\ndi 25 -> ding 29 (should ACCEPT):")
for qj in qj_options:
    for qy in qy_options:
        sh = s.high(qj, qy)
        sl = s.low(qj, qy)
        eh = e.high(qj, qy)
        el = e.low(qj, qy)
        check1 = sh > eh
        check2 = sl > el
        check4 = sh > el
        print(f"  {qj:10s} {qy:14s}: sh={sh:8.1f} sl={sl:8.1f} eh={eh:8.1f} el={el:8.1f} | "
              f"sh>eh={check1} sl>el={check2} sh>el={check4}")

# up 65 -> 74 (VALID, pyarmor accepts, k_gap=12)
s = fx_map[65]
e = fx_map[74]
print(f"\ndi 65 -> ding 74 (should ACCEPT):")
for qj in qj_options:
    for qy in qy_options:
        sh = s.high(qj, qy)
        sl = s.low(qj, qy)
        eh = e.high(qj, qy)
        el = e.low(qj, qy)
        check1 = sh > eh
        check2 = sl > el
        check4 = sh > el
        print(f"  {qj:10s} {qy:14s}: sh={sh:8.1f} sl={sl:8.1f} eh={eh:8.1f} el={el:8.1f} | "
              f"sh>eh={check1} sl>el={check2} sh>el={check4}")
