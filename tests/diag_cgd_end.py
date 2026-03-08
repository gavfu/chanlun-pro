# -*- coding: utf-8 -*-
"""
Test theory: strict check = CGD check on END_FX setting, not just confirmation.
When allow_bi_fx_strict=True and k_gap < fx_check_k_nums:
  For up stroke (di->ding): reject ding if there exists a HIGHER ding between start and this ding
  For down stroke (ding->di): reject di if there exists a LOWER di between start and this di

This is essentially "strict" = require optimal end point (no 次高低 allowed) for short strokes.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from chanlun.cl_pyarmor import CL as CL_Pyarmor
from chanlun.cl_open import CL as CL_Open

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "test_data", "BTC_USDT_60m_500.parquet"))

cd_p = CL_Pyarmor("BTC/USDT", "60m")
cd_p.process_klines(df)
cd_o = CL_Open("BTC/USDT", "60m")
cd_o.process_klines(df)

cd_off = CL_Pyarmor("BTC/USDT", "60m", config={"allow_bi_fx_strict": 0})
cd_off.process_klines(df)

p_bis = cd_p.get_bis()
off_bis = cd_off.get_bis()
o_fxs = cd_o.get_fxs()

# Find strokes in OFF but not ON (BLOCKED by strict)
on_set = {(bi.start.k.index, bi.end.k.index) for bi in p_bis}
off_set = {(bi.start.k.index, bi.end.k.index) for bi in off_bis}

print("=== BLOCKED strokes: check if between start and end there exists a more extreme same-type FX ===\n")

for bi in off_bis:
    key = (bi.start.k.index, bi.end.k.index)
    if key not in on_set:
        # This stroke is BLOCKED by strict
        s = bi.start
        e = bi.end
        k_gap = e.k.k_index - s.k.k_index
        
        # Check: between start and end, is there a same-type FX as end_fx 
        # that has a more extreme value?
        more_extreme = []
        for fx in o_fxs:
            if fx.k.index <= s.k.index or fx.k.index >= e.k.index:
                continue
            if fx.type != e.type:
                continue
            # Same type as end_fx
            if e.type == "ding" and fx.val > e.val:
                more_extreme.append(fx)
            elif e.type == "di" and fx.val < e.val:
                more_extreme.append(fx)
        
        has_more_extreme = len(more_extreme) > 0
        
        # Also check: same type as end_fx that is BETWEEN start and end (inclusive of fx_check range)
        print(f"  BLOCKED: {bi.type:5s} {s.k.index:3d}->{e.k.index:3d} k={k_gap:3d} "
              f"end_val={e.val:8.1f} has_more_extreme={has_more_extreme}", end="")
        if more_extreme:
            for fx in more_extreme:
                print(f" [{fx.type}@{fx.k.index} val={fx.val:.1f}]", end="")
        print()
        
        # Also check: looking between start and end, is there a fractal 
        # of the OPPOSITE type (same type as start) that's more extreme than end?
        # For up (start=di, end=ding): is there a ding BETWEEN with h > end.val? 
        # No wait, that's what we checked above.
        
        # What about: start's FX range (3 CLKlines) has a high > end.val?
        # For up: start di's high3 > end ding's val?
        sh3 = s.high('fx_qj_k', 'fx_qy_three')
        if bi.type == "up":
            s_contains = sh3 > e.val
        else:
            sl3 = s.low('fx_qj_k', 'fx_qy_three')
            s_contains = sl3 < e.val
        print(f"         start_h3={sh3:.1f} > end_val={e.val:.1f} = {s_contains if bi.type == 'up' else ''}")

print("\n=== ACCEPTED strokes (strict ON, k_gap < 13): verify no more extreme ===\n")
for bi in p_bis:
    k_gap = bi.end.k.k_index - bi.start.k.k_index
    if k_gap >= 13:
        continue
    s = bi.start
    e = bi.end
    
    more_extreme = []
    for fx in o_fxs:
        if fx.k.index <= s.k.index or fx.k.index >= e.k.index:
            continue
        if fx.type != e.type:
            continue
        if e.type == "ding" and fx.val > e.val:
            more_extreme.append(fx)
        elif e.type == "di" and fx.val < e.val:
            more_extreme.append(fx)
    
    has_more_extreme = len(more_extreme) > 0
    print(f"  ACCEPTED: {bi.type:5s} {s.k.index:3d}->{e.k.index:3d} k={k_gap:3d} "
          f"end_val={e.val:8.1f} has_more_extreme={has_more_extreme}", end="")
    if more_extreme:
        for fx in more_extreme:
            print(f" [{fx.type}@{fx.k.index} val={fx.val:.1f}]", end="")
    print()
