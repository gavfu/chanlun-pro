# -*- coding: utf-8 -*-
"""Systematic analysis of ALL initial candidate validations to find the exact strict check formula"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "test_data", "BTC_USDT_60m_500.parquet"))

cd_p = CL_Pyarmor("BTC/USDT", "60m")
cd_p.process_klines(df)

cd_o = CL_Open("BTC/USDT", "60m")
cd_o.process_klines(df)

o_fxs = cd_o.get_fxs()
p_bis = cd_p.get_bis()
qj = cd_o.fx_qj
qy = cd_o.fx_qy

# Build a set of pyarmor stroke (start_ck, end_ck) pairs
p_stroke_pairs = set()
for bi in p_bis:
    p_stroke_pairs.add((bi.start.k.index, bi.end.k.index))

# Also build the set of start/end CKs used by pyarmor
p_used_ck = set()
for bi in p_bis:
    p_used_ck.add(bi.start.k.index)
    p_used_ck.add(bi.end.k.index)

# For each pyarmor stroke with k_gap < 13, show the fractal range details
print("=== Pyarmor strokes with k_gap < 13 (where strict check applies) ===")
print(f"{'bi':>4s} {'type':>5s} {'s_ck':>5s}{'->':>2s}{'e_ck':>5s} {'k_gap':>5s} | start: val  high3   low3   | end: val   high3   low3   | s.h>e.h s.l<e.l")
for i, bi in enumerate(p_bis):
    k_gap = bi.end.k.k_index - bi.start.k.k_index
    if k_gap >= 13:
        continue
    s = bi.start
    e = bi.end
    sh = s.high(qj, qy)
    sl = s.low(qj, qy)
    eh = e.high(qj, qy)
    el = e.low(qj, qy)
    
    # For up stroke (di→ding): strict check is s.high > e.high
    # For down stroke (ding→di): strict check is s.low < e.low
    if bi.type == "up":
        check1 = f"s.h>e.h:{sh > eh}"
        check2 = f"s.l<e.l:{sl < el}"
    else:
        check1 = f"s.l<e.l:{sl < el}"
        check2 = f"s.h>e.h:{sh > eh}"
    
    print(f"bi[{i:2d}] {bi.type:5s} {s.k.index:5d}->{e.k.index:5d} k={k_gap:3d} | "
          f"sv={s.val:8.1f} sh={sh:8.1f} sl={sl:8.1f} | "
          f"ev={e.val:8.1f} eh={eh:8.1f} el={el:8.1f} | {check1:15s} {check2:15s}")

# Now check ALL possible candidate pairs in the fxs that are NOT used by pyarmor
# (i.e., pairs that should be REJECTED)
print()
print("=== Candidate pairs that SHOULD be REJECTED (not used by pyarmor) ===")
print("Looking for opposite-type pairs with cl_gap>=4 and k_gap in [4,13) where one is a pyarmor stroke start/end")

# Trace through the algorithm to find which candidates are formed but should be rejected
# Actually, let me just find pairs where start is a pyarmor end_fx (meaning it could be a start of next attempt)
# and end is a valid fractal that pyarmor doesn't use
for i, bi in enumerate(p_bis):
    end_fx = bi.end  # This becomes start_fx for next stroke
    # Find the pyarmor next stroke's end
    if i + 1 < len(p_bis):
        next_end = p_bis[i + 1].end
    else:
        continue
    
    # Find all opposite-type fractals between end_fx and next_end that are NOT next_end
    for fx in o_fxs:
        if fx.k.index <= end_fx.k.index:
            continue
        if fx.k.index >= next_end.k.index:
            break
        if fx.type == end_fx.type:
            continue  # same type, not a candidate
        
        cl_gap = fx.k.index - end_fx.k.index
        k_gap = fx.k.k_index - end_fx.k.k_index
        if cl_gap < 4 or k_gap < 4:
            continue
        if k_gap >= 13:
            continue  # strict not applied
        
        sh = end_fx.high(qj, qy)
        sl = end_fx.low(qj, qy)
        fh = fx.high(qj, qy)
        fl = fx.low(qj, qy)
        
        if end_fx.type == "di":  # up stroke: di → ding
            strict_block = sh > fh
            check_str = f"s.h({sh:.1f})>e.h({fh:.1f})={strict_block}"
        else:  # down stroke: ding → di
            strict_block = sl < fl
            check_str = f"s.l({sl:.1f})<e.l({fl:.1f})={strict_block}"
        
        # This candidate should NOT form a stroke (pyarmor skips it)
        print(f"  REJECT bi[{i}]_end->{end_fx.type} ck={end_fx.k.index} → {fx.type} ck={fx.k.index} "
              f"(cl={cl_gap} k={k_gap}) | {check_str}")
        # Check if strict already blocks it
        if strict_block:
            print(f"    → Already blocked by strict check")
        else:
            print(f"    → NOT blocked by strict check! Need additional mechanism!")
            # List all details
            print(f"      start: val={end_fx.val:.1f} h3={sh:.1f} l3={sl:.1f} k.h={end_fx.k.h:.1f} k.l={end_fx.k.l:.1f}")
            print(f"      end:   val={fx.val:.1f} h3={fh:.1f} l3={fl:.1f} k.h={fx.k.h:.1f} k.l={fx.k.l:.1f}")
