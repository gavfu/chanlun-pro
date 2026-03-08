# -*- coding: utf-8 -*-
"""Detail investigation of bi[21] divergence: why ding 267 should NOT be confirmed"""
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
qj = cd_o.fx_qj
qy = cd_o.fx_qy

# For each confirmed stroke in pyarmor, analyze the confirmation pair
p_bis = cd_p.get_bis()
p_fxs = cd_p.get_fxs()

print("=== Analyzing confirmation patterns for all pyarmor strokes ===")
print(f"fx_check_k_nums = {cd_o.fx_check_k_nums}")
print()

# For each stroke, find the confirmation fractal (the first fractal after end_fx that creates next valid stroke)
for i, bi in enumerate(p_bis):
    if i >= len(p_bis) - 1:
        break
    next_bi = p_bis[i + 1]
    
    # The confirmation fractal is the end of the next stroke (which starts at current stroke's end)
    # Actually, the confirmation is: end_fx of current stroke → end_fx of next stroke
    # Which is: bi.end → next_bi.end
    # But actually, confirmation = next stroke's end_fx forming valid stroke from current end_fx
    # The confirming fractal is next_bi.end
    
    confirm_fx = next_bi.end
    end_fx = bi.end
    start_fx = bi.start
    
    # Compute gaps for the confirmation pair
    cl_gap_confirm = confirm_fx.k.index - end_fx.k.index
    k_gap_confirm = confirm_fx.k.k_index - end_fx.k.k_index
    
    # Compute gaps for the stroke itself
    cl_gap_stroke = end_fx.k.index - start_fx.k.index
    k_gap_stroke = end_fx.k.k_index - start_fx.k.k_index
    
    strict_applied = k_gap_confirm < cd_o.fx_check_k_nums
    
    # Check CGD: between end_fx and confirm_fx, is there a same-type fractal as confirm_fx that's more extreme?
    cgd_block = False
    # Find end_fx index in our fxs
    end_idx_in_fxs = None
    confirm_idx_in_fxs = None
    for idx, fx in enumerate(o_fxs):
        if fx.k.index == end_fx.k.index:
            end_idx_in_fxs = idx
        if fx.k.index == confirm_fx.k.index:
            confirm_idx_in_fxs = idx
    
    if end_idx_in_fxs is not None and confirm_idx_in_fxs is not None:
        for j in range(end_idx_in_fxs + 1, confirm_idx_in_fxs):
            mid_fx = o_fxs[j]
            if mid_fx.type == confirm_fx.type:
                if confirm_fx.type == "ding" and mid_fx.val > confirm_fx.val:
                    cgd_block = True
                    break
                elif confirm_fx.type == "di" and mid_fx.val < confirm_fx.val:
                    cgd_block = True
                    break
    
    print(f"bi[{i:2d}] {bi.type:4s} {start_fx.k.index:3d}->{end_fx.k.index:3d} -- confirmed by {confirm_fx.type} ck={confirm_fx.k.index} "
          f"| stroke_k_gap={k_gap_stroke:3d} confirm_k_gap={k_gap_confirm:3d} "
          f"strict={'Y' if strict_applied else 'N'} cgd_confirm={'BLOCK' if cgd_block else 'ok'}")

# Now trace bi[21] in detail
print()
print("="*60)
print("=== bi[21] detailed trace ===")
print("="*60)

# In our code, bi[21] is ding 267 confirmed by di 274
# In pyarmor, bi[21] is ding 276
# Let's check what happens if we DON'T set ding 267 as end_fx

# What if _bi_fx_valid(di 262, ding 267) should return FALSE?
fx_262 = [fx for fx in o_fxs if fx.k.index == 262][0]
fx_265 = [fx for fx in o_fxs if fx.k.index == 265][0]
fx_267 = [fx for fx in o_fxs if fx.k.index == 267][0]

h_262 = fx_262.high(qj, qy)
l_262 = fx_262.low(qj, qy)
h_265 = fx_265.high(qj, qy) 
l_265 = fx_265.low(qj, qy)
h_267 = fx_267.high(qj, qy)
l_267 = fx_267.low(qj, qy)

print(f"\ndi ck=262: val={fx_262.val:.1f} high3={h_262:.1f} low3={l_262:.1f}")
print(f"ding ck=265: val={fx_265.val:.1f} high3={h_265:.1f} low3={l_265:.1f}")
print(f"ding ck=267: val={fx_267.val:.1f} high3={h_267:.1f} low3={l_267:.1f}")

print(f"\nStrict check di→ding:")
print(f"  _bi_fx_valid(di 262, ding 265): cl_gap={265-262} k_gap={fx_265.k.k_index-fx_262.k.k_index}")
print(f"    strict: h_262({h_262:.1f}) > h_265({h_265:.1f})? {h_262 > h_265}")
print(f"  _bi_fx_valid(di 262, ding 267): cl_gap={267-262} k_gap={fx_267.k.k_index-fx_262.k.k_index}")
print(f"    strict: h_262({h_262:.1f}) > h_267({h_267:.1f})? {h_262 > h_267}")

# What if the strict check also checks the OTHER direction?
# For di→ding: also check l_start > l_end? (meaning di's low must be lower than ding's low)
print(f"\n  Alternative strict check:")
print(f"    l_262({l_262:.1f}) > l_267({l_267:.1f})? {l_262 > l_267}")
print(f"    h_262({h_262:.1f}) > h_267({h_267:.1f})? {h_262 > h_267}")
print(f"    di 262 range: [{l_262:.1f}, {h_262:.1f}]")
print(f"    ding 267 range: [{l_267:.1f}, {h_267:.1f}]")
print(f"    Overlap? ding_low < di_high? {l_267 < h_262} (l_267={l_267:.1f} < h_262={h_262:.1f})")

# Maybe the check is: ding's low3 must be > di's high3 (no overlap)?
# Or: di's high3 must be < ding's low3?
# Let's check various formulations:
print(f"\n  Possible strict formulations for di→ding:")
print(f"    1. h_start > h_end:  {h_262:.1f} > {h_267:.1f} = {h_262 > h_267}")
print(f"    2. l_start > l_end:  {l_262:.1f} > {l_267:.1f} = {l_262 > l_267}")
print(f"    3. h_start > l_end:  {h_262:.1f} > {l_267:.1f} = {h_262 > l_267}")
print(f"    4. l_start < h_end:  {l_262:.1f} < {h_267:.1f} = {l_262 < h_267}")
print(f"    5. val_start > val_end: {fx_262.val:.1f} > {fx_267.val:.1f} = {fx_262.val > fx_267.val}")

# Let's compare with bi[0] confirmation to see which formulation works
print()
print("="*60)
print("=== Compare with successful strokes ===")
print("="*60)

# bi[0] down ding 1 → di 11, should be VALID
# The INITIAL candidate: ding 1 → first valid di
fx_1 = [fx for fx in o_fxs if fx.k.index == 1][0]
fx_11 = [fx for fx in o_fxs if fx.k.index == 11][0]
h_1 = fx_1.high(qj, qy)
l_1 = fx_1.low(qj, qy)
h_11 = fx_11.high(qj, qy)
l_11 = fx_11.low(qj, qy)

print(f"\nbi[0] ding 1 → di 11 (SHOULD be valid as initial candidate)")
print(f"  ding 1: val={fx_1.val:.1f} high3={h_1:.1f} low3={l_1:.1f}")
print(f"  di 11: val={fx_11.val:.1f} high3={h_11:.1f} low3={l_11:.1f}")
print(f"  cl_gap={11-1} k_gap={fx_11.k.k_index - fx_1.k.k_index}")
print(f"  ding→di strict: l_1({l_1:.1f}) < l_11({l_11:.1f})? {l_1 < l_11}")

# bi[3] up di 25 → ding 29 (VALID)
fx_25 = [fx for fx in o_fxs if fx.k.index == 25][0]
fx_29 = [fx for fx in o_fxs if fx.k.index == 29][0]
h_25 = fx_25.high(qj, qy)
l_25 = fx_25.low(qj, qy)
h_29 = fx_29.high(qj, qy)
l_29 = fx_29.low(qj, qy)

print(f"\nbi[3] di 25 → ding 29 (SHOULD be valid)")
print(f"  di 25: val={fx_25.val:.1f} high3={h_25:.1f} low3={l_25:.1f}")
print(f"  ding 29: val={fx_29.val:.1f} high3={h_29:.1f} low3={l_29:.1f}")
print(f"  cl_gap={29-25} k_gap={fx_29.k.k_index - fx_25.k.k_index}")
print(f"  di→ding strict: h_25({h_25:.1f}) > h_29({h_29:.1f})? {h_25 > h_29}")

# Now check the FAILING case: di 262 → ding 267 (should NOT be valid candidate, or not confirmable)
print(f"\nbi[21] di 262 → ding 267 (should NOT be used by pyarmor)")
print(f"  di 262: val={fx_262.val:.1f} high3={h_262:.1f} low3={l_262:.1f}")
print(f"  ding 267: val={fx_267.val:.1f} high3={h_267:.1f} low3={l_267:.1f}")
print(f"  cl_gap={267-262} k_gap={fx_267.k.k_index - fx_262.k.k_index}")
print(f"  di→ding strict: h_262({h_262:.1f}) > h_267({h_267:.1f})? {h_262 > h_267}")
print(f"  ALTERNATIVE: l_262({l_262:.1f}) < l_267({l_267:.1f})? {l_262 < l_267}")
print(f"  Both together (block if h_start > h_end OR l_start < l_end):")
print(f"    h_262 > h_267: {h_262 > h_267}")
print(f"    l_262 < l_267: {l_262 < l_267}")
