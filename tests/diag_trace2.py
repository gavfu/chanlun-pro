# -*- coding: utf-8 -*-
"""
Trace the exact algorithm steps for our code vs what pyarmor produces,
focusing on the bi[20]-bi[21] transition area (around ck=262-280).

Key insight: maybe the issue isn't the strict check formula but the algorithm
flow - specifically how CGD interacts with candidate selection/confirmation.
"""
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
o_bis = cd_o.get_bis()
p_bis = cd_p.get_bis()

# Our bi[20] ends at ck 262 (same as pyarmor)
# Our bi[21] goes up 262->267, pyarmor goes up 262->276

# Let's look at ALL fractals from ck 260 to 290
print("=== Fractals ck 255-295 ===")
for fx in o_fxs:
    if 255 <= fx.k.index <= 295:
        print(f"  ck={fx.k.index:3d} kk={fx.k.k_index:3d} {fx.type:4s} val={fx.val:8.1f} "
              f"h3={fx.high('fx_qj_k','fx_qy_three'):8.1f} l3={fx.low('fx_qj_k','fx_qy_three'):8.1f}")

print()
print("=== Our strokes bi[19]-bi[26] ===")
for i in range(19, min(27, len(o_bis))):
    bi = o_bis[i]
    print(f"  bi[{i:2d}] {bi.type:5s} {bi.start.k.index:3d}->{bi.end.k.index:3d} "
          f"k_gap={bi.end.k.k_index - bi.start.k.k_index}")

print()
print("=== Pyarmor strokes bi[19]-bi[26] ===")
for i in range(19, min(27, len(p_bis))):
    bi = p_bis[i]
    print(f"  bi[{i:2d}] {bi.type:5s} {bi.start.k.index:3d}->{bi.end.k.index:3d} "
          f"k_gap={bi.end.k.k_index - bi.start.k.k_index}")

# Now trace: When we're at di 262 (start), what happens in the algorithm?
# Our algorithm: start=di 262, no end_fx
# Then we scan for opposite-type (ding) fractals
# First ding encountered after 262: let's find it
print()
print("=== Trace from di 262 ===")
di_262 = None
for fx in o_fxs:
    if fx.k.index == 262 and fx.type == 'di':
        di_262 = fx
        break

print(f"start_fx: ck={di_262.k.index} type={di_262.type} val={di_262.val:.1f}")

# Find all ding fractals after 262
print(f"\nAll ding fractals after ck=262:")
for fx in o_fxs:
    if fx.k.index > 262 and fx.type == 'ding' and fx.k.index < 300:
        cl_gap = fx.k.index - 262
        k_gap = fx.k.k_index - di_262.k.k_index
        valid_gap = cl_gap >= 4 and k_gap >= 4
        strict_applies = k_gap < 13
        
        # Current strict check: s.high3 > e.high3 → block
        sh = di_262.high('fx_qj_k', 'fx_qy_three')
        eh = fx.high('fx_qj_k', 'fx_qy_three')
        strict_block = sh > eh if strict_applies else False
        
        print(f"  ding ck={fx.k.index:3d} cl_gap={cl_gap:2d} k_gap={k_gap:2d} "
              f"val={fx.val:8.1f} valid_gap={valid_gap} strict={strict_applies} "
              f"sh={sh:.1f}>eh={eh:.1f}={strict_block}")

# Now let's understand: what does pyarmor do at di 262?
# Pyarmor's next stroke is up 262->276
# ding 276 is at k_gap=?
print()
for fx in o_fxs:
    if fx.k.index == 276 and fx.type == 'ding':
        k_gap = fx.k.k_index - di_262.k.k_index
        sh = di_262.high('fx_qj_k', 'fx_qy_three')
        eh = fx.high('fx_qj_k', 'fx_qy_three')
        print(f"ding 276: k_gap={k_gap} sh={sh:.1f} eh={eh:.1f}")
        
# Key question: why does pyarmor skip ding 267 and go to ding 276?
# Possible reasons:
# 1. Strict check blocks ding 267 (but our formula says it passes)
# 2. CGD check blocks ding 267 (but we already checked this)
# 3. The algorithm never considers ding 267 as end_fx (some other logic prevents it)
# 4. The pyarmor algorithm handles "extension" differently

# Let's check: between di 262 and ding 267, is there a di fractal that's lower?
# That would trigger the "update start_fx" logic
print()
print("=== All fractals between ck 262 and 276 ===")
for fx in o_fxs:
    if 262 < fx.k.index < 276:
        print(f"  ck={fx.k.index:3d} {fx.type:4s} val={fx.val:8.1f}")

# Check: is there a di between 262 and 267 that's lower?
# That would mean we reject di 262 and use the new lower di as start
print()
print("=== Is there a lower di between 262 and 267? ===")
for fx in o_fxs:
    if 262 < fx.k.index <= 267 and fx.type == 'di':
        print(f"  di ck={fx.k.index} val={fx.val:.1f} (262 val={di_262.val:.1f}, lower={fx.val < di_262.val})")

# Now let's think about the confirmation logic differently
# After we set ding 267 as end_fx, we continue scanning
# If CGD check blocks the CONFIRMATION (not end_fx setting), then:
# - end_fx = ding 267
# - Then we encounter next opposite (di) at some point
# - We check: between end_fx(267) and confirm_fx(di), is there a HIGHER ding?
# - If yes, we reject the confirmation and extend end_fx to ding 276
print()
print("=== After ding 267, what ding fractals exist before the next di? ===")
next_di_after_267 = None
for fx in o_fxs:
    if fx.k.index > 267 and fx.type == 'di':
        next_di_after_267 = fx
        break

if next_di_after_267:
    print(f"Next di after ding 267: ck={next_di_after_267.k.index} val={next_di_after_267.val:.1f}")
    
    # Between ding 267 and this di, are there higher ding fractals?
    for fx in o_fxs:
        if 267 < fx.k.index < next_di_after_267.k.index and fx.type == 'ding':
            print(f"  ding ck={fx.k.index} val={fx.val:.1f} (higher than 267's {67014.9}? {fx.val > 67014.9})")

    # Also: between ding 267 and next_di, what are ALL the fractals?
    print()
    print("=== All fractals between ding 267 and next di ===")
    for fx in o_fxs:
        if 267 <= fx.k.index <= next_di_after_267.k.index:
            print(f"  ck={fx.k.index:3d} {fx.type:4s} val={fx.val:8.1f}")

# Now check the key CGD confirmation scenario:
# When we have end_fx=ding 267 and see confirm_fx=di X:
# CGD check: between end_fx(267) and confirm_fx(di X), 
# is there a SAME-TYPE-as-confirm (i.e., another di) that is MORE extreme (lower)?
# If yes → reject confirmation, reset end_fx
# 
# But this is OUR CGD check. The PYARMOR CGD check might be different:
# Maybe CGD check checks for a ding HIGHER than end_fx between end_fx and confirm_fx?
# This would extend the end_fx rather than reject confirmation

print()
print("=== Check: extension vs confirmation CGD ===")
# Extension CGD: when we have end_fx=ding 267, then encounter a same-type ding:
# If it's higher, we should extend to it
# Let's check: are there any ding fractals from 267 to 276 (pyarmor's target)?
print("Ding fractals from 268 to 276:")
for fx in o_fxs:
    if 267 < fx.k.index <= 276 and fx.type == 'ding':
        k_gap_from_start = fx.k.k_index - di_262.k.k_index
        valid = (fx.k.index - 262) >= 4 and k_gap_from_start >= 4
        print(f"  ding ck={fx.k.index} val={fx.val:.1f} k_gap_from_start={k_gap_from_start} valid={valid}")
