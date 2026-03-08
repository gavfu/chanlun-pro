# -*- coding: utf-8 -*-
"""Find the fractal divergence point between open and pyarmor"""
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
p_fxs = cd_p.get_fxs()
print(f"open fxs: {len(o_fxs)}, pyarmor fxs: {len(p_fxs)}")

# Find divergence
for i in range(max(len(o_fxs), len(p_fxs))):
    of = o_fxs[i] if i < len(o_fxs) else None
    pf = p_fxs[i] if i < len(p_fxs) else None
    if of and pf:
        if of.k.index != pf.k.index or of.type != pf.type:
            print(f"Diverge at [{i}]:")
            for j in range(max(0, i-2), min(i+8, max(len(o_fxs), len(p_fxs)))):
                oj = o_fxs[j] if j < len(o_fxs) else None
                pj = p_fxs[j] if j < len(p_fxs) else None
                o_str = f"{oj.type:4s} ck={oj.k.index:3d} val={oj.val:.1f}" if oj else "N/A"
                p_str = f"{pj.type:4s} ck={pj.k.index:3d} val={pj.val:.1f}" if pj else "N/A"
                match = "✅" if (oj and pj and oj.k.index == pj.k.index and oj.type == pj.type) else "❌"
                print(f"  [{j:3d}] o={o_str} | p={p_str} {match}")
            break
    elif pf:
        print(f"Extra pyarmor at [{i}]: {pf.type} ck={pf.k.index}")
    elif of:
        print(f"Extra open at [{i}]: {of.type} ck={of.k.index}")

# Also check: does disabling allow_bi_fx_strict affect fractal count? No, it shouldn't.
# The issue is in fractal construction, not stroke construction.

# Check bi_fx_cgd effect
print(f"\nbi_fx_cgd config: {cd_o.bi_fx_cgd}")
print(f"allow_bi_fx_strict config: {cd_o.allow_bi_fx_strict}")

# Now test: which stroke is confirmed by the confirmation logic?
# Trace from ding ck=113
print("\n--- Tracing from ding ck=113 (bi[12]) ---")
print("After bi[11] up 96->113, we start from ding ck=113")
print("Looking for di to form down stroke:")
for fx in o_fxs:
    if fx.k.index > 113 and fx.k.index <= 160:
        print(f"  ck={fx.k.index:3d} {fx.type:4s} val={fx.val:.1f} k.h={fx.k.h:.1f} k.l={fx.k.l:.1f}")
        if fx.type == "di":
            valid_from_113 = cd_o._bi_fx_valid(
                [f for f in o_fxs if f.k.index == 113][0], fx
            )
            print(f"    valid from ck=113: {valid_from_113}")
