"""Trace BTC60 BI construction near bi[15] divergence."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

CL_CONFIG = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
}

df = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cd_o = CL_O("test", "test", config=CL_CONFIG)
cd_o.process_klines(df)
cd_p = CL_P("test", "test", config=CL_CONFIG)
cd_p.process_klines(df)

bis_o = cd_o.get_bis()
bis_p = cd_p.get_bis()
fxs_o = cd_o.get_fxs()

# Show BIs 13-19 for context
print("=== Open BIs 13-19 ===")
for bi in bis_o[13:20]:
    print(f"  bi[{bi.index:>2}] {bi.type:>4} k={bi.start.k.k_index}→{bi.end.k.k_index}")

print("\n=== Pyarmor BIs 13-19 ===")
for bi in bis_p[13:20]:
    print(f"  bi[{bi.index:>2}] {bi.type:>4} k={bi.start.k.k_index}→{bi.end.k.k_index}")

# Open bi[14] ends at some di, then bi[15] starts. bi[15] is up, ending at ding@308 (open) vs ding@304 (pyarmor)
# Show FXes in range 280-330 (around bi[15])
print("\n=== FXes in range k_index 280-330 ===")
for fx in fxs_o:
    if 280 <= fx.k.k_index <= 330:
        print(f"  fx {fx.type:>4} k_index={fx.k.k_index} val={fx.val:.2f} k.index={fx.k.index}")

# Check the specific FXes
# bi[14] open ends at some di, then bi[15] goes up to ding@308
# pyarmor bi[15] goes up to ding@304
# Let's find both ding FXes
print("\n=== Checking ding@304 and ding@308 ===")
fx_ding_304 = None
fx_ding_308 = None
for fx in fxs_o:
    if fx.k.k_index == 304 and fx.type == "ding":
        fx_ding_304 = fx
    if fx.k.k_index == 308 and fx.type == "ding":
        fx_ding_308 = fx

# What is the start_fx for bi[15]?
print(f"  Open bi[14]: {bis_o[14].type} k={bis_o[14].start.k.k_index}→{bis_o[14].end.k.k_index}")
start_fx = bis_o[14].end  # bi[15]'s start = bi[14]'s end
print(f"  bi[15] start_fx: {start_fx.type} k_index={start_fx.k.k_index} val={start_fx.val:.2f}")

if fx_ding_304 and fx_ding_308:
    qj = cd_o.fx_qj
    qy = cd_o.fx_qy
    
    print(f"  ding@304: val={fx_ding_304.val:.2f} high={fx_ding_304.high(qj,qy):.2f} low={fx_ding_304.low(qj,qy):.2f}")
    print(f"  ding@308: val={fx_ding_308.val:.2f} high={fx_ding_308.high(qj,qy):.2f} low={fx_ding_308.low(qj,qy):.2f}")
    
    # Check _bi_fx_valid for both
    v304 = cd_o._bi_fx_valid(start_fx, fx_ding_304)
    v308 = cd_o._bi_fx_valid(start_fx, fx_ding_308)
    print(f"  _bi_fx_valid(start, ding@304) = {v304}")
    print(f"  _bi_fx_valid(start, ding@308) = {v308}")
    
    # If both are valid, trace why the algorithm picks 308 over 304
    # The algorithm in _build_bis: once end_fx=ding@304 is found, a higher ding@308 would replace it
    if fx_ding_308.val >= fx_ding_304.val:
        print(f"\n  ding@308 (val={fx_ding_308.val:.2f}) >= ding@304 (val={fx_ding_304.val:.2f})")
        print(f"  → Open's algorithm replaces end_fx with ding@308 (higher top)")
        print(f"  → Pyarmor keeps ding@304 (doesn't extend)")
        
    # Check what's between ding@304 and ding@308 that might confirm bi at 304
    print(f"\n=== FXes between ding@304 and ding@308 ===")
    for fx in fxs_o:
        if 304 <= fx.k.k_index <= 310:
            print(f"    fx {fx.type:>4} k_index={fx.k.k_index} val={fx.val:.2f} k.index={fx.k.index}")
    
    # Check if any di between 304 and 310 confirms the BI at ding@304
    for fx in fxs_o:
        if fx.type == "di" and 304 < fx.k.k_index <= 310:
            v_confirm = cd_o._bi_fx_valid(fx_ding_304, fx)
            cl_gap = fx.k.index - fx_ding_304.k.index
            k_gap = fx.k.k_index - fx_ding_304.k.k_index
            print(f"\n    Confirm check: _bi_fx_valid(ding@304, di@{fx.k.k_index}) = {v_confirm}")
            print(f"      cl_gap={cl_gap} k_gap={k_gap}")
elif fx_ding_304 is None:
    print("  ding@304 NOT FOUND in FX list!")
    # Find the closest ding near 304
    for fx in fxs_o:
        if fx.type == "ding" and 295 <= fx.k.k_index <= 310:
            print(f"    ding near 304: k_index={fx.k.k_index} val={fx.val:.2f}")
