"""
Investigate ETH5m BI divergence: 71 vs 73 BIs.
Find the FXes that pyarmor uses but open doesn't.
"""
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

df = pd.read_parquet("tests/test_data/ETH_USDT_5m_1000.parquet")
cd_o = CL_O("test", "test", config=CL_CONFIG)
cd_o.process_klines(df)
cd_p = CL_P("test", "test", config=CL_CONFIG)
cd_p.process_klines(df)

fxs_o = cd_o.get_fxs()
fxs_p = cd_p.get_fxs()
bis_o = cd_o.get_bis()
bis_p = cd_p.get_bis()

print(f"Open:    {len(fxs_o)} FXes, {len(bis_o)} BIs")
print(f"Pyarmor: {len(fxs_p)} FXes, {len(bis_p)} BIs")

# First divergence: open bi[26] k=333→349 vs pyarmor bi[26] k=333→337
# Show FXes in range 330-365
print("\n=== FXes near first divergence (k_index 325-365) ===")
print("  Open:")
for fx in fxs_o:
    if 325 <= fx.k.k_index <= 365:
        klines_indices = [k.k_index for k in fx.klines]
        print(f"    fx {fx.type:>4} k_index={fx.k.k_index} val={fx.val:.2f} done={fx.done} klines={klines_indices}")
print("  Pyarmor:")
for fx in fxs_p:
    if 325 <= fx.k.k_index <= 365:
        klines_indices = [k.k_index for k in fx.klines]
        print(f"    fx {fx.type:>4} k_index={fx.k.k_index} val={fx.val:.2f} done={fx.done} klines={klines_indices}")

# FX comparison - find first difference
print("\n=== FX comparison (first 10 differences) ===")
diffs = 0
for i in range(min(len(fxs_o), len(fxs_p))):
    fo = fxs_o[i]
    fp = fxs_p[i]
    if fo.k.k_index != fp.k.k_index or fo.type != fp.type or abs(fo.val - fp.val) > 0.001:
        print(f"  fx[{i}] open: {fo.type} k_index={fo.k.k_index} val={fo.val:.2f}")
        print(f"         pya:  {fp.type} k_index={fp.k.k_index} val={fp.val:.2f}")
        diffs += 1
        if diffs >= 10:
            break

if diffs == 0:
    print("  All FXes identical!")

# Show BIs 24-30 for both
print("\n=== Open BIs 24-29 ===")
for bi in bis_o[24:30]:
    k_gap = bi.end.k.k_index - bi.start.k.k_index
    print(f"  bi[{bi.index:>2}] {bi.type:>4} k={bi.start.k.k_index}→{bi.end.k.k_index} gap={k_gap}")

print("\n=== Pyarmor BIs 24-31 ===")
for bi in bis_p[24:32]:
    k_gap = bi.end.k.k_index - bi.start.k.k_index
    print(f"  bi[{bi.index:>2}] {bi.type:>4} k={bi.start.k.k_index}→{bi.end.k.k_index} gap={k_gap}")

# Check merged K-lines around k_index 333-337
cklines_o = cd_o.get_cl_klines()

# Find which merged K-lines cover raw indices 333-349 
print("\n=== Merged K-lines covering raw 330-360 ===")
for ck in cklines_o:
    raw_indices = [k.index for k in ck.klines]
    if any(330 <= r <= 360 for r in raw_indices):
        print(f"  ck[{ck.index:>3}] k_index={ck.k_index} h={ck.h:.2f} l={ck.l:.2f} raws={raw_indices}")

# Check why open rejects bi from ding@333 to di@337
# Find the actual FX objects
fx_ding_333 = None
fx_di_337 = None
for fx in fxs_o:
    if fx.k.k_index == 333 and fx.type == "ding":
        fx_ding_333 = fx
    if fx.k.k_index == 337 and fx.type == "di":
        fx_di_337 = fx

if fx_ding_333 and fx_di_337:
    print("\n=== Checking _bi_fx_valid(ding@333, di@337) ===")
    print(f"  ding@333: val={fx_ding_333.val:.2f} k.k_index={fx_ding_333.k.k_index} k.index={fx_ding_333.k.index}")
    print(f"  di@337:   val={fx_di_337.val:.2f} k.k_index={fx_di_337.k.k_index} k.index={fx_di_337.k.index}")
    
    cl_gap = fx_di_337.k.index - fx_ding_333.k.index
    k_gap = fx_di_337.k.k_index - fx_ding_333.k.k_index
    print(f"  cl_gap={cl_gap} k_gap={k_gap}")
    print(f"  cl_gap < 4? {cl_gap < 4}")
    print(f"  k_gap < fx_check_k_nums(13)? {k_gap < 13}")
    
    # Check strict FX conditions
    qj = cd_o.fx_qj
    qy = cd_o.fx_qy
    print(f"  fx_qj={qj} fx_qy={qy}")
    print(f"  Strict check (ding → di, downward BI):")
    print(f"    ding.low={fx_ding_333.low(qj, qy):.2f} di.low={fx_di_337.low(qj, qy):.2f}")
    print(f"    ding.low < di.low? {fx_ding_333.low(qj, qy) < fx_di_337.low(qj, qy)}")
    print(f"    di.high={fx_di_337.high(qj, qy):.2f} ding.high={fx_ding_333.high(qj, qy):.2f}")
    print(f"    di.high > ding.high? {fx_di_337.high(qj, qy) > fx_ding_333.high(qj, qy)}")
    
    result = cd_o._bi_fx_valid(fx_ding_333, fx_di_337)
    print(f"  _bi_fx_valid result: {result}")
    
    # Also check pyarmor
    fx_ding_333_p = None
    fx_di_337_p = None
    for fx in fxs_p:
        if fx.k.k_index == 333 and fx.type == "ding":
            fx_ding_333_p = fx
        if fx.k.k_index == 337 and fx.type == "di":
            fx_di_337_p = fx
    
    if fx_ding_333_p and fx_di_337_p:
        result_p = cd_p._bi_fx_valid(fx_ding_333_p, fx_di_337_p)
        print(f"  pyarmor _bi_fx_valid result: {result_p}")
        
        print(f"\n  Pyarmor strict check:")
        print(f"    ding.low={fx_ding_333_p.low(qj, qy):.2f} di.low={fx_di_337_p.low(qj, qy):.2f}")
        print(f"    ding.low < di.low? {fx_ding_333_p.low(qj, qy) < fx_di_337_p.low(qj, qy)}")
        print(f"    di.high={fx_di_337_p.high(qj, qy):.2f} ding.high={fx_ding_333_p.high(qj, qy):.2f}")
        print(f"    di.high > ding.high? {fx_di_337_p.high(qj, qy) > fx_ding_333_p.high(qj, qy)}")
