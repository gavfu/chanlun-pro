"""Check if BTC60 up 299→308 triggers _bi_special_bi_split.
threshold=20, tolerance=1.
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

df = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cd_o = CL_O("test", "test", config=CL_CONFIG)
cd_o.process_klines(df)

fxs = cd_o.get_fxs()
bis_o = cd_o.get_bis()

# Find baseline bi[15] = up 299→308
bi15 = bis_o[15]
print(f"=== Baseline bi[15]: {bi15.type} {bi15.start.k.k_index}→{bi15.end.k.k_index} ===")
print(f"  start ck_idx={bi15.start.k.index}, end ck_idx={bi15.end.k.index}")

# Internal FXes
internal = [fx for fx in fxs if bi15.start.k.index < fx.k.index < bi15.end.k.index]
print(f"  Internal FXes: {len(internal)}")
for fx in internal:
    print(f"    {fx.type} k={fx.k.k_index} (ck_idx={fx.k.index}) val={fx.val:.2f}")

# Check: start is di@299 (ck_idx=215), end is ding@308 (ck_idx=?)
# Find ding@308
ding308 = None
for fx in fxs:
    if fx.k.k_index == 308 and fx.type == "ding":
        ding308 = fx
        print(f"\n  ding@308 ck_idx={ding308.k.index}")

# ding@304 ck_idx=216, which is between 215 and ding@308's ck_idx
# So ding@304 IS an internal FX!

# But we need at least 3 internal FXes. How many are there between 215 and ding@308?
if ding308:
    internal2 = [fx for fx in fxs if 215 < fx.k.index < ding308.k.index]
    print(f"\n  Internal FXes between ck=215 and ck={ding308.k.index}: {len(internal2)}")
    for fx in internal2:
        print(f"    {fx.type} k={fx.k.k_index} (ck_idx={fx.k.index})")

# Check open's bi[14] and bi[15] specifically
print(f"\n=== Open BIs around divergence ===")
for j in range(13, 20):
    bo = bis_o[j]
    cl = bo.end.k.index - bo.start.k.index
    print(f"  open[{j}]: {bo.type:4s} {bo.start.k.k_index:4d}→{bo.end.k.k_index:4d} "
          f"cl={cl} k={bo.end.k.k_index - bo.start.k.k_index}")

# If bi[15] is up 299→308, internal FXes between ck=215-219:
# ding@304 at ck=216 is the only one (cl_gap=1 from start)
# We need 3 internal FXes to even attempt split. With only 1, split won't trigger.

# So _bi_special_bi_split CANNOT explain pyarmor's bi[15]=up 299→304.
# Pyarmor must build this BI directly in _build_bis.

# Let's check: what if pyarmor uses cl_gap >= 4 for gap check 
# BUT uses a DIFFERENT strict check that makes di@299→ding@304 PASS?
# 
# We know: cl_gap=1 < 4 → FAILS current gap check.
# But what if pyarmor doesn't check gap for the PRIMARY BI candidate,
# only for CONFIRMATION?
# 
# Let me test this: gap check only on confirmation, not on primary check

print(f"\n\n=== NEW HYPOTHESIS: gap check only on CONFIRMATION ===")
print("Testing: no gap check for primary BI, gap check for confirmation")

# Also: what about the check in EXTENSION?
# In _build_bis: extension uses check_valid with same start_fx
# Confirmation uses check_valid with end_fx as start

# What if gap is checked ONLY in confirmation (end_fx → confirm_fx)?
# Then di@299→ding@304 passes (no gap check)
# And ding@304's confirmation: cl_gap from ding@304 to next di
# Let's check what confirmations look like

ding304 = None
for fx in fxs:
    if fx.k.k_index == 304 and fx.type == "ding": ding304 = fx; break

qj = cd_o.fx_qj; qy = cd_o.fx_qy
print(f"\n  Confirmations from ding@304:")
for fx in fxs:
    if fx.type == "di" and fx.k.k_index > 304:
        cl = fx.k.index - ding304.k.index
        k = fx.k.k_index - ding304.k.k_index
        h_s = ding304.high(qj, qy); l_s = ding304.low(qj, qy)
        h_e = fx.high(qj, qy); l_e = fx.low(qj, qy)
        # down BI: ding→di
        c1 = l_s < l_e  # start.low < end.low
        c2 = h_e > h_s  # end.high > start.high
        strict_fail = (c1 or c2) if k < 13 else False
        gap_cl = cl >= 4
        gap_k = k >= 4
        status = "→ CONFIRMED" if gap_cl and not strict_fail else ""
        print(f"    di@{fx.k.k_index}: cl={cl} k={k} "
              f"gap_cl={gap_cl} gap_k={gap_k} "
              f"C1={c1} C2={c2} strict={'FAIL' if strict_fail else 'PASS'} "
              f"{status}")
        if fx.k.k_index > 320: break
