"""Check what happens if pyarmor uses k_gap for gap check in BTC5m bi[61] area.
If k_gap >= 4 is the gap check, then di@883 (k_gap=4) passes the gap check first.
Then strict check determines if confirmation succeeds."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O

CL_CONFIG = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
}

df = pd.read_parquet("tests/test_data/BTC_USDT_5m_1000.parquet")
cd = CL_O("test", "test", config=CL_CONFIG)
cd.process_klines(df)

fxs = cd.get_fxs()
qj = cd.fx_qj
qy = cd.fx_qy

# Find the FXes we care about
fx_map = {}
for fx in fxs:
    fx_map[fx.k.k_index] = fx

# Key FXes
di875 = fx_map[875]   # start_fx
ding879 = fx_map[879] # end_fx candidate
di883 = fx_map[883]   # first confirmation candidate
ding884 = fx_map[884]
di887 = fx_map[887]
ding889 = fx_map[889]
di891 = fx_map[891]

print("=== Key FX details ===")
for k_idx in [875, 879, 883, 884, 887, 889, 891]:
    fx = fx_map[k_idx]
    print(f"  {fx.type:>4}@{fx.k.k_index} val={fx.val:.2f} "
          f"high={fx.high(qj,qy):.2f} low={fx.low(qj,qy):.2f} "
          f"ck_idx={fx.k.index}")

print("\n=== Checking strict conditions for confirmation from ding@879 ===")
# Confirmation = checking if end_fx(ding) → cur_fx(di) forms valid BI
# This is a DOWN BI: ding@start → di@end
# Strict: start.low < end.low ? start → ding@879, end → di@X
#         end.high > start.high ?

# For each di candidate as confirmation from ding@879:
for fx_kid in [883, 887]:
    di_fx = fx_map[fx_kid]
    cl_gap = di_fx.k.index - ding879.k.index
    k_gap = di_fx.k.k_index - ding879.k.k_index
    
    # Strict check for down BI (ding@879 → di@X):
    c1 = ding879.low(qj, qy) < di_fx.low(qj, qy)  # start low < end low
    c2 = di_fx.high(qj, qy) > ding879.high(qj, qy)  # end high > start high
    
    print(f"\n  ding@879 → di@{fx_kid}:")
    print(f"    cl_gap={cl_gap}, k_gap={k_gap}")
    print(f"    ding879.low={ding879.low(qj,qy):.2f}, di{fx_kid}.low={di_fx.low(qj,qy):.2f}")
    print(f"    ding879.high={ding879.high(qj,qy):.2f}, di{fx_kid}.high={di_fx.high(qj,qy):.2f}")
    print(f"    C1 (start.low < end.low): {c1}")
    print(f"    C2 (end.high > start.high): {c2}")
    if c1 or c2:
        print(f"    → STRICT FAIL (would reject)")
    else:
        print(f"    → STRICT PASS (would confirm)")

print("\n=== Checking primary BI: di@875 → ding@879 ===")
cl_gap = ding879.k.index - di875.k.index
k_gap = ding879.k.k_index - di875.k.k_index
c1 = di875.high(qj, qy) > ding879.high(qj, qy)
c2 = ding879.low(qj, qy) < di875.low(qj, qy)
print(f"  cl_gap={cl_gap}, k_gap={k_gap}")
print(f"  di875.high={di875.high(qj,qy):.2f}, ding879.high={ding879.high(qj,qy):.2f}")
print(f"  di875.low={di875.low(qj,qy):.2f}, ding879.low={ding879.low(qj,qy):.2f}")
print(f"  C1 (start.high > end.high): {c1}")
print(f"  C2 (end.low < start.low): {c2}")

# Also check what happens if ding@879 is NOT set as end_fx (if primary fails with k_gap)
print(f"\n=== If primary di@875→ding@879 uses k_gap: k_gap={k_gap} ===")
if k_gap >= 4:
    print("  → PASSES k_gap >= 4")
else:
    print("  → FAILS k_gap >= 4")

# Check ding@884 and ding@889 as alternatives for end_fx
for ding_kid in [884, 889]:
    ding = fx_map[ding_kid]
    cl_gap_p = ding.k.index - di875.k.index
    k_gap_p = ding.k.k_index - di875.k.k_index
    c1p = di875.high(qj, qy) > ding.high(qj, qy)
    c2p = ding.low(qj, qy) < di875.low(qj, qy)
    print(f"\n  di@875 → ding@{ding_kid}:")
    print(f"    cl_gap={cl_gap_p} k_gap={k_gap_p}")
    print(f"    di875.high={di875.high(qj,qy):.2f}, ding{ding_kid}.high={ding.high(qj,qy):.2f}")
    print(f"    di875.low={di875.low(qj,qy):.2f}, ding{ding_kid}.low={ding.low(qj,qy):.2f}")
    print(f"    C1 (start.high > end.high): {c1p}")
    print(f"    C2 (end.low < start.low): {c2p}")
    print(f"    Better val than ding@879? {ding.val > ding879.val}")

# Check confirmations from ding@889
print("\n=== Checking confirmation from ding@889 ===")
for di_kid in [891]:
    di_fx = fx_map[di_kid]
    cl_gap_c = di_fx.k.index - ding889.k.index
    k_gap_c = di_fx.k.k_index - ding889.k.k_index
    c1c = ding889.low(qj, qy) < di_fx.low(qj, qy)
    c2c = di_fx.high(qj, qy) > ding889.high(qj, qy)
    print(f"  ding@889 → di@{di_kid}:")
    print(f"    cl_gap={cl_gap_c} k_gap={k_gap_c}")
    print(f"    C1: {c1c}, C2: {c2c}")
    if c1c or c2c:
        print(f"    → STRICT FAIL")
    else:
        print(f"    → STRICT PASS")
