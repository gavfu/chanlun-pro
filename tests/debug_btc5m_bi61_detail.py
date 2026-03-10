"""Check strict check values for BTC5m ding@879→di@887 confirmation."""
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
cd_o = CL_O("test", "test", config=CL_CONFIG)
cd_o.process_klines(df)
fxs = cd_o.get_fxs()

qj = cd_o.fx_qj
qy = cd_o.fx_qy

ding879 = None
di883 = None
di887 = None
ding889 = None

for fx in fxs:
    if fx.type == "ding" and fx.k.k_index == 879: ding879 = fx
    if fx.type == "di" and fx.k.k_index == 883: di883 = fx
    if fx.type == "di" and fx.k.k_index == 887: di887 = fx
    if fx.type == "ding" and fx.k.k_index == 889: ding889 = fx

print("=== ding@879 (end_fx candidate for open bi[61]) ===")
print(f"  val={ding879.val:.2f}")
print(f"  high={ding879.high(qj,qy):.2f} low={ding879.low(qj,qy):.2f}")
print(f"  k.index={ding879.k.index}")
print(f"  klines:")
for kl in ding879.klines:
    if kl is None: 
        print(f"    None")
        continue
    print(f"    ck k_index={kl.k_index} h={kl.h:.2f} l={kl.l:.2f} (n_sub={len(kl.klines)})")
    for sub in kl.klines:
        print(f"      raw h={sub.h:.2f} l={sub.l:.2f}")

print(f"\n=== di@887 (confirmation FX) ===")
print(f"  val={di887.val:.2f}")
print(f"  high={di887.high(qj,qy):.2f} low={di887.low(qj,qy):.2f}")
print(f"  k.index={di887.k.index}")
print(f"  klines:")
for kl in di887.klines:
    if kl is None:
        print(f"    None")
        continue
    print(f"    ck k_index={kl.k_index} h={kl.h:.2f} l={kl.l:.2f} (n_sub={len(kl.klines)})")
    for sub in kl.klines:
        print(f"      raw h={sub.h:.2f} l={sub.l:.2f}")

print(f"\n=== Strict check for down BI: ding@879 → di@887 ===")
cl_gap = di887.k.index - ding879.k.index
k_gap = di887.k.k_index - ding879.k.k_index
print(f"  cl_gap={cl_gap} k_gap={k_gap}")
print(f"  k_gap < fx_check_k_nums? {k_gap} < 13 = {k_gap < 13}")

# For down BI (ding→di):
print(f"  C1: ding879.low ({ding879.low(qj,qy):.2f}) < di887.low ({di887.low(qj,qy):.2f})? = {ding879.low(qj,qy) < di887.low(qj,qy)}")
print(f"  C2: di887.high ({di887.high(qj,qy):.2f}) > ding879.high ({ding879.high(qj,qy):.2f})? = {di887.high(qj,qy) > ding879.high(qj,qy)}")

result = cd_o._bi_fx_valid(ding879, di887)
print(f"\n  _bi_fx_valid(ding@879, di@887) = {result}")

# Also check if di@883 might pass with k_gap check
print(f"\n=== di@883 (first di after ding@879) ===")
cl_gap_883 = di883.k.index - ding879.k.index
k_gap_883 = di883.k.k_index - ding879.k.k_index
print(f"  cl_gap={cl_gap_883} k_gap={k_gap_883}")
print(f"  _bi_fx_valid(ding@879, di@883) = {cd_o._bi_fx_valid(ding879, di883)}")

# Check what happens with lower gap threshold: would ding@879→di@883 pass with cl_gap >= 3?
if cl_gap_883 < 4 and cl_gap_883 >= 3:
    print(f"  Would pass with cl_gap >= 3 threshold!")
    
# If confirmation at di@883 passes, then open would confirm bi[61] = up @875→879
# and start new bi from ding@879. Then looking from ding@879, we need di endpoint.
# di@883 with cl_gap=3 would still be too close. But the NEXT di after ding@879 
# that passes gap+strict would be di@887 (cl_gap=6, k_gap=8).

# The key question: in pyarmor, does ding@879 ever become end_fx?
# If pyarmor rejects ding@879 as end_fx (because of stricter conditions on primary BI),
# then it would continue scanning and find ding@889 (higher val).

# Let's check: does di@875 → ding@879 pass with different strict configs?
di875 = None
for fx in fxs:
    if fx.type == "di" and fx.k.k_index == 875:
        di875 = fx
        break

print(f"\n=== Primary BI check: di@875 → ding@879 ===")
cl_gap_p = ding879.k.index - di875.k.index
k_gap_p = ding879.k.k_index - di875.k.k_index
print(f"  cl_gap={cl_gap_p} k_gap={k_gap_p}")
print(f"  _bi_fx_valid(di@875, ding@879) = {cd_o._bi_fx_valid(di875, ding879)}")

# Check strict for up BI (di→ding):
print(f"  C1: di875.high ({di875.high(qj,qy):.2f}) > ding879.high ({ding879.high(qj,qy):.2f})? = {di875.high(qj,qy) > ding879.high(qj,qy)}")
print(f"  C2: ding879.low ({ding879.low(qj,qy):.2f}) < di875.low ({di875.low(qj,qy):.2f})? = {ding879.low(qj,qy) < di875.low(qj,qy)}")
