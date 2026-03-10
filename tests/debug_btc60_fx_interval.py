"""Check how FX.high/low values change with different qy modes for BTC60 ding@304."""
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

df = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cd = CL_O("test", "test", config=CL_CONFIG)
cd.process_klines(df)

fxs = cd.get_fxs()
cls = cd.get_cl_klines()
srcs = cd.get_src_klines()

# Find ding@304 and show its 3 merged K-lines
for fx in fxs:
    if fx.k.k_index == 304:
        ding304 = fx
        break

print("=== ding@304 FX structure ===")
print(f"Center CK: index={ding304.k.index}, k_index={ding304.k.k_index}")
print(f"val={ding304.val}")

# The FX uses klines[0], klines[1] (center), klines[2]
# klines is the 3 merged K-lines of the FX
for i, ck in enumerate(ding304.klines):
    if ck is None:
        print(f"  klines[{i}]: None")
        continue
    print(f"  klines[{i}]: CK index={ck.index} k_index={ck.k_index} h={ck.h:.2f} l={ck.l:.2f}")
    # Show raw K-lines within this merged K-line
    for sk in ck.klines:
        print(f"    raw: k_index={sk.index} h={sk.h:.2f} l={sk.l:.2f}")

# Show high/low with different modes
print(f"\n  high(fx_qj_k, fx_qy_three) = {ding304.high('fx_qj_k', 'fx_qy_three'):.2f}")
print(f"  low(fx_qj_k, fx_qy_three)  = {ding304.low('fx_qj_k', 'fx_qy_three'):.2f}")

# What would a "middle only" calculation give?
# Only use center merged K-line's raw K-lines
center_ck = ding304.klines[1] if ding304.klines[1] is not None else ding304.k
center_highs = [sk.h for sk in center_ck.klines]
center_lows = [sk.l for sk in center_ck.klines]
print(f"  Center-only high = {max(center_highs):.2f}")
print(f"  Center-only low  = {min(center_lows):.2f}")

# What about using CK h/l (merged K-line h/l) rather than raw K-line h/l?
print(f"\n  CK-based high/low:")
for i, ck in enumerate(ding304.klines):
    if ck is not None:
        print(f"    klines[{i}]: h={ck.h:.2f} l={ck.l:.2f}")

# Now show same for di@309
for fx in fxs:
    if fx.k.k_index == 309:
        di309 = fx
        break

print(f"\n=== di@309 FX structure ===")
print(f"Center CK: index={di309.k.index}, k_index={di309.k.k_index}")
print(f"val={di309.val}")
for i, ck in enumerate(di309.klines):
    if ck is None:
        print(f"  klines[{i}]: None")
        continue
    print(f"  klines[{i}]: CK index={ck.index} k_index={ck.k_index} h={ck.h:.2f} l={ck.l:.2f}")
    for sk in ck.klines:
        print(f"    raw: index={sk.index} h={sk.h:.2f} l={sk.l:.2f}")

print(f"\n  high(fx_qj_k, fx_qy_three) = {di309.high('fx_qj_k', 'fx_qy_three'):.2f}")
print(f"  low(fx_qj_k, fx_qy_three)  = {di309.low('fx_qj_k', 'fx_qy_three'):.2f}")

center_ck = di309.klines[1] if di309.klines[1] is not None else di309.k
center_highs = [sk.h for sk in center_ck.klines]
center_lows = [sk.l for sk in center_ck.klines]
print(f"  Center-only high = {max(center_highs):.2f}")
print(f"  Center-only low  = {min(center_lows):.2f}")

# Strict check comparison
print("\n=== Strict check: ding@304 → di@309 (down BI) ===")
qj, qy = 'fx_qj_k', 'fx_qy_three'
print(f"  C1: ding304.low({ding304.low(qj,qy):.2f}) < di309.low({di309.low(qj,qy):.2f}) = {ding304.low(qj,qy) < di309.low(qj,qy)}")
print(f"  C2: di309.high({di309.high(qj,qy):.2f}) > ding304.high({ding304.high(qj,qy):.2f}) = {di309.high(qj,qy) > ding304.high(qj,qy)}")

# With CK-based (merged K-line h/l):
ding304_ck_low = min(ck.l for ck in ding304.klines if ck is not None)
ding304_ck_high = max(ck.h for ck in ding304.klines if ck is not None)
di309_ck_low = min(ck.l for ck in di309.klines if ck is not None)
di309_ck_high = max(ck.h for ck in di309.klines if ck is not None)
print(f"\n  CK-based strict check:")
print(f"  C1: ding304.ck_low({ding304_ck_low:.2f}) < di309.ck_low({di309_ck_low:.2f}) = {ding304_ck_low < di309_ck_low}")
print(f"  C2: di309.ck_high({di309_ck_high:.2f}) > ding304.ck_high({ding304_ck_high:.2f}) = {di309_ck_high > ding304_ck_high}")

# Also check ding@304 → di@306
for fx in fxs:
    if fx.k.k_index == 306:
        di306 = fx
        break

print(f"\n=== Strict check: ding@304 → di@306 (down BI) ===")
k_gap_306 = di306.k.k_index - ding304.k.k_index
print(f"  k_gap={k_gap_306}")
print(f"  C1: ding304.low({ding304.low(qj,qy):.2f}) < di306.low({di306.low(qj,qy):.2f}) = {ding304.low(qj,qy) < di306.low(qj,qy)}")
print(f"  C2: di306.high({di306.high(qj,qy):.2f}) > ding304.high({ding304.high(qj,qy):.2f}) = {di306.high(qj,qy) > ding304.high(qj,qy)}")

di306_ck_low = min(ck.l for ck in di306.klines if ck is not None)
di306_ck_high = max(ck.h for ck in di306.klines if ck is not None)
print(f"  CK-based C1: ding304_ck_low({ding304_ck_low:.2f}) < di306_ck_low({di306_ck_low:.2f}) = {ding304_ck_low < di306_ck_low}")
print(f"  CK-based C2: di306_ck_high({di306_ck_high:.2f}) > ding304_ck_high({ding304_ck_high:.2f}) = {di306_ck_high > ding304_ck_high}")

# Show di@299→ding@304 with strict
for fx in fxs:
    if fx.k.k_index == 299:
        di299 = fx
        break

print(f"\n=== Strict check: di@299 → ding@304 (up BI) ===")
print(f"  k_gap={ding304.k.k_index - di299.k.k_index}")
print(f"  C1: di299.high({di299.high(qj,qy):.2f}) > ding304.high({ding304.high(qj,qy):.2f}) = {di299.high(qj,qy) > ding304.high(qj,qy)}")
print(f"  C2: ding304.low({ding304.low(qj,qy):.2f}) < di299.low({di299.low(qj,qy):.2f}) = {ding304.low(qj,qy) < di299.low(qj,qy)}")

di299_ck_low = min(ck.l for ck in di299.klines if ck is not None)
di299_ck_high = max(ck.h for ck in di299.klines if ck is not None)
print(f"  CK-based C1: di299_ck_high({di299_ck_high:.2f}) > ding304_ck_high({ding304_ck_high:.2f}) = {di299_ck_high > ding304_ck_high}")
print(f"  CK-based C2: ding304_ck_low({ding304_ck_low:.2f}) < di299_ck_low({di299_ck_low:.2f}) = {ding304_ck_low < di299_ck_low}")
