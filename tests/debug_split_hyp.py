"""Investigate pyarmor's split logic more deeply for Case 2.
Hypothesis: pyarmor uses triplet-only candidates (no before-triplet lookback)
AND k_gap for split gap check."""
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
cd.bi_split_k_cross_nums = 0
cd.process_klines(df)
bis_raw = cd.get_bis()
fxs = cd.get_fxs()

print("=== HYPOTHESIS: triplet-only candidates + k_gap for split gap ===\n")

# Test Case 1: up 299→330
bi1 = bis_raw[13]
start1 = bi1.start.k.index
end1 = bi1.end.k.index
ifx1 = [fx for fx in fxs if start1 < fx.k.index < end1]
trip1 = (ifx1[0], ifx1[1], ifx1[2])  # ding@304, di@306, ding@308

print("Case 1: up 299→330, triplet=(ding@304, di@306, ding@308)")
# UP: split1=ding, triplet-only
cands1 = [fx for fx in trip1 if fx.type == "ding"]
# k_gap check from start
for f in cands1:
    k = f.k.k_index - bi1.start.k.k_index
    print(f"  ding@{f.k.k_index}: k_gap={k} {'✓' if k>=4 else '✗'}")
# First with k_gap>=4
split1_1 = next(f for f in cands1 if (f.k.k_index - bi1.start.k.k_index) >= 4)
print(f"  → split1 = ding@{split1_1.k.k_index}")

# split2: di after ding@304, split1.val > di.val
cands2_1 = [fx for fx in ifx1 if fx.type == "di" and fx.k.index > split1_1.k.index
            and split1_1.val > fx.val]
gap_ok_1 = [f for f in cands2_1 if (f.k.k_index - split1_1.k.k_index) >= 4]
print(f"  di after split1 with k_gap>=4: {[(f.k.k_index, f.val) for f in gap_ok_1]}")
if gap_ok_1:
    split2_1 = min(gap_ok_1, key=lambda f: f.val)
    print(f"  → split2 = di@{split2_1.k.k_index} (min val)")
print(f"  Result: {bi1.start.k.k_index}→{split1_1.k.k_index}→{split2_1.k.k_index}→{bi1.end.k.k_index}")
print(f"  Pyarmor: 299→304→309→330 {'✓ MATCH' if split1_1.k.k_index==304 and split2_1.k.k_index==309 else '✗ MISMATCH'}")

# Test Case 2: down 911→992
bi2 = bis_raw[48]
start2 = bi2.start.k.index
end2 = bi2.end.k.index
ifx2 = [fx for fx in fxs if start2 < fx.k.index < end2]
# Triplet: [14]=(di@957, ding@959, di@960)
trip2 = (ifx2[14], ifx2[15], ifx2[16])

print(f"\nCase 2: down 911→992, triplet=(di@957, ding@959, di@960)")
# DOWN: split1=di, triplet-only
cands2 = [fx for fx in trip2 if fx.type == "di"]
for f in cands2:
    k = f.k.k_index - bi2.start.k.k_index
    print(f"  di@{f.k.k_index}: k_gap={k} {'✓' if k>=4 else '✗'}")
split1_2 = next(f for f in cands2 if (f.k.k_index - bi2.start.k.k_index) >= 4)
print(f"  → split1 = di@{split1_2.k.k_index}")

# split2: ding after di@957, val > split1.val
cands2_2 = [fx for fx in ifx2 if fx.type == "ding" and fx.k.index > split1_2.k.index
            and fx.val > split1_2.val]
# For DOWN, need max val ding
gap_ok_2 = [f for f in cands2_2 if (f.k.k_index - split1_2.k.k_index) >= 4]
print(f"  ding after split1 with k_gap>=4:")
for f in gap_ok_2:
    print(f"    ding@{f.k.k_index} val={f.val:.2f}")
if gap_ok_2:
    split2_2 = max(gap_ok_2, key=lambda f: f.val)
    print(f"  → split2 = ding@{split2_2.k.k_index} (max val)")
print(f"  Result: {bi2.start.k.k_index}→{split1_2.k.k_index}→{split2_2.k.k_index}→{bi2.end.k.k_index}")
print(f"  Pyarmor: 911→957→967→992 {'✓ MATCH' if split1_2.k.k_index==957 and split2_2.k.k_index==967 else '✗ MISMATCH'}")

print(f"\n\n=== ALTERNATE: triplet-only + cl_gap ===")
# Case 1: cl_gap
cands1_cl = [fx for fx in trip1 if fx.type == "ding"]
for f in cands1_cl:
    cl = f.k.index - bi1.start.k.index
    print(f"  ding@{f.k.k_index}: cl_gap={cl} {'✓' if cl>=4 else '✗'}")
cl_ok = [f for f in cands1_cl if (f.k.index - bi1.start.k.index) >= 4]
if cl_ok:
    s1 = cl_ok[0]
    print(f"  → split1 = ding@{s1.k.k_index}")
else:
    s1 = cands1_cl[-1]
    print(f"  → split1 = ding@{s1.k.k_index} (last candidate, no gap_ok)")
print(f"  Pyarmor wants ding@304 → {'✓' if s1.k.k_index==304 else '✗ WRONG'}")

print(f"\n=== CONCLUSION ===")
print(f"triplet-only + k_gap: Case 1 ✓, Case 2 ✓")
print(f"triplet-only + cl_gap: Case 1 ✗ (gives 308), Case 2 ✓")
print(f"before-triplet + k_gap: Case 1 ✓, Case 2 ✗ (gives 915)")
print(f"before-triplet + cl_gap: Case 1 ✗, Case 2 ✗")
print(f"\n→ FIX: Use triplet-only candidates for split1 + k_gap for _split_gap_ok")
