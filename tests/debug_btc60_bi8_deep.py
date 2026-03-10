"""Focus on BTC60 raw[8] split to understand why pyarmor picks di@199 over di@192.
Maybe the selection is: from ALL internal FXes, pick the most extreme split1 
that is inside or near the triggered triplet, with gap_ok for ALL 3 sub-BIs.
Or maybe it's: from the triplet, pick split1 that maximizes split2's extremeness."""
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
fxs = cd.get_fxs()

fx_by_k = {}
for fx in fxs:
    fx_by_k[fx.k.k_index] = fx

# Internal FXes for raw[8] down 187→215
bi_start = fx_by_k[187]
bi_end = fx_by_k[215]
start_ci = bi_start.k.index
end_ci = bi_end.k.index
internal = [fx for fx in fxs if start_ci < fx.k.index < end_ci]

# Triplet: [0]=(di@192, ding@197, di@199)
trip = (internal[0], internal[1], internal[2])

# For all pairs, run _select_split_down logic with each possible approach
print("=== All possible (di, ding) pairs with gap_ok ===")
dis = [fx for fx in internal if fx.type == "di"]
dings = [fx for fx in internal if fx.type == "ding"]

for d in dis:
    for g in dings:
        if g.k.index <= d.k.index:
            continue
        if g.val <= d.val:
            continue
        g1 = cd._split_gap_ok(bi_start, d)
        g2 = cd._split_gap_ok(d, g)
        g3 = cd._split_gap_ok(g, bi_end)
        if g1 and g2 and g3:
            print(f"  di@{d.k.k_index}({d.val:.1f}) → ding@{g.k.k_index}({g.val:.1f}) "
                  f"k_gap: {d.k.k_index-187},{g.k.k_index-d.k.k_index},{215-g.k.k_index}")

# Now let me check: what if pyarmor doesn't iterate from triplet at all,
# but instead uses a completely different approach?
# Maybe: iterate ALL dis sorted by val ascending (most extreme first for down),
# and for each, find the best ding with gap_ok?
print(f"\n=== Approach: iterate dis by val ascending, find best ding ===")
for d in sorted(dis, key=lambda f: f.val):
    if not cd._split_gap_ok(bi_start, d):
        continue
    valid_dings = [g for g in dings 
                   if g.k.index > d.k.index 
                   and g.val > d.val 
                   and cd._split_gap_ok(d, g)
                   and cd._split_gap_ok(g, bi_end)]
    if valid_dings:
        best = max(valid_dings, key=lambda f: f.val)
        print(f"  di@{d.k.k_index}({d.val:.1f}) → ding@{best.k.k_index}({best.val:.1f})")

# What if pyarmor uses triplet-ONLY for BOTH split1 and then checks gap?
# Triplet dis: [192, 199]
# With gap_ok (k_gap>=4): both pass
# Then it picks the one whose ding (split2) has the highest val?
print(f"\n=== Triplet-only, pick split1 that gives highest ding split2 ===")
for d in [internal[0], internal[2]]:  # di@192, di@199
    if d.type != "di":
        continue
    if not cd._split_gap_ok(bi_start, d):
        continue
    valid_dings = [g for g in internal 
                   if g.type == "ding" and g.k.index > d.k.index 
                   and g.val > d.val 
                   and cd._split_gap_ok(d, g)
                   and cd._split_gap_ok(g, bi_end)]
    if valid_dings:
        best = max(valid_dings, key=lambda f: f.val)
        print(f"  split1=di@{d.k.k_index}({d.val:.1f}) → split2=ding@{best.k.k_index}({best.val:.1f})")

# What about: the OPPOSITE approach? Find split2 first (most extreme ding),
# then find split1 that gives gap_ok?
print(f"\n=== Find best ding first, then best di ===")
for g in sorted(dings, key=lambda f: f.val, reverse=True):
    if not cd._split_gap_ok(g, bi_end):
        continue
    valid_dis = [d for d in dis 
                 if d.k.index < g.k.index 
                 and d.val < g.val 
                 and cd._split_gap_ok(bi_start, d)
                 and cd._split_gap_ok(d, g)]
    if valid_dis:
        # For down: pick lowest di
        best = min(valid_dis, key=lambda f: f.val)
        print(f"  ding@{g.k.k_index}({g.val:.1f}) → di@{best.k.k_index}({best.val:.1f})")
        break  # Take first (highest ding)

# What if the current code's approach is correct but the SORT order is different?
# Current: _find_split1_from_triplet returns first gap_ok from triplet
# Then _find_split2 returns max/min val with gap_ok from ALL internal
# But wait: _find_split2 uses cl_gap check (fx.k.index - split1_fx.k.index >= 4)
# which we changed to _split_gap_ok (k_gap >= 4)
# Let me check what _find_split2 gives with each split1:

print(f"\n=== _find_split2 results for each split1 ===")
for d in [fx_by_k[192], fx_by_k[199]]:
    # split2: ding after di, val > di.val, gap_ok
    valid = [g for g in dings 
             if g.k.index > d.k.index and g.val > d.val]
    gap_ok = [g for g in valid if cd._split_gap_ok(d, g)]
    if gap_ok:
        best = max(gap_ok, key=lambda f: f.val)
        print(f"  split1=di@{d.k.k_index}: gap_ok dings={[(g.k.k_index, g.val) for g in gap_ok]} → max=ding@{best.k.k_index}")
