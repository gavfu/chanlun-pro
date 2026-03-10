"""Check if there's a validation reason pyarmor rejects split1=di@192 for BTC60 raw[8].
Maybe the sub-BIs don't satisfy gap/strict checks?"""
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

# Find the FXes
fx_by_k = {}
for fx in fxs:
    fx_by_k[fx.k.k_index] = fx

# Check gap conditions for both candidates
# Option A: 187→192→197→215
# Option B: 187→199→204→215
for s1_k, s2_k, label in [(192, 197, "Option A (open)"), (199, 204, "Option B (pyarmor)")]:
    ding187 = fx_by_k[187]
    di_s1 = fx_by_k[s1_k]
    ding_s2 = fx_by_k[s2_k]
    di215 = fx_by_k[215]
    
    print(f"\n{label}: 187→{s1_k}→{s2_k}→215")
    
    # Sub-BI 1: start→split1 (down: ding@187→di@s1)
    cl1 = di_s1.k.index - ding187.k.index
    k1 = di_s1.k.k_index - ding187.k.k_index
    valid1 = cd._split_gap_ok(ding187, di_s1)
    print(f"  Sub-BI 1 down {ding187.k.k_index}→{di_s1.k.k_index}: cl={cl1}, k={k1}, gap_ok={valid1}")
    
    # Sub-BI 2: split1→split2 (up: di→ding)
    cl2 = ding_s2.k.index - di_s1.k.index
    k2 = ding_s2.k.k_index - di_s1.k.k_index
    valid2 = cd._split_gap_ok(di_s1, ding_s2)
    print(f"  Sub-BI 2 up   {di_s1.k.k_index}→{ding_s2.k.k_index}: cl={cl2}, k={k2}, gap_ok={valid2}")
    
    # Sub-BI 3: split2→end (down: ding→di)
    cl3 = di215.k.index - ding_s2.k.index
    k3 = di215.k.k_index - ding_s2.k.k_index
    valid3 = cd._split_gap_ok(ding_s2, di215)
    print(f"  Sub-BI 3 down {ding_s2.k.k_index}→{di215.k.k_index}: cl={cl3}, k={k3}, gap_ok={valid3}")
    
    # Check _bi_fx_valid (full check including strict)
    full_valid1 = cd._bi_fx_valid(ding187, di_s1)
    full_valid2 = cd._bi_fx_valid(di_s1, ding_s2)
    full_valid3 = cd._bi_fx_valid(ding_s2, di215)
    print(f"  Full _bi_fx_valid: sub1={full_valid1}, sub2={full_valid2}, sub3={full_valid3}")
    
    # Direction checks
    dir_ok1 = ding187.val > di_s1.val  # down: ding > di
    dir_ok2 = ding_s2.val > di_s1.val  # up: ding > di  
    dir_ok3 = ding_s2.val > di215.val  # down: ding > di
    print(f"  Direction: sub1={'✓' if dir_ok1 else '✗'}({ding187.val:.1f}>{di_s1.val:.1f}) "
          f"sub2={'✓' if dir_ok2 else '✗'}({ding_s2.val:.1f}>{di_s1.val:.1f}) "
          f"sub3={'✓' if dir_ok3 else '✗'}({ding_s2.val:.1f}>{di215.val:.1f})")

# Now check ALL internal FXes in BTC60 raw[8] for gap_ok patterns
print(f"\n\n=== Detailed FX analysis for down 187→215 ===")
bi_start = fx_by_k[187]
bi_end = fx_by_k[215]
start_ci = bi_start.k.index
end_ci = bi_end.k.index

internal = [fx for fx in fxs if start_ci < fx.k.index < end_ci]
print(f"Internal FXes:")
for fx in internal:
    k_gap = fx.k.k_index - bi_start.k.k_index 
    cl_gap = fx.k.index - bi_start.k.index
    print(f"  {fx.type}@{fx.k.k_index} val={fx.val:>10.2f} cl_gap={cl_gap:>2d} k_gap={k_gap:>2d}")

# Check all valid (di, ding) split pairs
print(f"\n=== All valid split pairs ===")
dis = [fx for fx in internal if fx.type == "di"]
dings = [fx for fx in internal if fx.type == "ding"]

valid_pairs = []
for d in dis:
    for g in dings:
        if g.k.index > d.k.index and g.val > d.val:  # ding after di, ding > di
            # Check all 3 sub-BI gaps
            g1 = cd._split_gap_ok(bi_start, d)
            g2 = cd._split_gap_ok(d, g)
            g3 = cd._split_gap_ok(g, bi_end)
            if g1 and g2 and g3:
                valid_pairs.append((d, g, g1, g2, g3))

print(f"Valid pairs (gap_ok for all 3 sub-BIs):")
for d, g, g1, g2, g3 in valid_pairs:
    print(f"  di@{d.k.k_index}({d.val:.1f}) → ding@{g.k.k_index}({g.val:.1f})")

# Now: which one does pyarmor choose? 
# Pyarmor: di@199, ding@204
# What selection criterion gives (199, 204)?
print(f"\n=== Selection criteria comparison ===")
if valid_pairs:
    # Sort by different criteria
    # By split1 val ascending (lowest di first)
    by_val = sorted(valid_pairs, key=lambda p: p[0].val)
    print(f"  Lowest di: di@{by_val[0][0].k.k_index}→ding@{by_val[0][1].k.k_index}")
    
    # By split1 position (first di) 
    by_pos = sorted(valid_pairs, key=lambda p: p[0].k.index)
    print(f"  First di: di@{by_pos[0][0].k.k_index}→ding@{by_pos[0][1].k.k_index}")
    
    # By split1 position (last di in triplet)
    by_pos_rev = sorted(valid_pairs, key=lambda p: p[0].k.index, reverse=True)
    print(f"  Last di: di@{by_pos_rev[0][0].k.k_index}→ding@{by_pos_rev[0][1].k.k_index}")
