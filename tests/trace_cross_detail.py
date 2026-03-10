"""
Targeted tracer to capture the exact cross-counting mechanism in pyarmor's
_bi_special_bi_split. Focuses on testing different cross-counting hypotheses
against the known pyarmor result (triplet0=4hits, triplet1=23hits).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_interface import *
from chanlun.cl_pyarmor import CL as CL_Pyarmor

# Load ETH5m data
df = pd.read_parquet(os.path.join(os.path.dirname(__file__), 'test_data', 'ETH_USDT_5m_1000.parquet'))

cl_p = CL_Pyarmor('ETH/USDT', '5m')
cl_p.process_klines(df)

# Find the BI 191→213
target_bi = None
for bi in cl_p.bis:
    if bi.start.k.index == 191 and bi.end.k.index == 213:
        target_bi = bi
        break

if target_bi is None:
    # The split already happened, so look for the 3 sub-BIs
    print("BI 191→213 already split in pyarmor output")
    for bi in cl_p.bis:
        if 191 <= bi.start.k.index <= 213 or 191 <= bi.end.k.index <= 213:
            print(f"  Sub-BI: {bi.start.k.index}→{bi.end.k.index} ({bi.type})")

print(f"\nTotal BIs: {len(cl_p.bis)}")
print(f"Total FXs: {len(cl_p.fxs)}")

# Find internal FXs between 191 and 213
internal_fxs = [fx for fx in cl_p.fxs if 191 < fx.k.index < 213]
print(f"\nInternal FXs between 191 and 213 ({len(internal_fxs)}):")
for fx in internal_fxs:
    h = fx.high('fx_qj_k', 'fx_qy_three')
    l = fx.low('fx_qj_k', 'fx_qy_three')
    print(f"  FX({fx.k.index},{fx.type},val={fx.val:.2f},"
          f"ki={fx.k.k_index},h={h:.2f},l={l:.2f})")

# Also get bi start and end FXs
start_fx = None
end_fx = None
for fx in cl_p.fxs:
    if fx.k.index == 191:
        start_fx = fx
    if fx.k.index == 213:
        end_fx = fx

if start_fx:
    h = start_fx.high('fx_qj_k', 'fx_qy_three')
    l = start_fx.low('fx_qj_k', 'fx_qy_three')
    print(f"\nStart FX: FX({start_fx.k.index},{start_fx.type},val={start_fx.val:.2f},"
          f"ki={start_fx.k.k_index},h={h:.2f},l={l:.2f})")
if end_fx:
    h = end_fx.high('fx_qj_k', 'fx_qy_three')
    l = end_fx.low('fx_qj_k', 'fx_qy_three')
    print(f"End FX:   FX({end_fx.k.index},{end_fx.type},val={end_fx.val:.2f},"
          f"ki={end_fx.k.k_index},h={h:.2f},l={l:.2f})")

# Get all FXs from 191 to 213 inclusive
all_fxs_in_bi = [fx for fx in cl_p.fxs if 191 <= fx.k.index <= 213]
print(f"\nAll FXs in BI range ({len(all_fxs_in_bi)}):")
for fx in all_fxs_in_bi:
    h = fx.high('fx_qj_k', 'fx_qy_three')
    l = fx.low('fx_qj_k', 'fx_qy_three')
    print(f"  FX({fx.k.index},{fx.type},val={fx.val:.2f},"
          f"ki={fx.k.k_index},range=[{l:.2f},{h:.2f}])")

# Show raw K-lines in the BI's range
start_ki = start_fx.k.k_index if start_fx else 0
end_ki = end_fx.k.k_index if end_fx else 0
print(f"\nRaw K-lines from ki={start_ki} to ki={end_ki}:")
for ki in range(start_ki, end_ki + 1):
    k = cl_p.src_klines[ki]
    print(f"  ki={ki}: h={k.h:.2f}, l={k.l:.2f}")

# Now test different cross-counting hypotheses
print("\n" + "="*60)
print("CROSS-COUNTING HYPOTHESIS TESTING")
print("="*60)

# For each FX triplet, test different cross mechanisms
from itertools import combinations

fxs_for_triplets = all_fxs_in_bi  # Use all FXs in the BI

for ti in range(len(fxs_for_triplets) - 2):
    fx1 = fxs_for_triplets[ti]
    fx2 = fxs_for_triplets[ti + 1]
    fx3 = fxs_for_triplets[ti + 2]
    
    h1 = fx1.high('fx_qj_k', 'fx_qy_three')
    l1 = fx1.low('fx_qj_k', 'fx_qy_three')
    h2 = fx2.high('fx_qj_k', 'fx_qy_three')
    l2 = fx2.low('fx_qj_k', 'fx_qy_three')
    h3 = fx3.high('fx_qj_k', 'fx_qy_three')
    l3 = fx3.low('fx_qj_k', 'fx_qy_three')
    
    print(f"\nTriplet {ti}: FX({fx1.k.index},{fx1.type}) FX({fx2.k.index},{fx2.type}) FX({fx3.k.index},{fx3.type})")
    print(f"  Ranges: [{l1:.2f},{h1:.2f}] [{l2:.2f},{h2:.2f}] [{l3:.2f},{h3:.2f}]")
    
    # Hypothesis A: K-line overlaps intersection of all 3 ranges
    cross_low_a = max(l1, l2, l3)
    cross_high_a = min(h1, h2, h3)
    
    # Hypothesis B: K-line overlaps with each range individually (AND)
    # (same as A when we check the combined intersection)
    
    # Hypothesis C: K-line overlaps intersection of fx1 and fx3 (same type FXs only)
    cross_low_c = max(l1, l3)
    cross_high_c = min(h1, h3)
    
    # Count hits for each hypothesis with tolerance=1
    for hyp_name, cross_low, cross_high in [
        ("A: all-3-intersect", cross_low_a, cross_high_a),
        ("C: fx1&fx3-only", cross_low_c, cross_high_c),
    ]:
        miss = 0
        hit = 0
        for ki in range(start_ki, end_ki + 1):
            k = cl_p.src_klines[ki]
            # K-line [k.l, k.h] overlaps [cross_low, cross_high]
            if cross_high >= cross_low and k.h >= cross_low and k.l <= cross_high:
                hit += 1
                miss = 0
            else:
                miss += 1
            if miss > 1:  # tolerance = 1
                break
        print(f"  {hyp_name}: zone=[{cross_low:.2f},{cross_high:.2f}] hits={hit}")
    
    # Hypothesis D: overlaps with fx2's range only (middle FX)
    miss = 0
    hit = 0
    for ki in range(start_ki, end_ki + 1):
        k = cl_p.src_klines[ki]
        if k.h >= l2 and k.l <= h2:
            hit += 1
            miss = 0
        else:
            miss += 1
        if miss > 1:
            break
    print(f"  D: fx2-range-only: zone=[{l2:.2f},{h2:.2f}] hits={hit}")

    # Hypothesis E: K-line overlaps with ALL 3 individual ranges
    miss = 0
    hit = 0
    for ki in range(start_ki, end_ki + 1):
        k = cl_p.src_klines[ki]
        crosses_1 = k.h >= l1 and k.l <= h1
        crosses_2 = k.h >= l2 and k.l <= h2
        crosses_3 = k.h >= l3 and k.l <= h3
        if crosses_1 and crosses_2 and crosses_3:
            hit += 1
            miss = 0
        else:
            miss += 1
        if miss > 1:
            break
    print(f"  E: each-range-AND: hits={hit}")
