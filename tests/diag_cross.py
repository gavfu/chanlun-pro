# -*- coding: utf-8 -*-
"""
Check FX inclusion relationships between fractals in the 315-370 range.
Understand bi_split mechanism.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "test_data", "BTC_USDT_60m_500.parquet"))

from chanlun.cl_open import CL
cd = CL("BTC/USDT", "60m")
cd.process_klines(df)

fx_by_idx = {fx.k.index: fx for fx in cd.fxs}

# Get consecutive FX indices in range
fx_indices = sorted([fx.k.index for fx in cd.fxs if 315 <= fx.k.index <= 375])

qj = "fx_qj_k"
qy = "fx_qy_three"

print("=== FX inclusion check (adjacent pairs) ===")
print(f"{'FX_A':>6} {'FX_B':>6} {'h_A':>10} {'l_A':>10} {'h_B':>10} {'l_B':>10} {'A⊃B':>5} {'B⊃A':>5} {'overlap':>7}")

cross_count = 0
non_cross_count = 0
overlaps = []

for i in range(len(fx_indices) - 1):
    a_idx = fx_indices[i]
    b_idx = fx_indices[i+1]
    a = fx_by_idx[a_idx]
    b = fx_by_idx[b_idx]
    
    ha = a.high(qj, qy)
    la = a.low(qj, qy)
    hb = b.high(qj, qy)
    lb = b.low(qj, qy)
    
    # A includes B: ha >= hb and la <= lb
    a_inc_b = ha >= hb and la <= lb
    # B includes A: hb >= ha and lb <= la
    b_inc_a = hb >= ha and lb <= la
    # Overlap: ranges [la, ha] and [lb, hb] intersect but don't fully include
    overlap = not (ha < lb or hb < la)  # ranges intersect
    
    tag = ""
    if a_inc_b:
        tag = "A⊃B"
        cross_count += 1
    elif b_inc_a:
        tag = "B⊃A"
        cross_count += 1
    elif overlap:
        tag = "cross"
        cross_count += 1
    else:
        tag = "none"
        non_cross_count += 1
    
    overlaps.append((a_idx, b_idx, tag))
    
    print(f"{a.type}@{a_idx:3d} {b.type}@{b_idx:3d} {ha:10.1f} {la:10.1f} {hb:10.1f} {lb:10.1f} {str(a_inc_b):>5} {str(b_inc_a):>5} {tag:>7}")

print(f"\nTotal pairs: {len(overlaps)}")
print(f"Cross/overlap: {cross_count}")
print(f"Non-cross: {non_cross_count}")

# Check consecutive cross runs
print(f"\n=== Consecutive cross runs ===")
run_start = None
run_count = 0
for i, (a, b, tag) in enumerate(overlaps):
    if tag != "none":
        if run_start is None:
            run_start = a
        run_count += 1
    else:
        if run_count > 0:
            print(f"  Run: {run_start} to {overlaps[i-1][1]}, count={run_count}")
        run_start = None
        run_count = 0
if run_count > 0:
    print(f"  Run: {run_start} to {overlaps[-1][1]}, count={run_count}")

# Also check K-line cross (adjacent FX K-lines overlap in original K-lines)
print(f"\n=== K-line overlap check ===")
for i in range(len(fx_indices) - 1):
    a_idx = fx_indices[i]
    b_idx = fx_indices[i+1]
    a = fx_by_idx[a_idx]
    b = fx_by_idx[b_idx]
    
    # Original K-line ranges
    a_klines = []
    for elem in [a.elements[0] if hasattr(a, 'elements') else None, a.k, a.elements[2] if hasattr(a, 'elements') else None]:
        if elem:
            a_klines.extend([k.k_index for k in elem.klines])
    b_klines = []
    for elem in [b.elements[0] if hasattr(b, 'elements') else None, b.k, b.elements[2] if hasattr(b, 'elements') else None]:
        if elem:
            b_klines.extend([k.k_index for k in elem.klines])
    
    if a_klines and b_klines:
        a_min, a_max = min(a_klines), max(a_klines)
        b_min, b_max = min(b_klines), max(b_klines)
        overlap = not (a_max < b_min or b_max < a_min)
        if overlap:
            print(f"  {a.type}@{a_idx} k[{a_min}-{a_max}] ∩ {b.type}@{b_idx} k[{b_min}-{b_max}] = OVERLAP")
