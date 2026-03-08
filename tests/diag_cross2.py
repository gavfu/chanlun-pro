# -*- coding: utf-8 -*-
"""
Count overlapping CLKlines within the stroke 315→370.
A cross/overlapping CLKline is one where it shares a price range with the previous.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "test_data", "BTC_USDT_60m_500.parquet"))

from chanlun.cl_open import CL
cd = CL("BTC/USDT", "60m")
cd.process_klines(df)

# CLKlines from 315 to 370
print("=== CLKline overlap count (315-370) ===")
cross_count = 0
no_cross_count = 0
cross_indices = []
max_consecutive_non_cross = 0
cur_non_cross = 0

for i in range(316, 371):  # Check each CLKline against the previous
    prev = cd.cl_klines[i-1]
    cur = cd.cl_klines[i]
    
    # Overlap: ranges (prev.l, prev.h) and (cur.l, cur.h) intersect
    overlap = not (prev.h < cur.l or cur.h < prev.l)
    # More specifically, "cross" might mean inclusion (one contains the other)
    # Or it could mean any overlap
    
    # Let's check both
    inclusion = (prev.h >= cur.h and prev.l <= cur.l) or (cur.h >= prev.h and cur.l <= prev.l)
    any_overlap = not (prev.h < cur.l or cur.h < prev.l)
    
    if any_overlap:
        cross_count += 1
        cross_indices.append(i)
        if cur_non_cross > max_consecutive_non_cross:
            max_consecutive_non_cross = cur_non_cross
        cur_non_cross = 0
    else:
        no_cross_count += 1
        cur_non_cross += 1

if cur_non_cross > max_consecutive_non_cross:
    max_consecutive_non_cross = cur_non_cross

print(f"Total CLKlines: {371-316+1}")
print(f"Overlapping: {cross_count}")
print(f"Non-overlapping: {no_cross_count}")
print(f"Max consecutive non-overlap: {max_consecutive_non_cross}")

# Count overlapping using the inclusion method (strict overlap)
print(f"\n=== Using strict inclusion ===")
incl_count = 0
for i in range(316, 371):
    prev = cd.cl_klines[i-1]
    cur = cd.cl_klines[i]
    inclusion = (prev.h >= cur.h and prev.l <= cur.l) or (cur.h >= prev.h and cur.l <= prev.l)
    if inclusion:
        incl_count += 1

print(f"Inclusion count: {incl_count}")

# Check the stroke 315→347 specifically
print(f"\n=== CLKline overlap count (315-347) ===")
cross_315_347 = 0
for i in range(316, 348):
    prev = cd.cl_klines[i-1]
    cur = cd.cl_klines[i]
    overlap = not (prev.h < cur.l or cur.h < prev.l)
    if overlap:
        cross_315_347 += 1
print(f"Overlapping: {cross_315_347}")

# Check the stroke 347→355
print(f"\n=== CLKline overlap count (347-355) ===")
cross_347_355 = 0
for i in range(348, 356):
    prev = cd.cl_klines[i-1]
    cur = cd.cl_klines[i]
    overlap = not (prev.h < cur.l or cur.h < prev.l)
    if overlap:
        cross_347_355 += 1
print(f"Overlapping: {cross_347_355}")

# Check what the original K-line cross looks like
# Original K-lines mapped by CLKline
print(f"\n=== K-line details (CLKlines 340-370) ===")
for i in range(340, 371):
    ck = cd.cl_klines[i]
    k_indices = [k.index for k in ck.klines]
    print(f"  CLK[{i:3d}] h={ck.h:10.1f} l={ck.l:10.1f} klines={k_indices}")
