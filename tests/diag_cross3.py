# -*- coding: utf-8 -*-
"""
Count consecutive overlapping CLKlines and find where count reaches 20.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "test_data", "BTC_USDT_60m_500.parquet"))

from chanlun.cl_open import CL
cd = CL("BTC/USDT", "60m")
cd.process_klines(df)

# The stroke is down 315 → 370
# But pyarmor first gets bi[24]=down 315→370 from _bi_check
# Then _bi_special_bi_split kicks in

# Let's check: how does bi_special_bi_split use the cross count?
# It checks FX pairs, not CLKline pairs

# Let me recount using FX-based check
fx_by_idx = {fx.k.index: fx for fx in cd.fxs}
fx_in_range = [fx for fx in cd.fxs if 315 <= fx.k.index <= 370]
fx_indices_in_range = [fx.k.index for fx in fx_in_range]

qj = "fx_qj_k"
qy = "fx_qy_three"

print("=== FX cross/overlap scan ===")
cross_count = 0
non_cross_streak = 0  # consecutive non-cross
last_cross_fx = None

for i in range(len(fx_in_range) - 1):
    a = fx_in_range[i]
    b = fx_in_range[i+1]
    
    ha = a.high(qj, qy)
    la = a.low(qj, qy)
    hb = b.high(qj, qy)
    lb = b.low(qj, qy)
    
    # Check if the FX ranges overlap (cross)
    overlap = not (ha < lb or hb < la)
    
    if overlap:
        cross_count += 1
        non_cross_streak = 0
        last_cross_fx = b
        status = f"cross (total={cross_count})"
    else:
        non_cross_streak += 1
        status = f"NO-CROSS (streak={non_cross_streak})"
    
    a_idx = a.k.index
    b_idx = b.k.index
    print(f"  {a.type}@{a_idx:3d} → {b.type}@{b_idx:3d}: [{la:.0f}, {ha:.0f}] vs [{lb:.0f}, {hb:.0f}] = {status}")
    
    if cross_count == 20:
        print(f"\n  *** CROSS COUNT REACHED 20 at {b.type}@{b_idx} ***")

# Now let's check: what is the NON-FX based count?
# Maybe it counts CLKline pairs?
print(f"\n=== CLKline cross scan (315-370) ===")
cross_count_ck = 0
for i in range(316, 371):
    prev = cd.cl_klines[i-1]
    cur = cd.cl_klines[i]
    overlap = not (prev.h < cur.l or cur.h < prev.l)
    if overlap:
        cross_count_ck += 1
        if cross_count_ck == 20:
            print(f"  *** CLK CROSS COUNT REACHED 20 at CLK[{i}] ***")
    status = f"cross({cross_count_ck})" if overlap else f"no-cross"
    if 340 <= i <= 360:  # Show details around the split area
        print(f"  CLK[{i-1}]→CLK[{i}]: [{prev.l:.0f},{prev.h:.0f}] vs [{cur.l:.0f},{cur.h:.0f}] = {status}")

print(f"\n  Total CLK crosses: {cross_count_ck}")

# Also count from AFTER the initial non-overlapping section
# Find where consecutive overlaps start
print(f"\n=== Finding longest consecutive overlap run ===")
runs = []
cur_start = None
cur_count = 0
for i in range(316, 371):
    prev = cd.cl_klines[i-1]
    cur = cd.cl_klines[i]
    overlap = not (prev.h < cur.l or cur.h < prev.l)
    if overlap:
        if cur_start is None:
            cur_start = i
        cur_count += 1
    else:
        if cur_count > 0:
            runs.append((cur_start, i-1, cur_count))
        cur_start = None
        cur_count = 0
if cur_count > 0:
    runs.append((cur_start, 370, cur_count))

for start, end, count in runs:
    print(f"  CLK[{start-1}]→CLK[{end}]: {count} consecutive overlaps")
