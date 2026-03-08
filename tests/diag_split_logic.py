# -*- coding: utf-8 -*-
"""
Test bi_split implementation logic.
For a DOWN stroke 315→370:
1. Count FX cross pairs from start to end
2. When count reaches threshold (20), split at that point
3. Find the split sub-strokes

For DOWN stroke: split produces [down start→split_low, up split_low→split_ding, down split_ding→end]
where split_low is the most extreme di BEFORE or AT the split point
and split_ding is the FX at the split threshold count
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "test_data", "BTC_USDT_60m_500.parquet"))

from chanlun.cl_open import CL
cd = CL("BTC/USDT", "60m")
cd.process_klines(df)

fx_by_idx = {fx.k.index: fx for fx in cd.fxs}

# For bi[24] down 315→370:
# FX in range with types
fxs_in_range = [fx for fx in cd.fxs if 315 <= fx.k.index <= 370]

qj = "fx_qj_k"
qy = "fx_qy_three"

# Count crosses with the "allow 1 non-consecutive" rule
cross_count = 0
non_cross_allowed = 1  # from config "20,1" 
non_cross_seen = 0
split_fx = None

for i in range(len(fxs_in_range) - 1):
    a = fxs_in_range[i]
    b = fxs_in_range[i+1]
    
    ha = a.high(qj, qy)
    la = a.low(qj, qy)
    hb = b.high(qj, qy)
    lb = b.low(qj, qy)
    
    overlap = not (ha < lb or hb < la)
    
    if overlap:
        cross_count += 1
    else:
        non_cross_seen += 1
        # If we've exceeded the allowed non-consecutive count, 
        # does the count continue or reset?
        # Based on the result matching 20 at ding@355, it seems to continue
        cross_count += 1  # Count the non-cross too? No...
        
    if cross_count >= 20 and split_fx is None:
        split_fx = b
        print(f"Split threshold reached at {b.type}@{b.k.index}")
        break

# Now let me try simpler: just count ALL pairs (cross or not)
print(f"\nAlternative: just count overlapping pairs")
cross_only = 0
for i in range(len(fxs_in_range) - 1):
    a = fxs_in_range[i]
    b = fxs_in_range[i+1]
    ha = a.high(qj, qy)
    la = a.low(qj, qy)
    hb = b.high(qj, qy)
    lb = b.low(qj, qy)
    overlap = not (ha < lb or hb < la)
    if overlap:
        cross_only += 1
        if cross_only == 20:
            print(f"Pure cross count 20 at {b.type}@{b.k.index}")
            break

# The answer: the 20th CROSS pair ends at ding@355
# Now find the split point

# For DOWN stroke, the internal structure is:
# 1. Find the most extreme (lowest) di within the FX range up to the split point
# 2. That becomes the end of the first sub-stroke
# 3. The sub-stroke goes: start (ding@315) → that lowest di
# 4. Then up from that di to the split ding
# 5. Then down from split ding to original end (di@370)

print(f"\nFinding lowest di from 315 to 355:")
lowest_di = None
for fx in fxs_in_range:
    if fx.k.index > 355:
        break
    if fx.type == "di":
        if lowest_di is None or fx.val < lowest_di.val:
            lowest_di = fx

if lowest_di:
    print(f"  Lowest di: di@{lowest_di.k.index} val={lowest_di.val:.1f}")

# But the result shows bi[24]=down 315→347, so the lowest di up to ding@355 is di@347 (val=67712.3)
# Let's verify: di@347.val=67712.3 is indeed the lowest di between 315 and 355
print(f"\nAll di's between 315 and 355:")
for fx in fxs_in_range:
    if fx.k.index > 355:
        break
    if fx.type == "di":
        print(f"  di@{fx.k.index} val={fx.val:.1f}")
