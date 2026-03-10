"""Verify if _bi_special_bi_split splits the up 299→330 BI in BTC60."""
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

# Store original _bi_special_bi_split
orig_split = cd._bi_special_bi_split.__func__
import types

def debug_split(self, bis):
    print(f"=== _bi_special_bi_split called with {len(bis)} BIs ===")
    # Show BIs around our region
    for j, b in enumerate(bis):
        if 290 <= b.start.k.k_index <= 340 or 290 <= b.end.k.k_index <= 340:
            print(f"  Input bi[{j}]: {b.type} {b.start.k.k_index}→{b.end.k.k_index}")
    
    result = orig_split(self, bis)
    
    # Show result around our region 
    for j, b in enumerate(result):
        if 290 <= b.start.k.k_index <= 340 or 290 <= b.end.k.k_index <= 340:
            print(f"  Output bi[{j}]: {b.type} {b.start.k.k_index}→{b.end.k.k_index}")
    
    return result

cd._bi_special_bi_split = types.MethodType(debug_split, cd)
cd.process_klines(df)

# Also examine the BI that contains 299→330 range
print(f"\n=== Pre-split BIs from _build_bis ===")
# Re-run without the monkey-patch to get raw BIs
cd2 = CL_O("test", "test", config=CL_CONFIG)
# Disable split to see raw BIs
cd2.bi_split_k_cross_nums = 0
cd2.process_klines(df)
bis_raw = cd2.get_bis()
for j, b in enumerate(bis_raw):
    if 250 <= b.start.k.k_index <= 360:
        start_idx = b.start.k.index
        end_idx = b.end.k.index
        internal_fxs = [fx for fx in cd2.get_fxs() 
                       if start_idx < fx.k.index < end_idx]
        print(f"  raw bi[{j}]: {b.type} {b.start.k.k_index}→{b.end.k.k_index} "
              f"(cl_idx {start_idx}→{end_idx}) internal_fxs={len(internal_fxs)}")

# Now check the specific BI that gets split
# Find the BI containing k_index 299→330
for j, b in enumerate(bis_raw):
    if b.start.k.k_index == 299:
        print(f"\n=== Examining raw bi[{j}]: {b.type} {b.start.k.k_index}→{b.end.k.k_index} ===")
        start_idx = b.start.k.index
        end_idx = b.end.k.index
        
        # Get internal fxs
        internal_fxs = [fx for fx in cd2.get_fxs()
                       if start_idx < fx.k.index < end_idx]
        print(f"Internal FXes ({len(internal_fxs)}):")
        for fx in internal_fxs:
            print(f"  {fx.type}@{fx.k.k_index} (cl_idx={fx.k.index}) val={fx.val:.2f}")
        
        # Check triplet hit counts
        print(f"\nTriplet hit counts:")
        qj = "fx_qj_k"; qy = "fx_qy_three"
        for ti in range(len(internal_fxs) - 2):
            fx1 = internal_fxs[ti]
            fx2 = internal_fxs[ti + 1]
            fx3 = internal_fxs[ti + 2]
            
            h1, l1 = fx1.high(qj, qy), fx1.low(qj, qy)
            h2, l2 = fx2.high(qj, qy), fx2.low(qj, qy)
            h3, l3 = fx3.high(qj, qy), fx3.low(qj, qy)
            
            hit_count = 0
            miss_count = 0
            for ki in range(fx1.k.k_index, b.end.k.k_index):
                k = cd2.src_klines[ki]
                if (k.h >= l1 and k.l <= h1
                        and k.h >= l2 and k.l <= h2
                        and k.h >= l3 and k.l <= h3):
                    hit_count += 1
                    miss_count = 0
                else:
                    miss_count += 1
                if miss_count > 1:  # tolerance=1
                    break
            
            print(f"  triplet[{ti}]: {fx1.type}@{fx1.k.k_index}, {fx2.type}@{fx2.k.k_index}, "
                  f"{fx3.type}@{fx3.k.k_index} → hit={hit_count} "
                  f"{'*** TRIGGER ***' if hit_count >= 20 else ''}")
        break
