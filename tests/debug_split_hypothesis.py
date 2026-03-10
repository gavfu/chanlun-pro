"""Check: could pyarmor's sequence 291→299, 299→304, 304→309 come from 
_bi_special_bi_split of a larger BI?

Pyarmor has:
  bi[14]: down 291→299
  bi[15]: up 299→304  
  bi[16]: down 304→309
  bi[17]: up 309→330

What if pyarmor first builds:
  bi[A]: down 291→309 (or 291→315)
And then _bi_special_bi_split splits it into three?

For down 291→309:
  Split into: down 291→299, up 299→304, down 304→309
  This is exactly [bi14, bi15, bi16]!

Let's verify if down 291→309 would trigger splitting:
- threshold=20, tolerance=1
- Need at least 3 internal FXes
- Need 20+ K-line crosses with 3 consecutive FXes
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P
from chanlun.cl_interface import FX, BI

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
cd_o = CL_O("test", "test", config=CL_CONFIG)
cd_o.process_klines(df)

fxs = cd_o.get_fxs()
qj = cd_o.fx_qj; qy = cd_o.fx_qy

# Find ding@291
ding291 = None
for fx in fxs:
    if fx.k.k_index == 291 and fx.type == "ding": ding291 = fx; break

# Find possible end FXes
di_candidates = []
for fx in fxs:
    if fx.type == "di" and 295 <= fx.k.k_index <= 320:
        di_candidates.append(fx)
        print(f"  di@{fx.k.k_index} (ck_idx={fx.k.index}) val={fx.val:.2f}")

# Check ding@291 structure
print(f"\nding@291: ck_idx={ding291.k.index}, val={ding291.val:.2f}")

# For a hypothetical bi: down 291→309
# start = ding@291, end = di@309
di309 = None
for fx in fxs:
    if fx.k.k_index == 309 and fx.type == "di": di309 = fx; break

if ding291 and di309:
    print(f"\n=== Hypothetical: down 291→309 ===")
    print(f"  start=ding@291 (ck={ding291.k.index}), end=di@309 (ck={di309.k.index})")
    
    # Internal FXes between ck_idx of start and end
    start_ck = ding291.k.index
    end_ck = di309.k.index
    internal = [fx for fx in fxs if start_ck < fx.k.index < end_ck]
    print(f"  Internal FXes: {len(internal)}")
    for fx in internal:
        print(f"    {fx.type:4s} k={fx.k.k_index} (ck={fx.k.index}) val={fx.val:.2f} "
              f"h={fx.high(qj,qy):.2f} l={fx.low(qj,qy):.2f}")
    
    if len(internal) >= 3:
        # Check K-line cross counting for triplets
        end_ki = di309.k.k_index
        for ti in range(len(internal) - 2):
            fx1 = internal[ti]
            fx2 = internal[ti + 1]
            fx3 = internal[ti + 2]
            
            h1, l1 = fx1.high(qj, qy), fx1.low(qj, qy)
            h2, l2 = fx2.high(qj, qy), fx2.low(qj, qy)
            h3, l3 = fx3.high(qj, qy), fx3.low(qj, qy)
            
            hit_count = 0
            miss_count = 0
            
            for ki in range(fx1.k.k_index, end_ki):
                k = cd_o.src_klines[ki]
                if (k.h >= l1 and k.l <= h1
                        and k.h >= l2 and k.l <= h2
                        and k.h >= l3 and k.l <= h3):
                    hit_count += 1
                    miss_count = 0
                else:
                    miss_count += 1
                if miss_count > 1:  # tolerance=1
                    break
            
            print(f"\n  Triplet [{ti}]: {fx1.type}@{fx1.k.k_index}, "
                  f"{fx2.type}@{fx2.k.k_index}, {fx3.type}@{fx3.k.k_index}")
            print(f"    hit_count={hit_count} (threshold=20)")
            print(f"    Would trigger: {'YES' if hit_count >= 20 else 'NO'}")

# Check also: down 291→315
print(f"\n\n=== What about down 291→315? ===")
di315 = None
for fx in fxs:
    if fx.k.k_index == 315 and fx.type == "di": di315 = fx; break

if ding291 and di315:
    start_ck = ding291.k.index
    end_ck = di315.k.index
    internal = [fx for fx in fxs if start_ck < fx.k.index < end_ck]
    print(f"  Internal FXes: {len(internal)}")
    for fx in internal:
        print(f"    {fx.type:4s} k={fx.k.k_index} (ck={fx.k.index}) val={fx.val:.2f}")
    
    if len(internal) >= 3:
        end_ki = di315.k.k_index
        for ti in range(len(internal) - 2):
            fx1 = internal[ti]
            fx2 = internal[ti + 1]
            fx3 = internal[ti + 2]
            
            h1, l1 = fx1.high(qj, qy), fx1.low(qj, qy)
            h2, l2 = fx2.high(qj, qy), fx2.low(qj, qy)
            h3, l3 = fx3.high(qj, qy), fx3.low(qj, qy)
            
            hit_count = 0
            miss_count = 0
            
            for ki in range(fx1.k.k_index, end_ki):
                k = cd_o.src_klines[ki]
                if (k.h >= l1 and k.l <= h1
                        and k.h >= l2 and k.l <= h2
                        and k.h >= l3 and k.l <= h3):
                    hit_count += 1
                    miss_count = 0
                else:
                    miss_count += 1
                if miss_count > 1:
                    break
            
            print(f"\n  Triplet [{ti}]: {fx1.type}@{fx1.k.k_index}, "
                  f"{fx2.type}@{fx2.k.k_index}, {fx3.type}@{fx3.k.k_index}")
            print(f"    hit_count={hit_count}")
            if hit_count >= 20:
                print(f"    *** TRIGGERED! ***")
                # Now check what split would produce
                if ding291.type == "ding":
                    # Down bi: ding→di. Split is: down→up→down
                    # Split1 = di (from triplet), Split2 = ding (after split1)
                    print(f"    For down BI split: split1=di, split2=ding")
                    # Find di candidates in/around triplet
                    split1_type = "di"
                    split2_type = "ding"
                    candidates_s1 = [fx for fx in [fx1,fx2,fx3] if fx.type == split1_type]
                    for s1 in candidates_s1:
                        gap_ok = s1.k.index - ding291.k.index >= 4
                        print(f"      split1 candidate: di@{s1.k.k_index} "
                              f"gap={s1.k.index - ding291.k.index} {'OK' if gap_ok else 'FAIL'}")
                    # Find split2 candidates after split1
                    for s1 in candidates_s1:
                        s2_cands = [fx for fx in internal if fx.type == split2_type 
                                    and fx.k.index > s1.k.index]
                        for s2 in s2_cands:
                            gap_mid = s2.k.index - s1.k.index
                            print(f"      split1=di@{s1.k.k_index} → split2=ding@{s2.k.k_index} "
                                  f"gap_mid={gap_mid}")
                break
