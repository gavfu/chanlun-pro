"""
Verify k_index of merged K-line 45 for ETH5m.
Show the raw K-lines that get merged into it.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

CL_CONFIG = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
}

df = pd.read_parquet("tests/test_data/ETH_USDT_5m_1000.parquet")
cd_o = CL_O("test", "test", config=CL_CONFIG)
cd_o.process_klines(df)
cd_p = CL_P("test", "test", config=CL_CONFIG)
cd_p.process_klines(df)

cklines_o = cd_o.get_cl_klines()
cklines_p = cd_p.get_cl_klines()

# Check k_index for ALL merged klines
print("=== ALL merged K-lines with n>0 (merged from multiple raws) ===")
print(f"{'idx':>5} {'dir':>5} {'p_kidx':>7} {'rule':>10} raw_indices")
for i in range(min(len(cklines_o), len(cklines_p))):
    cp = cklines_p[i]
    if cp.n == 0:
        continue
    raw_indices = [k.index for k in cp.klines]
    # Determine ACTUAL merge direction: compare current K-line with merged[-2]
    if i >= 2:
        prev_prev = cklines_p[i - 1]  # This is merged[-2] at time of first containment
        # At the time of first containment, last.h is the first raw K-line's h
        first_raw_h = cp.klines[0].h
        direction = "up" if first_raw_h >= prev_prev.h else "down"
    elif i == 1:
        # When only 1 previous merged K, direction is inferred from current vs up_qs
        direction = "up"
    else:
        direction = "?"
    
    # What rule does pyarmor follow?
    first_idx = raw_indices[0]
    last_idx = raw_indices[-1]
    p_kidx = cp.k_index
    
    if p_kidx == first_idx:
        rule = "FIRST"
    elif p_kidx == last_idx:
        rule = "LAST"
    else:
        rule = f"MID({p_kidx})"
    
    # Find which raw K-line defines the direction-relevant extreme
    # For "up": max h; for "down": min l
    if direction == "up":
        extremum_idx = max(cp.klines, key=lambda k: k.h).index
        extremum_type = "max_h"
    elif direction == "down":
        extremum_idx = min(cp.klines, key=lambda k: k.l).index
        extremum_type = "min_l"
    else:
        extremum_idx = -1
        extremum_type = "?"
    
    extremum_match = "✓" if p_kidx == extremum_idx else "✗"
    
    match = "✅" if cklines_o[i].k_index == cp.k_index else "❌"
    
    print(f"  {match} ck[{i:>4}] dir={direction:>4} p_kidx={p_kidx:>4} rule={rule:>6} "
          f"raws={raw_indices} {extremum_type}=raw[{extremum_idx}]{extremum_match}")

# Collect all differing indices dynamically
diff_indices = []
for i in range(min(len(cklines_o), len(cklines_p))):
    co = cklines_o[i]
    cp = cklines_p[i]
    if co.k_index != cp.k_index:
        diff_indices.append(i)

# Show full details for all differing cases
for diff_idx in diff_indices:
    print(f"\n=== ck[{diff_idx}] details ===")
    co = cklines_o[diff_idx]
    cp = cklines_p[diff_idx]
    # Show all h/l for context of prev 3 cklines
    for ctx in range(max(0, diff_idx - 3), diff_idx):
        ctx_o = cklines_o[ctx]
        print(f"  ck[{ctx}]: h={ctx_o.h:.2f} l={ctx_o.l:.2f} k_index={ctx_o.k_index} n={ctx_o.n}")
    # current merged ck direction
    if diff_idx >= 2:
        prev_o = cklines_o[diff_idx - 1]
        prev2 = cklines_o[diff_idx - 2]
        direction = "up" if prev_o.h >= prev2.h else "down"
    else:
        direction = "up"
    print(f"  --> ck[{diff_idx}]: direction={direction}")
    print(f"  Open:    k_index={co.k_index} h={co.h} l={co.l} n={co.n}")
    print(f"  Pyarmor: k_index={cp.k_index} h={cp.h} l={cp.l} n={cp.n}")
    
    # Show raw K-lines with direction-aware merge trace
    print(f"  Raw K-lines (merge direction={direction}):")
    merged_h = None
    merged_l = None
    for i, k in enumerate(co.klines):
        pk = cp.klines[i]
        if i == 0:
            merged_h = k.h
            merged_l = k.l
            print(f"    [{i}] raw[{k.index}] h={k.h:.2f} l={k.l:.2f}  INITIAL")
        else:
            # Direction-based merged values  
            if direction == "up":
                new_merged_h = max(merged_h, k.h)
                new_merged_l = max(merged_l, k.l)
            else:
                new_merged_h = min(merged_h, k.h)
                new_merged_l = min(merged_l, k.l)
            h_same = new_merged_h == merged_h
            l_same = new_merged_l == merged_l
            # raw kline vs PREVIOUS raw kline (not merged)
            prev_k = co.klines[i-1]
            raw_contain = k.h >= prev_k.h and k.l <= prev_k.l
            raw_strict = k.h > prev_k.h and k.l < prev_k.l
            # vs merged
            m_contain = k.h >= merged_h and k.l <= merged_l
            m_strict = k.h > merged_h and k.l < merged_l
            print(f"    [{i}] raw[{k.index}] h={k.h:.2f} l={k.l:.2f}  "
                  f"vs_prev(cont={raw_contain} strict={raw_strict}) "
                  f"vs_merged(cont={m_contain} strict={m_strict}) "
                  f"after_merge h={new_merged_h:.2f}({'same' if h_same else 'CHG'}) l={new_merged_l:.2f}({'same' if l_same else 'CHG'})")
            merged_h = new_merged_h
            merged_l = new_merged_l
