"""Detailed trace of _bi_fx_valid for BTC60 bi[15] divergence."""
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
cd_o = CL_O("test", "test", config=CL_CONFIG)
cd_o.process_klines(df)

fxs = cd_o.get_fxs()

# Find di@299 and ding@304
di299 = None
ding304 = None
for fx in fxs:
    if fx.type == "di" and fx.k.k_index == 299:
        di299 = fx
    if fx.type == "ding" and fx.k.k_index == 304:
        ding304 = fx

if di299 and ding304:
    qj = cd_o.fx_qj
    qy = cd_o.fx_qy
    
    cl_gap = ding304.k.index - di299.k.index
    k_gap = ding304.k.k_index - di299.k.k_index
    print(f"di@299 → ding@304:")
    print(f"  cl_gap={cl_gap}, k_gap={k_gap}")
    print(f"  bi_type={cd_o.bi_type}")
    print(f"  fx_check_k_nums={cd_o.fx_check_k_nums}")
    print(f"  allow_bi_fx_strict={cd_o.allow_bi_fx_strict}")
    print()
    
    # cl_gap check: bi_type_old needs cl_gap >= 4
    print(f"  Step 1: cl_gap < 4? {cl_gap} < 4 = {cl_gap < 4}")
    
    # k_gap < fx_check_k_nums check
    print(f"  Step 2: k_gap < fx_check_k_nums? {k_gap} < {cd_o.fx_check_k_nums} = {k_gap < cd_o.fx_check_k_nums}")
    
    # Strict check for up BI (di→ding)
    print(f"\n  Up BI strict check (di→ding):")
    print(f"    di299.high({qj},{qy}) = {di299.high(qj, qy):.2f}")
    print(f"    ding304.high({qj},{qy}) = {ding304.high(qj, qy):.2f}")
    print(f"    di299.high > ding304.high? {di299.high(qj, qy)} > {ding304.high(qj, qy)} = {di299.high(qj, qy) > ding304.high(qj, qy)}")
    
    print(f"    ding304.low({qj},{qy}) = {ding304.low(qj, qy):.2f}")
    print(f"    di299.low({qj},{qy}) = {di299.low(qj, qy):.2f}")
    print(f"    ding304.low < di299.low? {ding304.low(qj, qy)} < {di299.low(qj, qy)} = {ding304.low(qj, qy) < di299.low(qj, qy)}")
    
    # Show all klines in the FXes to understand the intervals
    print(f"\n  di@299 klines (raw):")
    for kl in di299.klines:
        print(f"    k_index={kl.k_index}, h={kl.h:.2f}, l={kl.l:.2f}")
    print(f"  di@299.k klines (merged k sub-klines):")
    for kl in di299.k.klines:
        print(f"    k_index={kl.k_index}, h={kl.h:.2f}, l={kl.l:.2f}")
    
    print(f"\n  ding@304 klines (raw):")
    for kl in ding304.klines:
        print(f"    k_index={kl.k_index}, h={kl.h:.2f}, l={kl.l:.2f}")
    print(f"  ding@304.k klines (merged k sub-klines):")
    for kl in ding304.k.klines:
        print(f"    k_index={kl.k_index}, h={kl.h:.2f}, l={kl.l:.2f}")
    
    # The actual high/low computation with fx_qj_k and fx_qy_three
    # means fx.high() = max of all raw k-lines across all 3 merged klines
    # and fx.low() = min of all raw k-lines across all 3 merged klines
    all_raw_di = []
    for kl in di299.klines:
        for sub in kl.klines:
            all_raw_di.append(sub)
    print(f"\n  All raw K-lines in di@299:")
    for kl in all_raw_di:
        print(f"    raw k_index={kl.k_index} h={kl.h:.2f} l={kl.l:.2f}")
    max_h_di = max(kl.h for kl in all_raw_di)
    min_l_di = min(kl.l for kl in all_raw_di)
    print(f"  -> max_h={max_h_di:.2f}, min_l={min_l_di:.2f}")
    
    all_raw_ding = []
    for kl in ding304.klines:
        for sub in kl.klines:
            all_raw_ding.append(sub)
    print(f"\n  All raw K-lines in ding@304:")
    for kl in all_raw_ding:
        print(f"    raw k_index={kl.k_index} h={kl.h:.2f} l={kl.l:.2f}")
    max_h_ding = max(kl.h for kl in all_raw_ding)
    min_l_ding = min(kl.l for kl in all_raw_ding)
    print(f"  -> max_h={max_h_ding:.2f}, min_l={min_l_ding:.2f}")
    
    print(f"\n  VERDICT: For up BI (di→ding):")
    print(f"    di.high ({max_h_di:.2f}) > ding.high ({max_h_ding:.2f})? = {max_h_di > max_h_ding}")
    if max_h_di > max_h_ding:
        print(f"    -> FAILS strict check! BI rejected.")
    print(f"    ding.low ({min_l_ding:.2f}) < di.low ({min_l_di:.2f})? = {min_l_ding < min_l_di}")
    if min_l_ding < min_l_di:
        print(f"    -> FAILS strict check! BI rejected.")
