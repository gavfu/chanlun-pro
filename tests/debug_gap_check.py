"""Check cl_gap vs k_gap for all divergent BI pairs across BTC60, BTC5m, ETH5m."""
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

datasets = [
    ("BTC60", "tests/test_data/BTC_USDT_60m_1000.parquet"),
    ("BTC5m", "tests/test_data/BTC_USDT_5m_1000.parquet"),
    ("ETH5m", "tests/test_data/ETH_USDT_5m_1000.parquet"),
]

for name, path in datasets:
    print(f"\n{'='*60}")
    print(f" {name}")
    print(f"{'='*60}")
    
    df = pd.read_parquet(path)
    cd_o = CL_O("test", "test", config=CL_CONFIG)
    cd_o.process_klines(df)
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    
    bis_o = cd_o.get_bis()
    bis_p = cd_p.get_bis()
    fxs_o = cd_o.get_fxs()
    
    # Find first divergence point
    first_diff = None
    for i in range(min(len(bis_o), len(bis_p))):
        bo = bis_o[i]
        bp = bis_p[i]
        if bo.start.k.k_index != bp.start.k.k_index or bo.end.k.k_index != bp.end.k.k_index:
            first_diff = i
            break
    
    if first_diff is None:
        if len(bis_o) != len(bis_p):
            print(f"  Same boundaries for first {min(len(bis_o), len(bis_p))} BIs, but counts differ: {len(bis_o)} vs {len(bis_p)}")
        else:
            print(f"  ALL {len(bis_o)} BIs match perfectly!")
        continue
    
    print(f"  First divergence at bi[{first_diff}]")
    print(f"  Open  BI count: {len(bis_o)}")
    print(f"  Pyarmor BI count: {len(bis_p)}")
    
    # Show divergent BIs and surrounding context
    start = max(0, first_diff - 2)
    end = min(min(len(bis_o), len(bis_p)), first_diff + 5)
    
    print(f"\n  --- Open BIs [{start}:{end}] ---")
    for i in range(start, min(end, len(bis_o))):
        bi = bis_o[i]
        cl_gap = bi.end.k.index - bi.start.k.index
        k_gap = bi.end.k.k_index - bi.start.k.k_index
        marker = " <<<" if i == first_diff else ""
        print(f"    bi[{bi.index:>2}] {bi.type:>4} k={bi.start.k.k_index}→{bi.end.k.k_index} cl_gap={cl_gap} k_gap={k_gap}{marker}")
    
    print(f"\n  --- Pyarmor BIs [{start}:{end}] ---")
    for i in range(start, min(end, len(bis_p))):
        bi = bis_p[i]
        cl_gap = bi.end.k.index - bi.start.k.index
        k_gap = bi.end.k.k_index - bi.start.k.k_index
        marker = " <<<" if i == first_diff else ""
        print(f"    bi[{bi.index:>2}] {bi.type:>4} k={bi.start.k.k_index}→{bi.end.k.k_index} cl_gap={cl_gap} k_gap={k_gap}{marker}")
    
    # For the pyarmor first-diff BI, show what open would have rejected
    bp = bis_p[first_diff]
    bo = bis_o[first_diff]
    
    # Find the start fx (should be same for both since divergence starts here)
    # The start should match the end of bi[first_diff-1]
    if first_diff > 0:
        start_fx_o = bis_o[first_diff-1].end
        start_fx_p = bis_p[first_diff-1].end
        print(f"\n  start_fx: open k_index={start_fx_o.k.k_index}, pyarmor k_index={start_fx_p.k.k_index}")
    
    # Check if pyarmor's end FX exists in open's FX list
    target_k_index = bp.end.k.k_index
    target_type = bp.end.type
    target_fx = None
    for fx in fxs_o:
        if fx.k.k_index == target_k_index and fx.type == target_type:
            target_fx = fx
            break
    
    if target_fx and first_diff > 0:
        cl_gap_target = target_fx.k.index - start_fx_o.k.index
        k_gap_target = target_fx.k.k_index - start_fx_o.k.k_index
        valid = cd_o._bi_fx_valid(start_fx_o, target_fx)
        print(f"\n  Pyarmor picks end_fx: {target_type}@{target_k_index}")
        print(f"    cl_gap={cl_gap_target}, k_gap={k_gap_target}")
        print(f"    _bi_fx_valid result in open: {valid}")
        if cl_gap_target < 4:
            print(f"    REJECTED because cl_gap ({cl_gap_target}) < 4!")
            print(f"    But k_gap = {k_gap_target} >= 4")
