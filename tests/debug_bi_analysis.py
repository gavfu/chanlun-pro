"""Instead of guessing, let's directly analyze the differences.
For each dataset where BIs differ, show EXACTLY which FX pairs form BIs 
in pyarmor but not in open (and vice versa).

Then categorize: what kind of test would each rejected/accepted pair pass or fail?"""
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
    print(f"\n{'='*70}")
    print(f" {name}")
    print(f"{'='*70}")
    
    df = pd.read_parquet(path)
    cd_o = CL_O("test", "test", config=CL_CONFIG)
    cd_o.process_klines(df)
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    
    bis_o = cd_o.get_bis()
    bis_p = cd_p.get_bis()
    
    qj = cd_o.fx_qj
    qy = cd_o.fx_qy
    
    # Build sets of BI spans
    open_spans = [(bi.type, bi.start.k.k_index, bi.end.k.k_index) for bi in bis_o]
    pyarmor_spans = [(bi.type, bi.start.k.k_index, bi.end.k.k_index) for bi in bis_p]
    
    # Find divergence point 
    first_diff = None
    for i in range(min(len(bis_o), len(bis_p))):
        if open_spans[i] != pyarmor_spans[i]:
            first_diff = i
            break
    
    if first_diff is None:
        if len(bis_o) == len(bis_p):
            print("  All BIs match perfectly!")
        else:
            print(f"  First {min(len(bis_o), len(bis_p))} match, count differs")
        continue
    
    print(f"  First divergence at bi[{first_diff}]")
    
    # Show context around divergence  
    start = max(0, first_diff - 1)
    end = min(min(len(bis_o), len(bis_p)), first_diff + 6)
    
    print(f"\n  Open BIs [{start}:{end}]:")
    for i in range(start, min(end, len(bis_o))):
        bi = bis_o[i]
        cl_g = bi.end.k.index - bi.start.k.index
        k_g = bi.end.k.k_index - bi.start.k.k_index
        m = " <<<" if i == first_diff else ""
        print(f"    bi[{i:>2}] {bi.type:>4} {bi.start.type}@{bi.start.k.k_index}→{bi.end.type}@{bi.end.k.k_index} cl={cl_g} k={k_g}{m}")
    
    print(f"\n  Pyarmor BIs [{start}:{end}]:")
    for i in range(start, min(end, len(bis_p))):
        bi = bis_p[i]
        cl_g = bi.end.k.index - bi.start.k.index
        k_g = bi.end.k.k_index - bi.start.k.k_index
        m = " <<<" if i == first_diff else ""
        print(f"    bi[{i:>2}] {bi.type:>4} {bi.start.type}@{bi.start.k.k_index}→{bi.end.type}@{bi.end.k.k_index} cl={cl_g} k={k_g}{m}")
    
    # What pyarmor has that open doesn't at the divergence point
    p_bi = bis_p[first_diff]
    o_bi = bis_o[first_diff]
    
    print(f"\n  Analysis:")
    print(f"    Pyarmor bi[{first_diff}]: {p_bi.type} {p_bi.start.type}@{p_bi.start.k.k_index}→{p_bi.end.type}@{p_bi.end.k.k_index}")
    print(f"    Open bi[{first_diff}]:    {o_bi.type} {o_bi.start.type}@{o_bi.start.k.k_index}→{o_bi.end.type}@{o_bi.end.k.k_index}")
    
    # Check: same start, different end?
    if p_bi.start.k.k_index == o_bi.start.k.k_index:
        print(f"    Same start, different end: pyarmor@{p_bi.end.k.k_index} vs open@{o_bi.end.k.k_index}")
        # Which end is shorter?
        if p_bi.end.k.k_index < o_bi.end.k.k_index:
            print(f"    Pyarmor BI is SHORTER (confirmed earlier)")
            # Why does open go further? Open must fail confirmation at some point
            # that pyarmor accepts.
            # Check FXes between pyarmor's end and open's end
            print(f"\n    FXes between k={p_bi.end.k.k_index} and k={o_bi.end.k.k_index}:")
            for fx in cd_o.get_fxs():
                if p_bi.end.k.k_index <= fx.k.k_index <= o_bi.end.k.k_index:
                    print(f"      {fx.type:>4}@{fx.k.k_index} val={fx.val:.2f}")
        else:
            print(f"    Pyarmor BI is LONGER (open confirmed earlier)")
            # Open confirms at a point that pyarmor skips
            # Check what FX open uses as confirmation
            # Open's end is closer, so open confirmed with some FX after o_bi.end
            print(f"\n    Looking for confirmation FX after open's end_fx ({o_bi.end.type}@{o_bi.end.k.k_index}):")
            o_end = o_bi.end
            for fx in cd_o.get_fxs():
                if o_end.k.k_index < fx.k.k_index <= p_bi.end.k.k_index and fx.type != o_end.type:
                    cl_g = fx.k.index - o_end.k.index
                    k_g = fx.k.k_index - o_end.k.k_index
                    v = cd_o._bi_fx_valid(o_end, fx)
                    print(f"      {fx.type:>4}@{fx.k.k_index} val={fx.val:.2f} cl_gap={cl_g} k_gap={k_g} valid={v}")
    else:
        print(f"    Different starts!")
