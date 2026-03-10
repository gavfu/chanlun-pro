"""Trace _find_first_xd_start candidates for all cases"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P
from chanlun.cl_interface import *

TEST_DATA = {
    "BTCd":  "tests/test_data/BTC_USDT_d_500.parquet",
    "BTC5m": "tests/test_data/BTC_USDT_5m_1000.parquet",
    "ETH5m": "tests/test_data/ETH_USDT_5m_1000.parquet",
}

CL_CONFIG = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
}

def run_case(name):
    print(f"\n{'='*80}")
    print(f"=== {name} ===")
    
    df = pd.read_parquet(TEST_DATA[name])
    cl = CL_O("test", "test", config=CL_CONFIG)
    cl.process_klines(df)
    
    cl_p = CL_P("test", "test", config=CL_CONFIG)
    cl_p.process_klines(df)
    
    bis = cl.get_bis()
    pyarmor_xds = cl_p.get_xds()
    
    print(f"Total BIs: {len(bis)}")
    print(f"Pyarmor first XD: {pyarmor_xds[0].type} bi[{pyarmor_xds[0].start.index}→{pyarmor_xds[0].end.index}]")
    
    # Manually trace _find_all_tzxl_fx and _find_first_xd_start
    ding_candidates = cl._find_all_tzxl_fx(bis, "down", "up", "ding")
    print(f"\nDing candidates: {ding_candidates}")
    for bi_idx, is_bad in ding_candidates:
        print(f"  Ding at bi[{bi_idx}] h={bis[bi_idx].high:.1f} l={bis[bi_idx].low:.1f} type={bis[bi_idx].type} bad={is_bad}")
    
    di_candidates = cl._find_all_tzxl_fx(bis, "up", "down", "di")
    print(f"\nDi candidates: {di_candidates}")
    for bi_idx, is_bad in di_candidates:
        print(f"  Di at bi[{bi_idx}] h={bis[bi_idx].high:.1f} l={bis[bi_idx].low:.1f} type={bis[bi_idx].type} bad={is_bad}")
    
    candidates = []
    for bi_idx, is_bad in ding_candidates:
        start_idx = bi_idx if bis[bi_idx].type == "down" else (bi_idx + 1 if bi_idx + 1 < len(bis) else None)
        if start_idx is not None:
            candidates.append(("down", start_idx, bi_idx))
    for bi_idx, is_bad in di_candidates:
        start_idx = bi_idx if bis[bi_idx].type == "up" else (bi_idx + 1 if bi_idx + 1 < len(bis) else None)
        if start_idx is not None:
            candidates.append(("up", start_idx, bi_idx))
    
    candidates.sort(key=lambda c: c[1])
    
    print(f"\nSorted candidates ({len(candidates)}):")
    for xd_type, start_idx, fx_bi_idx in candidates:
        print(f"  {xd_type} start_bi={start_idx} fx_bi={fx_bi_idx}")
    
    print(f"\nTrying each candidate:")
    for xd_type, start_idx, fx_bi_idx in candidates:
        result = cl._find_xd_end(bis, start_idx, xd_type)
        if result is None:
            print(f"  {xd_type} start_bi[{start_idx}]: _find_xd_end -> None")
            continue
        end_bi_idx, ding_fx, di_fx, tzxls = result
        print(f"  {xd_type} start_bi[{start_idx}] -> end_bi[{end_bi_idx}]")
        
        start_bi = bis[start_idx]
        rejected = False
        if xd_type == "down" and di_fx and di_fx.xl:
            print(f"    Checking di_fx.xl.lines ({len(di_fx.xl.lines)} lines):")
            for l in di_fx.xl.lines:
                exceeds = l.high > start_bi.high
                print(f"      bi[{l.index}] h={l.high:.1f} {'> ' if exceeds else '<='} start h={start_bi.high:.1f} {'REJECT!' if exceeds else 'OK'}")
                if exceeds:
                    rejected = True
        elif xd_type == "up" and ding_fx and ding_fx.xl:
            print(f"    Checking ding_fx.xl.lines ({len(ding_fx.xl.lines)} lines):")
            for l in ding_fx.xl.lines:
                exceeds = l.low < start_bi.low
                print(f"      bi[{l.index}] l={l.low:.1f} {'< ' if exceeds else '>='} start l={start_bi.low:.1f} {'REJECT!' if exceeds else 'OK'}")
                if exceeds:
                    rejected = True
        
        if rejected:
            print(f"    -> REJECTED by FX middle validation")
        else:
            print(f"    -> ACCEPTED! This is the chosen start.")
            break

for name in ["BTCd", "BTC5m", "ETH5m"]:
    run_case(name)
