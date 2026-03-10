"""
Deep investigation: BTCd down[22→24] vs down[22→28] and BTC5m down[54→56] vs down[54→64]

Compare open vs pyarmor TZXL construction to find divergence point.
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

def compare_tzxl(name, data_path, start_bi_idx, xd_type):
    df = pd.read_parquet(data_path)
    cd_o = CL_O("test", "test", config=CL_CONFIG)
    cd_o.process_klines(df)
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    
    bis_o = cd_o.get_bis()
    bis_p = cd_p.get_bis()
    target_fx_type = "ding" if xd_type == "up" else "di"
    
    print(f"\n{'='*80}")
    print(f"  {name}: {xd_type} from bi[{start_bi_idx}], looking for {target_fx_type}")
    print(f"{'='*80}")
    
    # Build lines from start
    lines_o = [bis_o[k] for k in range(start_bi_idx, len(bis_o))]
    lines_p = [bis_p[k] for k in range(start_bi_idx, len(bis_p))]
    
    # Get pyarmor TZXL
    tzxls_p, xlfxs_p = cd_p._xd_cal_line_xlfx(lines_p, target_fx_type, 'no_bh')
    
    print(f"\n  Pyarmor TZXL ({target_fx_type}, {len(tzxls_p)} elements):")
    for t, xl in enumerate(tzxls_p):
        bad = "BAD" if xl.line_bad else "   "
        bi_indices = [l.index for l in xl.lines]
        print(f"    TZXL[{t}] {bad} bi{bi_indices} max={xl.max:<12.2f} min={xl.min:<12.2f}")
    
    print(f"\n  Pyarmor XLFX:")
    for f_idx, fx in enumerate(xlfxs_p):
        bi_indices = [l.index for l in fx.xl.lines]
        print(f"    XLFX[{f_idx}] {fx.type} @ bi{bi_indices} bad={fx.xl.line_bad}")
    
    # Get open TZXL - use _find_xd_end internals
    # Build TZXL manually like _find_xd_end does
    start_bi = bis_o[start_bi_idx]
    lines_for_tzxl = []
    for k in range(start_bi_idx + 1, len(bis_o)):
        bi = bis_o[k]
        if bi.type != xd_type:
            lines_for_tzxl.append(bi)
    
    # Build TZXL with containment
    from chanlun.cl_interface import TZXL
    tzxls_o = []
    for bi in lines_for_tzxl:
        new_xl = TZXL(line=bi, _lines=[bi])
        new_xl.max = bi.high
        new_xl.min = bi.low
        new_xl.done = True
        new_xl.line_bad = False
        
        if tzxls_o:
            prev = tzxls_o[-1]
            # Check containment (no_bh)
            if prev.max >= new_xl.max and prev.min <= new_xl.min:
                # old contains new -> merge into old
                prev.lines.append(bi)
                continue
            elif new_xl.max >= prev.max and new_xl.min <= prev.min:
                # new contains old -> keep both but mark new as bad
                new_xl.line_bad = True
                tzxls_o.append(new_xl)
                continue
        tzxls_o.append(new_xl)
    
    print(f"\n  Open TZXL (manual reconstruction, {len(tzxls_o)} elements):")
    for t, xl in enumerate(tzxls_o):
        bad = "BAD" if xl.line_bad else "   "
        bi_indices = [l.index for l in xl.lines]
        print(f"    TZXL[{t}] {bad} bi{bi_indices} max={xl.max:<12.2f} min={xl.min:<12.2f}")
    
    # Check what _find_xd_end returns
    result = cd_o._find_xd_end(bis_o, start_bi_idx, xd_type)
    if result:
        end_bi_idx, ding_fx, di_fx, tzxls_result = result
        print(f"\n  Open _find_xd_end result: end_bi_idx={end_bi_idx}")
    else:
        print(f"\n  Open _find_xd_end result: None (incomplete)")
    
    # Trace through FX search manually
    print(f"\n  Manual FX search in open TZXL:")
    for i in range(1, len(tzxls_o) - 1):
        curr = tzxls_o[i]
        prev = tzxls_o[i-1]
        nxt = tzxls_o[i+1]
        is_fx = False
        if target_fx_type == "ding":
            if curr.max > prev.max and curr.max > nxt.max:
                is_fx = True
        else:
            if curr.min < prev.min and curr.min < nxt.min:
                is_fx = True
        
        if is_fx:
            bad = "BAD" if curr.line_bad else ""
            bi_indices = [l.index for l in curr.lines]
            
            # Check bi_pohuai
            pohuai = cd_o._check_xd_bi_pohuai(bis_o, start_bi_idx, curr, xd_type)
            
            # Check _build_xd_fx_result
            build_result = cd_o._build_xd_fx_result(
                bis_o, start_bi_idx, xd_type, target_fx_type,
                curr, prev, nxt, tzxls_o,
            )
            
            print(f"    TZXL[{i}] is {target_fx_type} FX @ bi{bi_indices} {bad}")
            print(f"      pohuai={pohuai}, build_result={'valid(end='+str(build_result[0])+')' if build_result else 'None'}")
            print(f"      line_bad={curr.line_bad}, position i={i}, would_skip={'YES' if curr.line_bad and i < 3 else 'NO'}")

# BTCd
compare_tzxl("BTCd", "tests/test_data/BTC_USDT_d_500.parquet", 22, "down")

# BTC5m
compare_tzxl("BTC5m", "tests/test_data/BTC_USDT_5m_1000.parquet", 54, "down")
