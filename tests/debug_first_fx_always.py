"""
Test: what if we always use the FIRST FX found (bad or not), without the
"more extreme" bad FX replacement logic?
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor
from chanlun.cl_interface import TZXL

config = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
    "xd_zs_max_lines_split": 11,
    "xd_allow_split_no_highlow": 1,
    "xd_allow_split_zs_kz": 0,
    "xd_allow_split_zs_more_line": 1,
    "xd_allow_split_zs_no_direction": 1,
}

def build_tzxls(bis, start_bi_idx, xd_type, mode='no_bh'):
    tzxl_bi_type = "down" if xd_type == "up" else "up"
    bh_direction = "up" if xd_type == "up" else "down"
    
    tzxl_bis = [b for b in bis[start_bi_idx:] if b.type == tzxl_bi_type]
    
    tzxls = []
    for bi in tzxl_bis:
        pre_line = bis[bi.index - 1] if bi.index > 0 else bi
        new_tzxl = TZXL(
            bh_direction=bh_direction,
            line=bi,
            pre_line=pre_line,
            line_bad=False,
            done=bi.is_done(),
        )
        
        if len(tzxls) == 0:
            tzxls.append(new_tzxl)
            continue
        
        last_tzxl = tzxls[-1]
        old_contains_new = last_tzxl.max >= new_tzxl.max and last_tzxl.min <= new_tzxl.min
        new_contains_old = new_tzxl.max >= last_tzxl.max and new_tzxl.min <= last_tzxl.min
        
        if old_contains_new:
            last_tzxl.lines.append(bi)
            last_tzxl.done = bi.is_done()
            last_tzxl.line_bad = False
            last_tzxl.update_maxmin()
        elif new_contains_old:
            if mode == 'bh':
                last_tzxl.lines.append(bi)
                last_tzxl.done = bi.is_done()
                last_tzxl.line_bad = False
                last_tzxl.update_maxmin()
            else:
                new_tzxl.line_bad = True
                tzxls.append(new_tzxl)
        else:
            tzxls.append(new_tzxl)
    
    return tzxls

def find_first_fx_simple(tzxls, target_fx_type, bis, start_bi_idx, xd_type, cl):
    """Find first valid FX, use it immediately (no bad rule)"""
    if len(tzxls) < 3:
        return None
    
    for i in range(1, len(tzxls) - 1):
        curr = tzxls[i]
        prev_xl = tzxls[i - 1]
        next_xl = tzxls[i + 1]
        
        if target_fx_type == "di":
            is_fx = curr.min < prev_xl.min and curr.min < next_xl.min
        else:
            is_fx = curr.max > prev_xl.max and curr.max > next_xl.max
        
        if is_fx:
            pohuai = cl._check_xd_bi_pohuai(bis, start_bi_idx, curr, xd_type)
            if not pohuai:
                if target_fx_type == "di":
                    end_bi = min(curr.lines, key=lambda l: l.low)
                else:
                    end_bi = max(curr.lines, key=lambda l: l.high)
                end_bi_idx = end_bi.index
                if xd_type == "down" and bis[end_bi_idx].type == "up":
                    end_bi_idx -= 1
                elif xd_type == "up" and bis[end_bi_idx].type == "down":
                    end_bi_idx -= 1
                
                if end_bi_idx - start_bi_idx >= 2:
                    return (end_bi_idx, [l.index for l in curr.lines], curr.line_bad)
    
    return None


cases = [
    ("BTCd", "BTC_USDT_d_500.parquet", "d"),
    ("ETH60", "ETH_USDT_60m_1000.parquet", "60m"),
    ("BTC60", "BTC_USDT_60m_1000.parquet", "60m"),
    ("BTC5m", "BTC_USDT_5m_1000.parquet", "5m"),
]

for name, file, freq in cases:
    df = pd.read_parquet(f"tests/test_data/{file}")
    cl_o = CL_Open(name, freq, config)
    cl_o.process_klines(df)
    cl_p = CL_Pyarmor(name, freq, config)
    cl_p.process_klines(df)
    
    bis = cl_o.get_bis()
    xds_p = cl_p.get_xds()
    
    print(f"\n{'='*80}")
    print(f"  {name}")
    
    for xd in xds_p:
        start = xd.start_line.index
        xd_type = xd.type
        target_fx = "ding" if xd_type == "up" else "di"
        expected_end = xd.end_line.index
        
        nobh_tzxls = build_tzxls(bis, start, xd_type, 'no_bh')
        
        # Strategy: always use first FX (no bad rule)
        first_fx = find_first_fx_simple(nobh_tzxls, target_fx, bis, start, xd_type, cl_o)
        first_end = first_fx[0] if first_fx else None
        
        match = "✅" if first_end == expected_end else "❌"
        
        if first_end != expected_end:
            print(f"  {xd_type} bi[{start}] expected={expected_end}: first={first_end}{match}")
            if first_fx:
                print(f"    lines={first_fx[1]} bad={first_fx[2]}")
        else:
            print(f"  {xd_type} bi[{start}→{expected_end}] {match}")
