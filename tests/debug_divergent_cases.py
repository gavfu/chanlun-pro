"""
Quick check: ETH60 up bi[31], BTC60 up bi[39] - what makes 'first' differ from 'current'?
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_open import CL
from chanlun.cl_interface import TZXL

config = {
    "bi_type": "bi_type_old", "fx_qj": "fx_qj_k", "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes", "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1", "xd_bzh": "xd_bzh_no",
    "xd_zs_max_lines_split": 11, "xd_allow_split_no_highlow": 1,
    "xd_allow_split_zs_kz": 0, "xd_allow_split_zs_more_line": 1,
    "xd_allow_split_zs_no_direction": 1,
}

def build_and_analyze(name, file, freq, start, xd_type, expected):
    df = pd.read_parquet(f"tests/test_data/{file}")
    cl = CL(name, freq, config)
    cl.process_klines(df)
    bis = cl.get_bis()
    
    tzxl_bi_type = "down" if xd_type == "up" else "up"
    bh_direction = "up" if xd_type == "up" else "down"
    target_fx = "ding" if xd_type == "up" else "di"
    
    tzxl_bis = [b for b in bis[start:] if b.type == tzxl_bi_type]
    
    tzxls = []
    for bi in tzxl_bis:
        pre_line = bis[bi.index - 1] if bi.index > 0 else bi
        new_tzxl = TZXL(bh_direction=bh_direction, line=bi, pre_line=pre_line,
                        line_bad=False, done=bi.is_done())
        if not tzxls:
            tzxls.append(new_tzxl)
            continue
        last = tzxls[-1]
        o_c_n = last.max >= new_tzxl.max and last.min <= new_tzxl.min
        n_c_o = new_tzxl.max >= last.max and new_tzxl.min <= last.min
        if o_c_n:
            last.lines.append(bi)
            last.done = bi.is_done()
            last.line_bad = False
            last.update_maxmin()
        elif n_c_o:
            new_tzxl.line_bad = True
            tzxls.append(new_tzxl)
        else:
            tzxls.append(new_tzxl)
    
    print(f"\n=== {name} {xd_type} bi[{start}] expected={expected} ===")
    for i, t in enumerate(tzxls[:10]):
        lines = [l.index for l in t.lines]
        print(f"  TZXL[{i}] max={t.max:.1f} min={t.min:.1f} bad={t.line_bad} lines={lines}")
    
    print(f"\n  FX search ({target_fx}):")
    for i in range(1, min(len(tzxls) - 1, 8)):
        curr = tzxls[i]
        prev_xl = tzxls[i-1]
        next_xl = tzxls[i+1]
        if target_fx == "ding":
            is_fx = curr.max > prev_xl.max and curr.max > next_xl.max
        else:
            is_fx = curr.min < prev_xl.min and curr.min < next_xl.min
        if is_fx:
            end_bi = max(curr.lines, key=lambda l: l.high) if target_fx == "ding" else min(curr.lines, key=lambda l: l.low)
            end_idx = end_bi.index
            if xd_type == "up" and bis[end_idx].type == "down":
                end_idx -= 1
            elif xd_type == "down" and bis[end_idx].type == "up":
                end_idx -= 1
            lines = [l.index for l in curr.lines]
            pohuai = cl._check_xd_bi_pohuai(bis, start, curr, xd_type)
            print(f"    FX [{i}] lines={lines} {'max' if target_fx=='ding' else 'min'}={curr.max if target_fx=='ding' else curr.min:.1f} "
                  f"bad={curr.line_bad} pohuai={pohuai} → end={end_idx}")

# ETH60 up bi[31]: current=41, first=33, expected=33
build_and_analyze("ETH60", "ETH_USDT_60m_1000.parquet", "60m", 31, "up", 33)

# BTC60 up bi[39]: current=45, first=41, expected=41
build_and_analyze("BTC60", "BTC_USDT_60m_1000.parquet", "60m", 39, "up", 41)

# BTC60 down bi[14]: current=44, first=24, expected=24
build_and_analyze("BTC60", "BTC_USDT_60m_1000.parquet", "60m", 14, "down", 24)
