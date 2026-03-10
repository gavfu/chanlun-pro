"""
For ALL cases where no_bh first FX is bad, show BOTH no_bh and bh first FX details.
Include TZXL index, bad status, end value.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_interface import TZXL

config = {
    "bi_type": "bi_type_old", "fx_qj": "fx_qj_k", "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes", "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1", "xd_bzh": "xd_bzh_no",
    "xd_zs_max_lines_split": 11, "xd_allow_split_no_highlow": 1,
    "xd_allow_split_zs_kz": 0, "xd_allow_split_zs_more_line": 1,
    "xd_allow_split_zs_no_direction": 1,
}

def build_tzxls(bis, start_bi_idx, xd_type, bh_mode):
    tzxl_bi_type = "down" if xd_type == "up" else "up"
    bh_direction = "up" if xd_type == "up" else "down"

    tzxl_bis = [b for b in bis[start_bi_idx:] if b.type == tzxl_bi_type]
    
    tzxls = []
    for bi in tzxl_bis:
        pre_line = bis[bi.index - 1] if bi.index > 0 else bi
        done = bi.is_done()
        new_tzxl = TZXL(bh_direction=bh_direction, line=bi, pre_line=pre_line, line_bad=False, done=done)
        if not tzxls:
            tzxls.append(new_tzxl)
            continue
        last = tzxls[-1]
        old_c_new = last.max >= new_tzxl.max and last.min <= new_tzxl.min
        new_c_old = new_tzxl.max >= last.max and new_tzxl.min <= last.min
        if bh_mode:
            if old_c_new or new_c_old:
                last.lines.append(bi)
                last.done = done
                last.line_bad = False
                last.update_maxmin()
            else:
                tzxls.append(new_tzxl)
        else:
            if old_c_new:
                last.lines.append(bi)
                last.done = done
                last.line_bad = False
                last.update_maxmin()
            elif new_c_old:
                new_tzxl.line_bad = True
                tzxls.append(new_tzxl)
            else:
                tzxls.append(new_tzxl)
    return tzxls

def find_first_fx(cl, bis, tzxls, start_bi_idx, xd_type, target_fx_type):
    """Returns (tzxl_idx, end_bi_idx, is_bad, extreme_val, bis_str, pohuai_rejected_count)"""
    pohuai_count = 0
    for i in range(1, len(tzxls) - 1):
        curr_xl = tzxls[i]
        prev_xl = tzxls[i - 1]
        next_xl = tzxls[i + 1]

        is_fx = False
        if target_fx_type == "ding":
            if curr_xl.max > prev_xl.max and curr_xl.max > next_xl.max:
                is_fx = True
        else:
            if curr_xl.min < prev_xl.min and curr_xl.min < next_xl.min:
                is_fx = True

        if not is_fx:
            continue

        pohuai = cl._check_xd_bi_pohuai(bis, start_bi_idx, curr_xl, xd_type)
        if pohuai:
            pohuai_count += 1
            continue
        
        if xd_type == "up":
            end_bi = max(curr_xl.lines, key=lambda l: l.high)
            end_idx = end_bi.index
            if bis[end_idx].type == "down" and end_idx > 0:
                end_idx -= 1
        else:
            end_bi = min(curr_xl.lines, key=lambda l: l.low)
            end_idx = end_bi.index
            if bis[end_idx].type == "up" and end_idx > 0:
                end_idx -= 1
        
        if end_idx - start_bi_idx < 2:
            continue
        
        extreme = curr_xl.max if target_fx_type == "ding" else curr_xl.min
        bis_str = ",".join([str(l.index) for l in curr_xl.lines])
        return (i, end_idx, curr_xl.line_bad, extreme, bis_str, pohuai_count)
    return None

test_cases = [
    ("ETH60", "ETH_USDT_60m_1000.parquet", "60m", 15, "up", 19),
    ("ETH60", "ETH_USDT_60m_1000.parquet", "60m", 31, "up", 41),
    ("BTC60", "BTC_USDT_60m_1000.parquet", "60m", 28, "down", 38),
    ("BTC60", "BTC_USDT_60m_1000.parquet", "60m", 39, "up", 41),
    ("BTC60", "BTC_USDT_60m_1000.parquet", "60m", 46, "down", 48),
    ("BTC5m", "BTC_USDT_5m_1000.parquet", "5m", 3, "up", 9),
    ("BTC5m", "BTC_USDT_5m_1000.parquet", "5m", 25, "up", 27),
]

print(f"{'Case':<12} {'Type':<5} {'Start':>5} {'Expected':>8} │ {'no_bh':^35} │ {'bh':^35} │ Pyarmor chooses")
print("─" * 120)

for name, file, freq, start_bi, xd_type, expected in test_cases:
    df = pd.read_parquet(f"tests/test_data/{file}")
    cl = CL_Open(name, freq, config)
    cl.process_klines(df)
    bis = cl.get_bis()
    
    target_fx_type = "ding" if xd_type == "up" else "di"
    
    tzxls_nobh = build_tzxls(bis, start_bi, xd_type, bh_mode=False)
    tzxls_bh = build_tzxls(bis, start_bi, xd_type, bh_mode=True)
    
    fx_nobh = find_first_fx(cl, bis, tzxls_nobh, start_bi, xd_type, target_fx_type)
    fx_bh = find_first_fx(cl, bis, tzxls_bh, start_bi, xd_type, target_fx_type)
    
    if fx_nobh:
        nobh_str = f"TZ[{fx_nobh[0]}] bi[{fx_nobh[4]}] end={fx_nobh[1]} bad={fx_nobh[2]}"
    else:
        nobh_str = "NONE"
    
    if fx_bh:
        bh_str = f"TZ[{fx_bh[0]}] bi[{fx_bh[4]}] end={fx_bh[1]} bad={fx_bh[2]}"
    else:
        bh_str = "NONE"
    
    # Determine which pyarmor chose
    nobh_end = fx_nobh[1] if fx_nobh else None
    bh_end = fx_bh[1] if fx_bh else None
    
    if expected == nobh_end and expected == bh_end:
        chose = "BOTH AGREE"
    elif expected == nobh_end:
        chose = "no_bh"
    elif expected == bh_end:
        chose = "bh"
    else:
        chose = "NEITHER!"
    
    print(f"{name:<12} {xd_type:<5} {start_bi:>5} {expected:>8} │ {nobh_str:<35} │ {bh_str:<35} │ {chose}")
