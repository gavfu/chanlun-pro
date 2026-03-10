"""
Build bh TZXLs for all 3 divergent cases and find first FX.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_interface import TZXL, BI

config = {
    "bi_type": "bi_type_old", "fx_qj": "fx_qj_k", "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes", "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1", "xd_bzh": "xd_bzh_no",
    "xd_zs_max_lines_split": 11, "xd_allow_split_no_highlow": 1,
    "xd_allow_split_zs_kz": 0, "xd_allow_split_zs_more_line": 1,
    "xd_allow_split_zs_no_direction": 1,
}

def build_tzxls(bis, start_bi_idx, xd_type, bh_mode):
    """Build TZXLs with optional bh (包含) merge mode"""
    tzxl_bi_type = "down" if xd_type == "up" else "up"
    bh_direction = "up" if xd_type == "up" else "down"

    tzxl_bis = []
    for i in range(start_bi_idx, len(bis)):
        if bis[i].type == tzxl_bi_type:
            tzxl_bis.append(bis[i])

    tzxls = []
    for bi in tzxl_bis:
        pre_line = bis[bi.index - 1] if bi.index > 0 else bi
        done = bi.is_done()
        new_tzxl = TZXL(
            bh_direction=bh_direction, line=bi, pre_line=pre_line,
            line_bad=False, done=done,
        )
        if len(tzxls) == 0:
            tzxls.append(new_tzxl)
            continue
        last_tzxl = tzxls[-1]
        old_contains_new = last_tzxl.max >= new_tzxl.max and last_tzxl.min <= new_tzxl.min
        new_contains_old = new_tzxl.max >= last_tzxl.max and new_tzxl.min <= last_tzxl.min
        
        if bh_mode:
            # bh mode: merge ALL containment (both OLD⊃NEW and NEW⊃OLD)
            if old_contains_new or new_contains_old:
                last_tzxl.lines.append(bi)
                last_tzxl.done = done
                last_tzxl.line_bad = False
                last_tzxl.update_maxmin()
            else:
                tzxls.append(new_tzxl)
        else:
            # no_bh mode: OLD⊃NEW merge, NEW⊃OLD separate+bad
            if old_contains_new:
                last_tzxl.lines.append(bi)
                last_tzxl.done = done
                last_tzxl.line_bad = False
                last_tzxl.update_maxmin()
            elif new_contains_old:
                new_tzxl.line_bad = True
                tzxls.append(new_tzxl)
            else:
                tzxls.append(new_tzxl)

    return tzxls

def find_first_fx(tzxls, target_fx_type, cl, bis, start_bi_idx, xd_type):
    """Find first FX in TZXLs"""
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

        if is_fx:
            if not cl._check_xd_bi_pohuai(bis, start_bi_idx, curr_xl, xd_type):
                if xd_type == "up":
                    end_bi = max(curr_xl.lines, key=lambda l: l.high)
                else:
                    end_bi = min(curr_xl.lines, key=lambda l: l.low)
                end_idx = end_bi.index
                if (xd_type == "up" and end_bi.type == "up") or (xd_type == "down" and end_bi.type == "down"):
                    end_idx -= 1
                    
                bis_str = ",".join([str(l.index) for l in curr_xl.lines])
                extreme_val = curr_xl.max if target_fx_type == "ding" else curr_xl.min
                return i, bis_str, extreme_val, end_idx, curr_xl.line_bad
    return None

# Test cases
cases_data = [
    ("ETH60", "ETH_USDT_60m_1000.parquet", "60m", 31, "up", "ding"),
    ("BTC60", "BTC_USDT_60m_1000.parquet", "60m", 28, "down", "di"),
    ("BTC5m", "BTC_USDT_5m_1000.parquet", "5m", 3, "up", "ding"),
]

for name, file, freq, start_bi, xd_type, fx_type in cases_data:
    df = pd.read_parquet(f"tests/test_data/{file}")
    cl = CL_Open(name, freq, config)
    cl.process_klines(df)
    bis = cl.get_bis()
    
    tzxls_nobh = build_tzxls(bis, start_bi, xd_type, bh_mode=False)
    tzxls_bh = build_tzxls(bis, start_bi, xd_type, bh_mode=True)
    
    fx_nobh = find_first_fx(tzxls_nobh, fx_type, cl, bis, start_bi, xd_type)
    fx_bh = find_first_fx(tzxls_bh, fx_type, cl, bis, start_bi, xd_type)
    
    print(f"\n{'='*80}")
    print(f"  {name} {xd_type} bi[{start_bi}]")
    
    print(f"\n  no_bh TZXLs ({len(tzxls_nobh)}):")
    for i, xl in enumerate(tzxls_nobh[:15]):
        bis_str = ",".join([str(l.index) for l in xl.lines])
        print(f"    [{i}] bi[{bis_str}] max={xl.max:.1f} min={xl.min:.1f} bad={xl.line_bad}")
    
    print(f"\n  bh TZXLs ({len(tzxls_bh)}):")
    for i, xl in enumerate(tzxls_bh[:15]):
        bis_str = ",".join([str(l.index) for l in xl.lines])
        print(f"    [{i}] bi[{bis_str}] max={xl.max:.1f} min={xl.min:.1f} bad={xl.line_bad}")
    
    if fx_nobh:
        print(f"\n  no_bh first FX: TZXL[{fx_nobh[0]}] bi[{fx_nobh[1]}] extreme={fx_nobh[2]:.1f} end={fx_nobh[3]} bad={fx_nobh[4]}")
    else:
        print(f"\n  no_bh first FX: NONE")
        
    if fx_bh:
        print(f"  bh first FX: TZXL[{fx_bh[0]}] bi[{fx_bh[1]}] extreme={fx_bh[2]:.1f} end={fx_bh[3]} bad={fx_bh[4]}")
    else:
        print(f"  bh first FX: NONE")
    
    # Now find ALL FX in no_bh to show the "more extreme" candidate
    print(f"\n  All no_bh FX:")
    for i in range(1, len(tzxls_nobh) - 1):
        curr_xl = tzxls_nobh[i]
        prev_xl = tzxls_nobh[i - 1]
        next_xl = tzxls_nobh[i + 1]
        is_fx = False
        if fx_type == "ding":
            if curr_xl.max > prev_xl.max and curr_xl.max > next_xl.max:
                is_fx = True
        else:
            if curr_xl.min < prev_xl.min and curr_xl.min < next_xl.min:
                is_fx = True
        if is_fx:
            if xd_type == "up":
                end_bi = max(curr_xl.lines, key=lambda l: l.high)
            else:
                end_bi = min(curr_xl.lines, key=lambda l: l.low)
            end_idx = end_bi.index
            if (xd_type == "up" and end_bi.type == "up") or (xd_type == "down" and end_bi.type == "down"):
                end_idx -= 1
            extreme_val = curr_xl.max if fx_type == "ding" else curr_xl.min
            bis_str = ",".join([str(l.index) for l in curr_xl.lines])
            pohuai = cl._check_xd_bi_pohuai(bis, start_bi, curr_xl, xd_type)
            print(f"    TZXL[{i}] bi[{bis_str}] extreme={extreme_val:.1f} end={end_idx} bad={curr_xl.line_bad} pohuai={pohuai}")
