"""
Detailed trace of ETH60 up bi[15] and BTC5m up bi[25] to understand why 
current "more extreme" rule correctly returns the bad FX.
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

def trace_find_xd_end(cl, bis, start_bi_idx, xd_type, label):
    tzxl_bi_type = "down" if xd_type == "up" else "up"
    bh_direction = "up" if xd_type == "up" else "down"
    target_fx_type = "ding" if xd_type == "up" else "di"

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

    print(f"\n{'='*80}")
    print(f"  {label}: {xd_type} bi[{start_bi_idx}], fx_type={target_fx_type}")
    print(f"\n  TZXLs ({len(tzxls)}):")
    for i, xl in enumerate(tzxls):
        bis_str = ",".join([str(l.index) for l in xl.lines])
        print(f"    [{i:2d}] bi[{bis_str:8s}] max={xl.max:10.1f} min={xl.min:10.1f} bad={xl.line_bad}")

    # Simulate _find_xd_end logic with detailed trace
    first_bad_result = None
    first_bad_extreme = None
    
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
            print(f"\n  TZXL[{i}] IS FX but POHUAI → skip")
            continue
        
        # Compute end
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
            print(f"\n  TZXL[{i}] IS FX, end={end_idx}, but too short (< 2 BIs from start)")
            continue
        
        extreme = curr_xl.max if target_fx_type == "ding" else curr_xl.min
        bis_str = ",".join([str(l.index) for l in curr_xl.lines])
        
        if curr_xl.line_bad:
            if first_bad_result is None:
                first_bad_result = end_idx
                first_bad_extreme = extreme
                print(f"\n  TZXL[{i}] IS FX, BAD, bi[{bis_str}], extreme={extreme:.1f}, end={end_idx} → stored as first_bad")
            else:
                print(f"\n  TZXL[{i}] IS FX, BAD(later), bi[{bis_str}], extreme={extreme:.1f}, end={end_idx} → ignored")
            continue
        
        # Non-bad
        if first_bad_result is not None:
            is_more_extreme = (extreme > first_bad_extreme) if target_fx_type == "ding" else (extreme < first_bad_extreme)
            if is_more_extreme:
                print(f"\n  TZXL[{i}] IS FX, non-bad, bi[{bis_str}], extreme={extreme:.1f} > bad({first_bad_extreme:.1f}), end={end_idx} → MORE EXTREME, CURRENT RETURNS THIS")
                return end_idx
            else:
                print(f"\n  TZXL[{i}] IS FX, non-bad, bi[{bis_str}], extreme={extreme:.1f} <= bad({first_bad_extreme:.1f}), end={end_idx} → LESS EXTREME, CURRENT RETURNS BAD end={first_bad_result}")
                return first_bad_result
        else:
            print(f"\n  TZXL[{i}] IS FX, non-bad(first), bi[{bis_str}], extreme={extreme:.1f}, end={end_idx} → CURRENT RETURNS THIS")
            return end_idx
    
    # End of loop
    if first_bad_result is not None:
        print(f"\n  No non-bad FX found after bad → CURRENT RETURNS BAD end={first_bad_result}")
        return first_bad_result
    print(f"\n  No FX found → returns None")
    return None


for name, file, freq, start, xd_type in [
    ("ETH60", "ETH_USDT_60m_1000.parquet", "60m", 15, "up"),
    ("BTC5m", "BTC_USDT_5m_1000.parquet", "5m", 25, "up"),
    ("BTC60", "BTC_USDT_60m_1000.parquet", "60m", 28, "down"),
    ("BTC60", "BTC_USDT_60m_1000.parquet", "60m", 39, "up"),
    ("BTC60", "BTC_USDT_60m_1000.parquet", "60m", 46, "down"),
]:
    df = pd.read_parquet(f"tests/test_data/{file}")
    cl = CL_Open(name, freq, config)
    cl.process_klines(df)
    bis = cl.get_bis()
    result = trace_find_xd_end(cl, bis, start, xd_type, name)
    print(f"\n  → RESULT: end={result}")
