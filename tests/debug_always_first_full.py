"""
Empirical test: what if _find_xd_end ALWAYS uses the first FX found
(no bad-FX skipping)? Compare against pyarmor's XD output for ALL datasets.

This tests the "always first FX" hypothesis.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor
from chanlun.cl_interface import XLFX, TZXL, BI, XD
from typing import List, Tuple, Union

config = {
    "bi_type": "bi_type_old", "fx_qj": "fx_qj_k", "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes", "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1", "xd_bzh": "xd_bzh_no",
    "xd_zs_max_lines_split": 11, "xd_allow_split_no_highlow": 1,
    "xd_allow_split_zs_kz": 0, "xd_allow_split_zs_more_line": 1,
    "xd_allow_split_zs_no_direction": 1,
}


def find_xd_end_always_first(cl, bis, start_bi_idx, xd_type):
    """_find_xd_end but always uses the first FX found (no bad skipping)."""
    tzxl_bi_type = "down" if xd_type == "up" else "up"
    bh_direction = "up" if xd_type == "up" else "down"
    target_fx_type = "ding" if xd_type == "up" else "di"

    tzxl_bis = [bi for bi in bis[start_bi_idx:] if bi.type == tzxl_bi_type]
    if len(tzxl_bis) < 3:
        return None

    tzxls = []
    for bi in tzxl_bis:
        pre_line = bis[bi.index - 1] if bi.index > 0 else bi
        done = bi.is_done()
        new_tzxl = TZXL(bh_direction=bh_direction, line=bi, pre_line=pre_line, line_bad=False, done=done)
        if len(tzxls) == 0:
            tzxls.append(new_tzxl)
            continue
        last_tzxl = tzxls[-1]
        if last_tzxl.max >= new_tzxl.max and last_tzxl.min <= new_tzxl.min:
            last_tzxl.lines.append(bi)
            last_tzxl.done = done
            last_tzxl.line_bad = False
            last_tzxl.update_maxmin()
        elif new_tzxl.max >= last_tzxl.max and new_tzxl.min <= last_tzxl.min:
            new_tzxl.line_bad = True
            tzxls.append(new_tzxl)
        else:
            tzxls.append(new_tzxl)

    if len(tzxls) < 3:
        return None

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
                result = cl._build_xd_fx_result(
                    bis, start_bi_idx, xd_type, target_fx_type,
                    curr_xl, prev_xl, next_xl, tzxls,
                )
                if result is not None:
                    return result  # Always use first valid FX
    return None


def build_xds_always_first(cl, bis):
    """Build XDs using always-first-FX rule."""
    xds_info = []
    xd_type, start_bi_idx = cl._find_first_xd_start(bis)
    
    while start_bi_idx < len(bis):
        result = find_xd_end_always_first(cl, bis, start_bi_idx, xd_type)
        if result is None:
            break
        end_bi_idx, ding_fx, di_fx, tzxls = result
        xds_info.append((xd_type, start_bi_idx, end_bi_idx))
        start_bi_idx = end_bi_idx + 1
        xd_type = "down" if xd_type == "up" else "up"
    return xds_info


datasets = [
    ("BTCd", "tests/test_data/BTC_USDT_d_500.parquet"),
    ("ETH60", "tests/test_data/ETH_USDT_60m_1000.parquet"),
    ("BTC60", "tests/test_data/BTC_USDT_60m_1000.parquet"),
    ("BTC5m", "tests/test_data/BTC_USDT_5m_1000.parquet"),
    ("ETH5m", "tests/test_data/ETH_USDT_5m_1000.parquet"),
]

for ds_name, ds_path in datasets:
    df = pd.read_parquet(ds_path)
    
    cl_o = CL_Open("test", "60m", config)
    cl_o.process_klines(df)
    bis_o = cl_o.get_bis()
    
    cl_p = CL_Pyarmor("test", "60m", config)
    cl_p.process_klines(df)
    xds_p = cl_p.get_xds()
    
    # Current cl_open pre-split XDs
    current_xds = []
    xd_type_c, start_c = cl_o._find_first_xd_start(bis_o)
    while start_c < len(bis_o):
        result = cl_o._find_xd_end(bis_o, start_c, xd_type_c)
        if result is None:
            break
        end_c = result[0]
        current_xds.append((xd_type_c, start_c, end_c))
        start_c = end_c + 1
        xd_type_c = "down" if xd_type_c == "up" else "up"
    
    # Always-first pre-split XDs
    first_xds = build_xds_always_first(cl_o, bis_o)
    
    # Pyarmor pre-split (approximate from final XDs)  
    pyarmor_xds = [(xd.type, xd.start_line.index, xd.end_line.index) for xd in xds_p]
    
    print(f"\n{'='*70}")
    print(f"{ds_name}: {len(bis_o)} BIs")
    print(f"{'='*70}")
    print(f"  Current cl_open pre-split: {len(current_xds)}")
    for x in current_xds:
        print(f"    {x[0]} [{x[1]}→{x[2]}]")
    print(f"  Always-first pre-split: {len(first_xds)}")
    for x in first_xds:
        print(f"    {x[0]} [{x[1]}→{x[2]}]")
    print(f"  Pyarmor final (post-split): {len(pyarmor_xds)}")
    for x in pyarmor_xds:
        print(f"    {x[0]} [{x[1]}→{x[2]}]")
    
    # Compare always-first with current
    if first_xds != current_xds:
        print(f"\n  *** DIFFERENCES (always-first vs current): ***")
        max_len = max(len(first_xds), len(current_xds))
        for i in range(max_len):
            f = first_xds[i] if i < len(first_xds) else None
            c = current_xds[i] if i < len(current_xds) else None
            if f != c:
                print(f"    [{i}] first={f} current={c}")
    else:
        print(f"\n  (no differences between always-first and current)")
