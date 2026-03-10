"""
Simulate "always-first" approach in cl_open by monkey-patching _find_xd_end.
Run full diagnostic.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor
from chanlun.cl_interface import TZXL, XLFX, BI
from typing import List, Union, Tuple

config = {
    "bi_type": "bi_type_old", "fx_qj": "fx_qj_k", "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes", "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1", "xd_bzh": "xd_bzh_no",
    "xd_zs_max_lines_split": 11, "xd_allow_split_no_highlow": 1,
    "xd_allow_split_zs_kz": 0, "xd_allow_split_zs_more_line": 1,
    "xd_allow_split_zs_no_direction": 1,
}

def patched_find_xd_end(self, bis, start_bi_idx, xd_type):
    """Same as original but with always-first rule (no more-extreme bad logic)"""
    tzxl_bi_type = "down" if xd_type == "up" else "up"
    bh_direction = "up" if xd_type == "up" else "down"
    target_fx_type = "ding" if xd_type == "up" else "di"

    tzxl_bis = []
    for i in range(start_bi_idx, len(bis)):
        if bis[i].type == tzxl_bi_type:
            tzxl_bis.append(bis[i])

    if len(tzxl_bis) < 3:
        return None

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

    if len(tzxls) < 3:
        return None

    # CHANGED: Always use first FX (no bad rule)
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
            if not self._check_xd_bi_pohuai(bis, start_bi_idx, curr_xl, xd_type):
                result = self._build_xd_fx_result(
                    bis, start_bi_idx, xd_type, target_fx_type,
                    curr_xl, prev_xl, next_xl, tzxls,
                )
                if result is not None:
                    return result

    return None


cases = [
    ("BTCd", "BTC_USDT_d_500.parquet", "d"),
    ("ETH60", "ETH_USDT_60m_1000.parquet", "60m"),
    ("BTC60", "BTC_USDT_60m_1000.parquet", "60m"),
    ("BTC5m", "BTC_USDT_5m_1000.parquet", "5m"),
    ("ETH5m", "ETH_USDT_5m_1000.parquet", "5m"),
]

for name, file, freq in cases:
    df = pd.read_parquet(f"tests/test_data/{file}")
    
    # Create and monkey-patch
    cl_o = CL_Open(name, freq, config)
    cl_o._find_xd_end = lambda bis, start, xd_type, self=cl_o: patched_find_xd_end(self, bis, start, xd_type)
    cl_o.process_klines(df)
    
    cl_p = CL_Pyarmor(name, freq, config)
    cl_p.process_klines(df)
    
    xds_o = cl_o.get_xds()
    xds_p = cl_p.get_xds()
    bis_o = cl_o.get_bis()
    bis_p = cl_p.get_bis()
    
    count_match = "✅" if len(xds_o) == len(xds_p) else "❌"
    
    print(f"\n{'='*80}")
    print(f"  {name}: BI={len(bis_o)}/{len(bis_p)}  XD={len(xds_o)}/{len(xds_p)} {count_match}")
    
    max_len = max(len(xds_o), len(xds_p))
    for i in range(max_len):
        if i < len(xds_o) and i < len(xds_p):
            o = xds_o[i]
            p = xds_p[i]
            o_desc = f"{o.type} bi[{o.start_line.index}→{o.end_line.index}]"
            p_desc = f"{p.type} bi[{p.start_line.index}→{p.end_line.index}]"
            split_o = f" split={o.is_split}" if o.is_split else ""
            split_p = f" split={p.is_split}" if p.is_split else ""
            match = "✅" if (o.type == p.type and o.start_line.index == p.start_line.index and o.end_line.index == p.end_line.index) else "❌"
            print(f"  xd[{i:2d}] {match} open: {o_desc:25s}{split_o:30s}  pyarmor: {p_desc:25s}{split_p}")
        elif i < len(xds_o):
            o = xds_o[i]
            print(f"  xd[{i:2d}]    open: {o.type} bi[{o.start_line.index}→{o.end_line.index}]  pyarmor: MISSING")
        else:
            p = xds_p[i]
            print(f"  xd[{i:2d}]    open: MISSING  pyarmor: {p.type} bi[{p.start_line.index}→{p.end_line.index}]")
