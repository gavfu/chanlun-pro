"""Simulate 'always first valid FX' rule to see full impact on all cases"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl import CL as CL_P

CL_CONFIG = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
}

# Monkey-patch _find_xd_end to use "always first valid FX"
original_find_xd_end = CL_O._find_xd_end

def first_fx_find_xd_end(self, bis, start_bi_idx, xd_type):
    """Modified: always use first valid FX, no bad/non-bad comparison"""
    from chanlun.cl_interface import TZXL, XLFX
    from typing import List
    
    tzxl_bi_type = "down" if xd_type == "up" else "up"
    bh_direction = "up" if xd_type == "up" else "down"
    target_fx_type = "ding" if xd_type == "up" else "di"
    
    tzxl_bis = [bi for bi in bis[start_bi_idx:] if bi.type == tzxl_bi_type]
    if len(tzxl_bis) < 3:
        return None
    
    tzxls: list = []
    for bi in tzxl_bis:
        pre_line = bis[bi.index - 1] if bi.index > 0 else bi
        new_tzxl = TZXL(bh_direction=bh_direction, line=bi, pre_line=pre_line,
                       line_bad=False, done=bi.is_done())
        if not tzxls:
            tzxls.append(new_tzxl)
            continue
        last = tzxls[-1]
        old_new = last.max >= new_tzxl.max and last.min <= new_tzxl.min
        new_old = new_tzxl.max >= last.max and new_tzxl.min <= last.min
        if old_new:
            last.lines.append(bi)
            last.done = bi.is_done()
            last.line_bad = False
            last.update_maxmin()
        elif new_old:
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
        
        if target_fx_type == "ding":
            is_fx = curr_xl.max > prev_xl.max and curr_xl.max > next_xl.max
        else:
            is_fx = curr_xl.min < prev_xl.min and curr_xl.min < next_xl.min
        
        if is_fx:
            if not self._check_xd_bi_pohuai(bis, start_bi_idx, curr_xl, xd_type):
                result = self._build_xd_fx_result(
                    bis, start_bi_idx, xd_type, target_fx_type,
                    curr_xl, prev_xl, next_xl, tzxls
                )
                if result is not None:
                    return result  # Always return first valid
    
    return None

cases = [
    ("BTCd", "tests/test_data/BTC_USDT_d_500.parquet"),
    ("ETH60", "tests/test_data/ETH_USDT_60m_1000.parquet"),
    ("BTC60", "tests/test_data/BTC_USDT_60m_1000.parquet"),
    ("BTC5m", "tests/test_data/BTC_USDT_5m_1000.parquet"),
]

for name, data_file in cases:
    df = pd.read_parquet(data_file)
    
    # Pyarmor
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    xds_p = cd_p.get_xds()
    
    # Open with "always first FX"
    CL_O._find_xd_end = first_fx_find_xd_end
    cd_o = CL_O("test", "test", config=CL_CONFIG)
    cd_o.process_klines(df)
    xds_o = cd_o.get_xds()
    
    n_pyarmor = len(xds_p)
    n_open = len(xds_o)
    
    match_count = 0
    print(f"\n{'='*60}")
    print(f"  {name}: pyarmor={n_pyarmor} xds, open={n_open} xds")
    print(f"{'='*60}")
    
    max_xds = max(n_pyarmor, n_open)
    for i in range(max_xds):
        if i < n_pyarmor:
            p = xds_p[i]
            ps = f"{p.type:>4s} bi[{p.start_line.index}->{p.end_line.index}]"
        else:
            ps = "N/A"
        if i < n_open:
            o = xds_o[i]
            os_ = f"{o.type:>4s} bi[{o.start_line.index}->{o.end_line.index}]"
        else:
            os_ = "N/A"
        
        match = ""
        if i < n_pyarmor and i < n_open:
            p, o = xds_p[i], xds_o[i]
            if p.type == o.type and p.start_line.index == o.start_line.index and p.end_line.index == o.end_line.index:
                match = "✅"
                match_count += 1
            else:
                match = "❌"
        
        print(f"  xd[{i:2d}] P: {ps:25s} O: {os_:25s} {match}")
    
    print(f"  Content matches: {match_count}/{min(n_pyarmor, n_open)}")

CL_O._find_xd_end = original_find_xd_end
