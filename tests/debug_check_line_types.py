"""
Verify: does pyarmor's _xd_cal_line_xlfx receive ALL BIs or same-direction only?
Check the line types in the calls.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_pyarmor import CL as CL_Pyarmor

config = {
    "bi_type": "bi_type_old", "fx_qj": "fx_qj_k", "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes", "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1", "xd_bzh": "xd_bzh_no",
    "xd_zs_max_lines_split": 11, "xd_allow_split_no_highlow": 1,
    "xd_allow_split_zs_kz": 0, "xd_allow_split_zs_more_line": 1,
    "xd_allow_split_zs_no_direction": 1,
}

original_xlfx = CL_Pyarmor._xd_cal_line_xlfx

def traced_xlfx(self, lines, fx_type='ding', bh_type='no_bh', *args, **kwargs):
    result = original_xlfx(self, lines, fx_type, bh_type, *args, **kwargs)
    
    lines_idx = [l.index for l in lines]
    
    # Check for our target calls: di with lines[28..39+]
    if lines_idx and lines_idx[0] == 28 and len(lines) >= 12 and fx_type == 'di':
        # Print line types
        line_types = [(l.index, l.type) for l in lines]
        print(f"  {bh_type:5s} {fx_type:4s} n={len(lines):2d}: {line_types[:20]}")
        
        # Now check the line types
        all_same = all(l.type == lines[0].type for l in lines)
        types_set = set(l.type for l in lines)
        print(f"  Types in call: {types_set}, all_same={all_same}")
        
    return result

CL_Pyarmor._xd_cal_line_xlfx = traced_xlfx

df60 = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cl60 = CL_Pyarmor("test", "60m", config)
cl60.process_klines(df60)
