"""Trace pyarmor's _xd_cal_line_xlfx calls"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
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

import inspect

# Check signature of key methods
for method_name in ['_xd_cal_line_xlfx', '_xd_get_next_tzfx', '_xd_line_to_tzxl',
                    '_xd_get_up_line_tzxl_info', 'process_up_line',
                    '_xd_check_and_split_up_line', '_xd_add_up_line',
                    '_xd_split_optimal_hl_ll']:
    try:
        method = getattr(CL_P, method_name)
        sig = inspect.signature(method)
        print(f"  {method_name}{sig}")
    except Exception as e:
        print(f"  {method_name}: {e}")
