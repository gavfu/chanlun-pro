"""
Trace pyarmor's _xd_get_up_line_tzxl_info to understand what it returns
and how results from bh vs no_bh are combined.

Focus on the BTC60 down bi[28] case.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_pyarmor import CL

df = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
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

cl = CL("BTC60", "60m", config)

# Trace _xd_get_up_line_tzxl_info
original_get_info = cl._xd_get_up_line_tzxl_info

def traced_get_info(base_lines, up_lines, cal_type=None):
    if cal_type is None:
        cal_type = ['di', 'ding', 'bh_di', 'bh_ding', 'line_di', 'line_ding', 'bh_line_di', 'bh_line_ding']
    
    result = original_get_info(base_lines, up_lines, cal_type)
    
    # Focus: find calls where base_lines include bi[28]
    base_idxs = [b.index for b in base_lines[-5:]]
    if 28 in [b.index for b in base_lines]:
        print(f"\n_xd_get_up_line_tzxl_info:")
        print(f"  base_lines last 5: {base_idxs}")
        print(f"  up_lines count: {len(up_lines)}")
        if up_lines:
            print(f"  up_lines: {[(u.type if hasattr(u, 'type') else '?', u.start_line.index if hasattr(u, 'start_line') else '?', u.end_line.index if hasattr(u, 'end_line') else '?') for u in up_lines[-5:]]}")
        print(f"  cal_type: {cal_type}")
        print(f"  result type: {type(result)}")
        # Try to understand the result structure
        if isinstance(result, dict):
            for key, val in result.items():
                print(f"  result[{key}]: {type(val)}")
                if isinstance(val, (list, tuple)):
                    for i, v in enumerate(val[:3]):
                        print(f"    [{i}]: {v}")
        elif isinstance(result, (list, tuple)):
            for i, item in enumerate(result):
                print(f"  result[{i}]: {type(item)}")
                if hasattr(item, '__dict__'):
                    for k, v in item.__dict__.items():
                        if not k.startswith('_'):
                            print(f"    .{k} = {v}")
        else:
            print(f"  result: {result}")
    
    return result

cl._xd_get_up_line_tzxl_info = traced_get_info

cl.process_klines(df)

# Also check what the return type looks like
print("\n\n=== Direct call to examine return type ===")
bis = cl.get_bis()
xds = cl.get_xds()

# Show XD that starts at bi[28]
for xd in xds:
    if xd.start_line.index in [27, 28, 29]:
        print(f"  XD: {xd.type} bi[{xd.start_line.index}→{xd.end_line.index}]")
