"""
Trace calls to pyarmor's _xd_cal_line_xlfx to see how it's invoked 
during XD building. Monkey-patch to log all calls.
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

# Save original method
original_xlfx = CL_Pyarmor._xd_cal_line_xlfx

call_log = []

def traced_xlfx(self, lines, fx_type='ding', bh_type='no_bh', *args, **kwargs):
    result = original_xlfx(self, lines, fx_type, bh_type, *args, **kwargs)
    lines_idx = [l.index for l in lines]
    n_tzxl = 0
    n_xlfx = 0
    fx_info = []
    if result is not None:
        n_tzxl = len(result[0])
        n_xlfx = len(result[1])
        for fx in result[1]:
            fx_lines = ",".join(str(l.index) for l in fx.xl.lines)
            fx_info.append(f"bi[{fx_lines}]bad={fx.is_line_bad}")
    
    entry = {
        'lines': lines_idx,
        'fx_type': fx_type,
        'bh_type': bh_type,
        'n_lines': len(lines),
        'n_tzxl': n_tzxl,
        'n_xlfx': n_xlfx,
        'fx_info': fx_info,
    }
    call_log.append(entry)
    return result

# Patch
CL_Pyarmor._xd_cal_line_xlfx = traced_xlfx

# Run BTC60 - the key case
print("=== BTC60 ===")
df60 = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cl60 = CL_Pyarmor("test", "60m", config)
cl60.process_klines(df60)

print(f"\nTotal _xd_cal_line_xlfx calls: {len(call_log)}")
print()
for i, entry in enumerate(call_log):
    first = entry['lines'][0] if entry['lines'] else '?'
    last = entry['lines'][-1] if entry['lines'] else '?'
    fx_str = "; ".join(entry['fx_info']) if entry['fx_info'] else "none"
    print(f"  [{i:3d}] {entry['bh_type']:5s} {entry['fx_type']:4s} n={entry['n_lines']:2d} "
          f"lines[{first}..{last}] → {entry['n_xlfx']} FXs: {fx_str}")
