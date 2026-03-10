"""
Trace how pyarmor builds XDs by looking at WHICH caller invokes _xd_cal_line_xlfx.
Use traceback to identify the calling function.
Focus on calls for "di" with lines starting at bi[29] or including bi[29..39].
"""
import sys
sys.path.insert(0, "src")
import traceback

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

call_count = [0]

def traced_xlfx(self, lines, fx_type='ding', bh_type='no_bh', *args, **kwargs):
    result = original_xlfx(self, lines, fx_type, bh_type, *args, **kwargs)
    call_count[0] += 1
    
    lines_idx = [l.index for l in lines]
    
    # Filter: only show calls for "di" where lines include bi indices 29-39
    # OR "ding" where lines include bi indices 40-48
    interesting_di = fx_type == 'di' and any(29 <= idx <= 45 for idx in lines_idx)
    interesting_ding = fx_type == 'ding' and any(40 <= idx <= 50 for idx in lines_idx)
    
    if interesting_di or interesting_ding:
        n_xlfx = 0
        fx_info = []
        if result is not None:
            n_xlfx = len(result[1])
            for fx in result[1]:
                fx_lines = ",".join(str(l.index) for l in fx.xl.lines)
                fx_info.append(f"bi[{fx_lines}]bad={fx.is_line_bad}max={fx.xl.max:.1f}min={fx.xl.min:.1f}")
        
        first = lines_idx[0] if lines_idx else '?'
        last = lines_idx[-1] if lines_idx else '?'
        fx_str = "; ".join(fx_info) if fx_info else "none"
        
        # Get caller info from traceback
        stack = traceback.extract_stack()
        # Find the caller (skip this function and the wrapper)
        callers = []
        for frame in stack[:-1]:  # skip current
            fn = frame.name
            if fn != 'traced_xlfx' and 'pyarmor' not in frame.filename.lower():
                continue
            if fn != 'traced_xlfx':
                callers.append(fn)
        caller = callers[-1] if callers else "?"
        
        print(f"  [{call_count[0]:4d}] {bh_type:5s} {fx_type:4s} n={len(lines):2d} "
              f"lines[{first}..{last}] → {n_xlfx} FXs: {fx_str}  caller={caller}")
    
    return result

CL_Pyarmor._xd_cal_line_xlfx = traced_xlfx

# Run BTC60
df60 = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cl60 = CL_Pyarmor("test", "60m", config)
cl60.process_klines(df60)

print(f"\nTotal calls: {call_count[0]}")
