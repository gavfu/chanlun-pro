"""
Focus on _xd_cal_line_xlfx calls where:
- fx_type='di' and lines span bi[29..39+] (down from bi[28])
- Look at which calls use bh and which use no_bh
- Track ALL returned FXes
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

call_count = [0]
relevant_calls = []

def traced_xlfx(self, lines, fx_type='ding', bh_type='no_bh', *args, **kwargs):
    result = original_xlfx(self, lines, fx_type, bh_type, *args, **kwargs)
    call_count[0] += 1
    
    lines_idx = [l.index for l in lines]
    
    # Capture calls where lines include our region of interest
    # For down[28], looking for di FXs on up BIs: 29,31,33,35,37,39,41,...
    # The lines passed would be these up-type BIs
    is_relevant = False
    
    # Calls with lines containing bi[29] through bi[45]
    if any(29 <= idx <= 45 for idx in lines_idx) and len(lines) >= 3:
        is_relevant = True
    
    if is_relevant:
        n_xlfx = 0
        fx_info = []
        tzxl_info = []
        if result is not None:
            n_xlfx = len(result[1])
            for tz in result[0]:
                tz_lines = ",".join(str(l.index) for l in tz.lines)
                tzxl_info.append(f"TZ[{tz_lines}]")
            for fx in result[1]:
                fx_lines = ",".join(str(l.index) for l in fx.xl.lines)
                fx_info.append(f"FX@bi[{fx_lines}]bad={fx.is_line_bad}max={fx.xl.max:.1f}min={fx.xl.min:.1f}")
        
        relevant_calls.append({
            'idx': call_count[0],
            'lines': lines_idx,
            'fx_type': fx_type,
            'bh_type': bh_type,
            'n_lines': len(lines),
            'n_xlfx': n_xlfx,
            'fx_info': fx_info,
            'tzxl_info': tzxl_info,
        })
    
    return result

CL_Pyarmor._xd_cal_line_xlfx = traced_xlfx

# Run BTC60
df60 = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cl60 = CL_Pyarmor("test", "60m", config)
cl60.process_klines(df60)

# Now filter: only show calls for "di" type with n >= 6 (enough to find the key FX)
print(f"Total calls: {call_count[0]}, relevant: {len(relevant_calls)}\n")

print("=== DI calls with lines starting <= 29 spanning through >= 37 ===")
for c in relevant_calls:
    if c['fx_type'] != 'di':
        continue
    if not c['lines']:
        continue
    if c['lines'][0] > 29 or c['lines'][-1] < 37:
        continue
    
    first = c['lines'][0]
    last = c['lines'][-1]
    fx_str = " | ".join(c['fx_info']) if c['fx_info'] else "none"
    print(f"  [{c['idx']:4d}] {c['bh_type']:5s} n={c['n_lines']:2d} "
          f"lines[{first}..{last}] → {c['n_xlfx']} FXs: {fx_str}")

print("\n=== DING calls with lines starting ~40 spanning through >= 44 ===")  
for c in relevant_calls:
    if c['fx_type'] != 'ding':
        continue
    if not c['lines']:
        continue
    if c['lines'][0] > 42 or c['lines'][-1] < 44:
        continue
    
    first = c['lines'][0]
    last = c['lines'][-1]
    fx_str = " | ".join(c['fx_info']) if c['fx_info'] else "none"
    print(f"  [{c['idx']:4d}] {c['bh_type']:5s} n={c['n_lines']:2d} "
          f"lines[{first}..{last}] → {c['n_xlfx']} FXs: {fx_str}")
