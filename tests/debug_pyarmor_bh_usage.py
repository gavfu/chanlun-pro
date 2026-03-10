"""
Trace all pyarmor _xd_cal_line_xlfx calls to see what bh_type values are used
for different segment directions, and what cal_types are passed to 
_xd_get_up_line_tzxl_info.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_pyarmor import CL

# Trace _xd_get_up_line_tzxl_info to see what cal_types are used
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

# Monkey-patch _xd_get_up_line_tzxl_info
original_get_tzxl_info = cl._xd_get_up_line_tzxl_info
calls = []

def traced_get_tzxl_info(base_lines, up_lines, cal_type=None):
    if cal_type is None:
        cal_type = ['di', 'ding', 'bh_di', 'bh_ding', 'line_di', 'line_ding', 'bh_line_di', 'bh_line_ding']
    
    result = original_get_tzxl_info(base_lines, up_lines, cal_type)
    
    # Log the call
    base_idxs = [b.index for b in base_lines[-3:]] if len(base_lines) > 3 else [b.index for b in base_lines]
    up_idxs = [u.start_line.index for u in up_lines[-3:]] if len(up_lines) > 3 else [u.start_line.index if hasattr(u, 'start_line') else '?' for u in up_lines]
    calls.append({
        'base_last3': base_idxs,
        'up_count': len(up_lines),
        'cal_type': cal_type,
    })
    return result

cl._xd_get_up_line_tzxl_info = traced_get_tzxl_info

# Also trace _xd_cal_line_xlfx
original_cal = cl._xd_cal_line_xlfx
xlfx_calls = []

def traced_cal(lines, fx_type='ding', bh_type='no_bh', no_done_fx=False, three_fx=False):
    result = original_cal(lines, fx_type, bh_type, no_done_fx, three_fx)
    line_idxs = [l.index for l in lines] if len(lines) <= 20 else [lines[0].index, '...', lines[-1].index]
    xlfx_calls.append({
        'lines': line_idxs,
        'fx_type': fx_type,
        'bh_type': bh_type,
        'no_done_fx': no_done_fx,
        'three_fx': three_fx,
        'tzxl_count': len(result[0]),
        'xlfx_count': len(result[1]),
    })
    return result

cl._xd_cal_line_xlfx = traced_cal

cl.process_klines(df)

print(f"Total _xd_get_up_line_tzxl_info calls: {len(calls)}")
print(f"Total _xd_cal_line_xlfx calls: {len(xlfx_calls)}")

# Show unique cal_types used
unique_cal_types = set()
for c in calls:
    unique_cal_types.add(tuple(c['cal_type']))
print(f"\nUnique cal_type combinations ({len(unique_cal_types)}):")
for ct in sorted(unique_cal_types):
    print(f"  {list(ct)}")

# Show unique bh_type / fx_type combinations
unique_bh_fx = set()
for c in xlfx_calls:
    unique_bh_fx.add((c['fx_type'], c['bh_type']))
print(f"\nUnique fx_type/bh_type combinations ({len(unique_bh_fx)}):")
for fx, bh in sorted(unique_bh_fx):
    count = sum(1 for c in xlfx_calls if c['fx_type'] == fx and c['bh_type'] == bh)
    print(f"  fx_type={fx}, bh_type={bh}: {count} calls")

# Show the first few _xd_cal_line_xlfx calls with bh_type='bh'
print(f"\nFirst 10 _xd_cal_line_xlfx calls with bh_type='bh':")
bh_calls = [c for c in xlfx_calls if c['bh_type'] == 'bh']
for c in bh_calls[:10]:
    print(f"  lines={c['lines']}, fx_type={c['fx_type']}, three_fx={c['three_fx']}, tzxls={c['tzxl_count']}, xlfxs={c['xlfx_count']}")

print(f"\nFirst 10 _xd_cal_line_xlfx calls with bh_type='no_bh':")
nobh_calls = [c for c in xlfx_calls if c['bh_type'] == 'no_bh']
for c in nobh_calls[:10]:
    print(f"  lines={c['lines']}, fx_type={c['fx_type']}, three_fx={c['three_fx']}, tzxls={c['tzxl_count']}, xlfxs={c['xlfx_count']}")
