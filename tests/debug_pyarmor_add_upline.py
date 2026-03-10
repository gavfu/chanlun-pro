"""
Trace pyarmor's _xd_add_up_line to see how it decides between bh and no_bh results.
This is the function that receives the result from _xd_get_up_line_tzxl_info
and makes the final decision about where the segment ends.
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

# Trace _xd_add_up_line
original_add = cl._xd_add_up_line
add_calls = []

def traced_add(base_line_type, base_lines, up_lines, default_zs_type, up_line_type, up_line, tzxlfx, tzxls, is_split='', not_del=False, not_yx=False):
    # Log key info about the call
    base_last = [b.index for b in base_lines[-3:]] if len(base_lines) > 0 else []
    up_line_info = f"{up_line.type} bi[{up_line.start_line.index}→{up_line.end_line.index}]" if hasattr(up_line, 'start_line') else str(up_line)
    
    # Check if tzxlfx contains relevant FX
    tzxlfx_info = {}
    if isinstance(tzxlfx, dict):
        for key in ['di', 'bh_di', 'ding', 'bh_ding']:
            if key in tzxlfx and isinstance(tzxlfx[key], dict):
                sub = tzxlfx[key]
                tzxls_list = sub.get('tzxls', [])
                xlfxs_list = sub.get('xlfxs', [])
                if not tzxls_list and not xlfxs_list:
                    # Try other key names
                    for k, v in sub.items():
                        if isinstance(v, list) and len(v) > 0:
                            tzxlfx_info[f"{key}.{k}"] = f"list({len(v)})"
                else:
                    tzxlfx_info[key] = f"tzxls={len(tzxls_list)}, xlfxs={len(xlfxs_list)}"
    
    # Only log when we're dealing with the segment containing bi[28]
    if any(b.index in [27, 28, 29] for b in base_lines[-5:]):
        print(f"\n_xd_add_up_line:")
        print(f"  base_line_type={base_line_type}")
        print(f"  base_lines last 3: {base_last}")
        print(f"  up_line: {up_line_info}")
        print(f"  up_line_type={up_line_type}")
        print(f"  is_split='{is_split}'")
        
        if isinstance(tzxlfx, dict):
            for key in sorted(tzxlfx.keys()):
                val = tzxlfx[key]
                if isinstance(val, dict):
                    sub_keys = list(val.keys())
                    # Show details for each sub-dict 
                    xlfx_list = None
                    tzxl_list = None
                    for sk in sub_keys:
                        sv = val[sk]
                        if isinstance(sv, list) and len(sv) > 0:
                            first = sv[0]
                            if hasattr(first, 'xl'):
                                xlfx_list = sv
                            elif hasattr(first, 'lines') and hasattr(first, 'max'):
                                tzxl_list = sv
                    
                    xlfx_info = ""
                    if xlfx_list:
                        for fx in xlfx_list[:2]:
                            xlfx_info += f" XLFX(type={fx.type},bad={fx.is_line_bad},xl_lines={[l.index for l in fx.xl.lines]})"
                    
                    tzxl_info = ""
                    if tzxl_list:
                        tzxl_info = f" {len(tzxl_list)} TZXLs"
                    
                    print(f"  tzxlfx[{key}]: keys={sub_keys}{tzxl_info}{xlfx_info}")
                elif isinstance(val, list):
                    print(f"  tzxlfx[{key}]: list({len(val)})")
                else:
                    if hasattr(val, 'index'):
                        print(f"  tzxlfx[{key}]: line idx={val.index}")
                    else:
                        print(f"  tzxlfx[{key}]: {type(val).__name__}")
    
    result = original_add(base_line_type, base_lines, up_lines, default_zs_type, up_line_type, up_line, tzxlfx, tzxls, is_split, not_del, not_yx)
    
    if any(b.index in [27, 28, 29] for b in base_lines[-5:]):
        print(f"  → result: {result}")
    
    return result

cl._xd_add_up_line = traced_add

cl.process_klines(df)

xds = cl.get_xds()
print(f"\n\nFinal XDs:")
for xd in xds:
    print(f"  {xd.type} bi[{xd.start_line.index}→{xd.end_line.index}] split='{xd.is_split}'")
