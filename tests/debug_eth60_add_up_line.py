"""
Trace pyarmor's _xd_add_up_line for ETH60 to see how it decides segment bi[34→40].
We need to intercept the call and see what `tzxlfx` data is passed and how it selects.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_pyarmor import CL

df = pd.read_parquet("tests/test_data/ETH_USDT_60m_1000.parquet")
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

cl = CL("ETH60", "60m", config)

# Monkey-patch _xd_add_up_line to trace calls
orig_add_up_line = cl._xd_add_up_line

call_count = [0]

def traced_add_up_line(base_line_type, base_lines, up_lines, default_zs_type, 
                        up_line_type, up_line, tzxlfx, tzxls, is_split='', not_del=False, not_yx=False):
    call_count[0] += 1
    
    # We care about when it's processing segments from bi base
    if base_line_type == 'bi':
        # Check if this is related to the segment starting at bi[34]
        start_idx = None
        if up_lines:
            last_up = up_lines[-1]
            if hasattr(last_up, 'start_line') and last_up.start_line:
                start_idx = last_up.start_line.index
            elif hasattr(last_up, 'end_line') and last_up.end_line:
                pass
        
        # Print info about this call
        if call_count[0] <= 60:  # Limit output
            bi_indices = [b.index for b in base_lines[-5:]] if base_lines else []
            up_types = []
            if up_lines:
                for ul in up_lines[-3:]:
                    if hasattr(ul, 'start_line') and hasattr(ul, 'end_line') and ul.start_line and ul.end_line:
                        up_types.append(f"{ul.type}[{ul.start_line.index}→{ul.end_line.index}]")
            
            # Show tzxlfx keys and content
            tzxlfx_info = {}
            if tzxlfx and isinstance(tzxlfx, dict):
                for k, v in tzxlfx.items():
                    if isinstance(v, dict):
                        sub_info = {}
                        for sk, sv in v.items():
                            if sv is not None:
                                if isinstance(sv, list) and len(sv) > 0:
                                    sub_info[sk] = f"[{len(sv)} items]"
                                elif hasattr(sv, 'type'):
                                    lines_idx = [l.index for l in sv.xls[1].lines] if sv.xls and sv.xls[1] else ['?']
                                    sub_info[sk] = f"type={sv.type} bad={sv.is_line_bad} xl_lines={lines_idx}"
                                else:
                                    sub_info[sk] = str(sv)[:50]
                            else:
                                sub_info[sk] = None
                        if any(v2 is not None for v2 in sub_info.values()):
                            tzxlfx_info[k] = sub_info
                    elif v is not None:
                        if isinstance(v, list):
                            tzxlfx_info[k] = f"[{len(v)} items]"
                        else:
                            tzxlfx_info[k] = str(v)[:60]
            
            # Only print if related to mid-range segments  
            # Print all for now
            print(f"\n_xd_add_up_line #{call_count[0]}:")
            print(f"  base_line_type={base_line_type}, up_line_type={up_line_type}")
            print(f"  is_split={repr(is_split)}, not_del={not_del}, not_yx={not_yx}")
            print(f"  up_line: type={up_line.type if up_line else None}", end="")
            if up_line and hasattr(up_line, 'start_line') and up_line.start_line:
                print(f" [{up_line.start_line.index}→{up_line.end_line.index}]", end="")
            elif up_line and hasattr(up_line, 'index'):
                print(f" bi[{up_line.index}]", end="")
            print()
            print(f"  last bis: {bi_indices}")
            print(f"  up_lines: {up_types}")
            
            if tzxlfx_info:
                for k, v in tzxlfx_info.items():
                    if isinstance(v, dict):
                        # Only show non-None items
                        non_none = {sk: sv for sk, sv in v.items() if sv is not None}
                        if non_none:
                            print(f"  tzxlfx[{k}]: {non_none}")
                    else:
                        print(f"  tzxlfx[{k}]: {v}")
    
    return orig_add_up_line(base_line_type, base_lines, up_lines, default_zs_type,
                            up_line_type, up_line, tzxlfx, tzxls, is_split, not_del, not_yx)

cl._xd_add_up_line = traced_add_up_line

cl.process_klines(df)

# Show resulting segments
xds = cl.get_xds()
print(f"\n\n=== Final Segments ===")
for i, xd in enumerate(xds):
    print(f"  xd[{i}] {xd.type} bi[{xd.start_line.index}→{xd.end_line.index}]"
          f" {'' if xd.is_split == '' else f'split={xd.is_split}'}")
