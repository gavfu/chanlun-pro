"""
Simplified trace: focus on _xd_add_up_line calls that create the segment
down bi[28→38]. We need to see what tzxlfx data leads to this result.
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

# Trace _xd_add_up_line - look for result that creates segment ending at bi[38]
original_add = cl._xd_add_up_line

def traced_add(base_line_type, base_lines, up_lines, default_zs_type, up_line_type, up_line, tzxlfx, tzxls, is_split='', not_del=False, not_yx=False):
    result = original_add(base_line_type, base_lines, up_lines, default_zs_type, up_line_type, up_line, tzxlfx, tzxls, is_split, not_del, not_yx)
    
    # Check if result creates or modifies a segment involving bi[28-38]
    if result and hasattr(result, 'start_line') and hasattr(result, 'end_line'):
        if result.start_line.index == 28 or result.end_line.index == 38:
            print(f"\n=== _xd_add_up_line → {result.type} bi[{result.start_line.index}→{result.end_line.index}] ===")
            print(f"  base_line_type={base_line_type}, up_line_type={up_line_type}")
            print(f"  base_lines: [{base_lines[0].index}..{base_lines[-1].index}] ({len(base_lines)})")
            print(f"  up_line: {up_line.type} bi[{up_line.start_line.index}→{up_line.end_line.index}]" if hasattr(up_line, 'start_line') else f"  up_line: {up_line}")
            print(f"  is_split='{is_split}', not_del={not_del}, not_yx={not_yx}")
            
            # Show the tzxlfx content in detail
            if isinstance(tzxlfx, dict):
                for key in ['di', 'bh_di', 'ding', 'bh_ding']:
                    if key in tzxlfx:
                        val = tzxlfx[key]
                        if isinstance(val, dict):
                            for sk, sv in val.items():
                                if isinstance(sv, list):
                                    if len(sv) > 0 and hasattr(sv[0], 'xl'):
                                        # XLFX list
                                        for fx in sv:
                                            print(f"  tzxlfx[{key}][{sk}]: XLFX type={fx.type} bad={fx.is_line_bad} done={fx.done} xl_lines={[l.index for l in fx.xl.lines]} xl.min={fx.xl.min}")
                                    elif len(sv) > 0 and hasattr(sv[0], 'lines') and hasattr(sv[0], 'max'):
                                        # TZXL list
                                        print(f"  tzxlfx[{key}][{sk}]: {len(sv)} TZXLs")
                                        for t in sv:
                                            print(f"    max={t.max} min={t.min} bad={t.line_bad} lines={[l.index for l in t.lines]}")
                                    else:
                                        print(f"  tzxlfx[{key}][{sk}]: list({len(sv)}) first_type={type(sv[0]).__name__ if sv else 'empty'}")
                                else:
                                    print(f"  tzxlfx[{key}][{sk}]: {type(sv).__name__} = {sv}")
            
            print(f"  up_lines count: {len(up_lines)}")
            for u in up_lines[-3:]:
                if hasattr(u, 'start_line'):
                    print(f"    {u.type} bi[{u.start_line.index}→{u.end_line.index}]")
    
    return result

cl._xd_add_up_line = traced_add

cl.process_klines(df)
