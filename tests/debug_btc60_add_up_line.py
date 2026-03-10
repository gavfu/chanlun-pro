"""
Trace pyarmor's _xd_add_up_line for BTC60 to understand how down[28→38] is determined.
Focus on tzxlfx content.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_pyarmor import CL
from chanlun.cl_interface import XLFX, TZXL

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

orig_add_up_line = cl._xd_add_up_line
call_count = [0]

def format_xlfx(fx):
    if fx is None:
        return "None"
    lines_idx = []
    if fx.xls and len(fx.xls) > 1 and fx.xls[1]:
        lines_idx = [l.index for l in fx.xls[1].lines]
    return f"type={fx.type} bad={fx.is_line_bad} done={fx.done} xl_lines={lines_idx}"

def traced_add_up_line(base_line_type, base_lines, up_lines, default_zs_type,
                        up_line_type, up_line, tzxlfx, tzxls, is_split='', not_del=False, not_yx=False):
    call_count[0] += 1
    
    should_trace = False
    if base_line_type == 'bi' and up_lines:
        for ul in up_lines:
            if hasattr(ul, 'start_line') and ul.start_line:
                if ul.start_line.index >= 25 and ul.start_line.index <= 50:
                    should_trace = True
                if ul.end_line and ul.end_line.index >= 25 and ul.end_line.index <= 50:
                    should_trace = True
    
    if should_trace and base_line_type == 'bi':
        up_descs = []
        for ul in up_lines:
            if hasattr(ul, 'start_line') and ul.start_line:
                up_descs.append(f"{ul.type}[{ul.start_line.index}→{ul.end_line.index}]")
        
        print(f"\n{'='*80}")
        print(f"_xd_add_up_line #{call_count[0]}:")
        print(f"  up_lines: {up_descs}")
        print(f"  up_line: type={up_line.type if up_line else None}", end="")
        if up_line and hasattr(up_line, 'start_line') and up_line.start_line:
            print(f" [{up_line.start_line.index}→{up_line.end_line.index}]", end="")
        print()
        print(f"  is_split={repr(is_split)}, not_del={not_del}, not_yx={not_yx}")
        
        # Show tzxlfx content - all keys
        if tzxlfx and isinstance(tzxlfx, dict):
            for key in ['di', 'bh_di', 'ding', 'bh_ding', 'line_di', 'bh_line_di', 'line_ding', 'bh_line_ding']:
                if key in tzxlfx and tzxlfx[key]:
                    val = tzxlfx[key]
                    if isinstance(val, dict):
                        for sk, sv in val.items():
                            if sv is not None:
                                if isinstance(sv, list):
                                    items = []
                                    for t in sv[:8]:
                                        if hasattr(t, 'lines'):
                                            lines = [l.index for l in t.lines]
                                            items.append(f"max={t.max:.1f} min={t.min:.1f} bad={t.line_bad} lines={lines}")
                                    print(f"  tzxlfx[{key}][{sk}]: {items}")
                                elif isinstance(sv, XLFX):
                                    print(f"  tzxlfx[{key}][{sk}]: {format_xlfx(sv)}")
            
            for key in ['next_base_lines', 'line_base_lines']:
                if key in tzxlfx and tzxlfx[key]:
                    if isinstance(tzxlfx[key], list):
                        indices = [b.index for b in tzxlfx[key][:10]]
                        print(f"  tzxlfx[{key}]: {indices}")
        
        # Show tzxls
        if tzxls:
            print(f"  tzxls ({len(tzxls)}):", end="")
            for t in tzxls[:8]:
                lines = [l.index for l in t.lines]
                print(f" max={t.max:.1f} min={t.min:.1f} bad={t.line_bad} lines={lines};", end="")
            print()
    
    return orig_add_up_line(base_line_type, base_lines, up_lines, default_zs_type,
                            up_line_type, up_line, tzxlfx, tzxls, is_split, not_del, not_yx)

cl._xd_add_up_line = traced_add_up_line

cl.process_klines(df)

xds = cl.get_xds()
print(f"\n\n{'='*80}")
print(f"=== Final Segments ===")
for i, xd in enumerate(xds):
    split = f" split={xd.is_split}" if xd.is_split else ""
    print(f"  xd[{i}] {xd.type} bi[{xd.start_line.index}→{xd.end_line.index}]{split}")
