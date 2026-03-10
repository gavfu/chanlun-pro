"""Trace pyarmor's _xd_cal_line_xlfx to see TZXL and XLFX results"""
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

original_cal = CL_P._xd_cal_line_xlfx

call_count = 0

def trace_cal(self, lines, fx_type='ding', bh_type='no_bh', no_done_fx=False, three_fx=False):
    global call_count
    call_count += 1
    result = original_cal(self, lines, fx_type, bh_type, no_done_fx, three_fx)
    
    tzxls, xlfxs = result
    
    # Only print for interesting calls (BI-level lines with reasonable count)
    if len(lines) >= 3 and hasattr(lines[0], 'index'):
        line_indices = [l.index for l in lines[:5]]
        fx_indices = []
        for xlfx in xlfxs:
            if xlfx.xl and xlfx.xl.lines:
                fx_lines = [l.index for l in xlfx.xl.lines]
                fx_indices.append((xlfx.type, xlfx.is_line_bad, fx_lines))
        
        # Focus on interesting segments (down from bi[28])
        if lines[0].index >= 27 and lines[0].index <= 29 and fx_type == "di" and bh_type == "bh":
            print(f"\n  _xd_cal_line_xlfx(lines[{lines[0].index}..{lines[-1].index}]({len(lines)}), fx_type={fx_type}, bh_type={bh_type})")
            print(f"    TZXL count: {len(tzxls)}")
            for i, xl in enumerate(tzxls):
                print(f"      [{i}]: max={xl.max:.1f} min={xl.min:.1f} bad={xl.line_bad} lines=[{','.join(str(l.index) for l in xl.lines)}]")
            print(f"    XLFX count: {len(xlfxs)}")
            for xlfx in xlfxs:
                xl_lines = [l.index for l in xlfx.xl.lines] if xlfx.xl else []
                print(f"      type={xlfx.type} bad={xlfx.is_line_bad} done={xlfx.done} xl_lines={xl_lines}")
    
    return result

CL_P._xd_cal_line_xlfx = trace_cal

print("=== BTC60 Pyarmor ===")
df = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cd = CL_P("test", "test", config=CL_CONFIG)
cd.process_klines(df)

print(f"\nTotal _xd_cal_line_xlfx calls: {call_count}")

CL_P._xd_cal_line_xlfx = original_cal
