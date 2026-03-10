"""
Check TZXL line_bad status from pyarmor's _xd_cal_line_xlfx
for BTC60 down[28] case, to see if pyarmor marks the same TZXLs as bad.
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

df60 = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cl60 = CL_Pyarmor("test", "60m", config)
cl60.process_klines(df60)
bis60 = cl60.get_bis()

# --- Call with same-direction up BIs only (our approach) ---
up_bis = [b for b in bis60[28:] if b.type == "up"]
print("=== Same-direction UP BIs only (for di FX in down segment from bi[28]) ===")
for n in [6, 7, 8, 9, 10]:
    lines = up_bis[:n]
    lines_idx = ",".join(str(l.index) for l in lines)
    
    result_nobh = cl60._xd_cal_line_xlfx(lines, "di", "no_bh")
    if result_nobh:
        tzxls, xlfxs = result_nobh
        print(f"\n  n={n} lines=[{lines_idx}]")
        print(f"  TZXLs ({len(tzxls)}):")
        for i, tz in enumerate(tzxls):
            tz_bis = ",".join(str(l.index) for l in tz.lines)
            print(f"    TZ[{i}] bi[{tz_bis}] max={tz.max:.1f} min={tz.min:.1f} line_bad={tz.line_bad}")
        print(f"  XLFXs ({len(xlfxs)}):")
        for i, fx in enumerate(xlfxs):
            fx_bis = ",".join(str(l.index) for l in fx.xl.lines)
            print(f"    FX[{i}] bi[{fx_bis}] bad={fx.is_line_bad} xl_bad={fx.xl.line_bad}")

# --- Call with ALL BIs (pyarmor's approach) ---
print("\n\n=== ALL BIs (both up and down) ===")
all_bis = bis60[28:]
for n in [12, 14, 15, 16]:
    lines = all_bis[:n]
    lines_idx = ",".join(str(l.index) for l in lines)
    
    result_nobh = cl60._xd_cal_line_xlfx(lines, "di", "no_bh")
    if result_nobh:
        tzxls, xlfxs = result_nobh
        print(f"\n  n={n} lines=[{lines_idx}]")
        print(f"  TZXLs ({len(tzxls)}):")
        for i, tz in enumerate(tzxls):
            tz_bis = ",".join(str(l.index) for l in tz.lines)
            print(f"    TZ[{i}] bi[{tz_bis}] max={tz.max:.1f} min={tz.min:.1f} line_bad={tz.line_bad}")
        print(f"  XLFXs ({len(xlfxs)}):")
        for i, fx in enumerate(xlfxs):
            fx_bis = ",".join(str(l.index) for l in fx.xl.lines)
            print(f"    FX[{i}] bi[{fx_bis}] bad={fx.is_line_bad} xl_bad={fx.xl.line_bad}")

# --- Compare with bh mode ---
print("\n\n=== Same-direction UP BIs, bh mode ===")
for n in [6, 7, 8]:
    lines = up_bis[:n]
    lines_idx = ",".join(str(l.index) for l in lines)
    
    result_bh = cl60._xd_cal_line_xlfx(lines, "di", "bh")
    if result_bh:
        tzxls, xlfxs = result_bh
        print(f"\n  n={n} lines=[{lines_idx}]")
        print(f"  TZXLs ({len(tzxls)}):")
        for i, tz in enumerate(tzxls):
            tz_bis = ",".join(str(l.index) for l in tz.lines)
            print(f"    TZ[{i}] bi[{tz_bis}] max={tz.max:.1f} min={tz.min:.1f} line_bad={tz.line_bad}")
        print(f"  XLFXs ({len(xlfxs)}):")
        for i, fx in enumerate(xlfxs):
            fx_bis = ",".join(str(l.index) for l in fx.xl.lines)
            print(f"    FX[{i}] bi[{fx_bis}] bad={fx.is_line_bad} xl_bad={fx.xl.line_bad}")
