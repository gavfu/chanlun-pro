"""
Trace pyarmor's _xd_cal_line_xlfx for ETH60 incrementally for down bi[34].
See at each step what FX is found.
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
cl.process_klines(df)
bis = cl.get_bis()

print("=== ETH60: Incremental _xd_cal_line_xlfx for down bi[34] ===")
print("  Looking for DI FX")
print()

for end in range(37, 52, 2):  # extend by 2 BIs at a time
    if end > len(bis):
        break
    subset = bis[34:end]
    
    # bh mode
    tzxls_bh, xlfxs_bh = cl._xd_cal_line_xlfx(subset, 'di', 'bh')
    bh_str = f"TZXLs={len(tzxls_bh)}"
    if xlfxs_bh:
        fx = xlfxs_bh[0]
        bh_str += f" → FX bad={fx.is_line_bad} xl_lines={[l.index for l in fx.xl.lines]}"
    else:
        bh_str += " → No FX"
    
    # no_bh mode
    tzxls_no, xlfxs_no = cl._xd_cal_line_xlfx(subset, 'di', 'no_bh')
    no_str = f"TZXLs={len(tzxls_no)}"
    if xlfxs_no:
        fx = xlfxs_no[0]
        no_str += f" → FX bad={fx.is_line_bad} xl_lines={[l.index for l in fx.xl.lines]}"
    else:
        no_str += " → No FX"
    
    print(f"  lines[34..{end-1}]({len(subset)}): bh: {bh_str}  |  no_bh: {no_str}")

# Also check for a larger range to see bh FX
print()
print("=== Full analysis with lines[34..50] ===")
subset = bis[34:50]

print("\n  bh TZXLs:")
tzxls_bh, xlfxs_bh = cl._xd_cal_line_xlfx(subset, 'di', 'bh')
for i, t in enumerate(tzxls_bh):
    print(f"    [{i}]: max={t.max} min={t.min} bad={t.line_bad} lines={[l.index for l in t.lines]}")
print(f"\n  bh XLFXs:")
for i, fx in enumerate(xlfxs_bh):
    print(f"    [{i}]: bad={fx.is_line_bad} xl_lines={[l.index for l in fx.xl.lines]} xl.min={fx.xl.min}")
    # What end would this give?
    end_bi = min(fx.xl.lines, key=lambda l: l.low)
    end_idx = end_bi.index
    if bis[end_idx].type == "up":
        end_idx -= 1
    print(f"         → end_bi = bi[{end_bi.index}] → end_bi_idx = {end_idx}")

print("\n  no_bh TZXLs:")
tzxls_no, xlfxs_no = cl._xd_cal_line_xlfx(subset, 'di', 'no_bh')
for i, t in enumerate(tzxls_no):
    print(f"    [{i}]: max={t.max} min={t.min} bad={t.line_bad} lines={[l.index for l in t.lines]}")
print(f"\n  no_bh XLFXs:")
for i, fx in enumerate(xlfxs_no):
    print(f"    [{i}]: bad={fx.is_line_bad} xl_lines={[l.index for l in fx.xl.lines]} xl.min={fx.xl.min}")
    end_bi = min(fx.xl.lines, key=lambda l: l.low)
    end_idx = end_bi.index
    if bis[end_idx].type == "up":
        end_idx -= 1
    print(f"         → end_bi = bi[{end_bi.index}] → end_bi_idx = {end_idx}")

# Key question: does pyarmor use bh DI FX[0] at [39,41] (bad=True, end=40)?
# Or no_bh DI FX[0] at [37] (bad=False, end=36)?
# Pyarmor result: down bi[34→40] → uses bh FX[0]!
