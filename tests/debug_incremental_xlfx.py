"""
Check if is_line_bad changes for BTC5m bi[6] and bi[10] when called with
different numbers of lines (incremental).

Key question: does the first FX have is_line_bad=False with fewer lines?
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

# BTC5m
df5m = pd.read_parquet("tests/test_data/BTC_USDT_5m_1000.parquet")
cl5m = CL_Pyarmor("test", "5m", config)
cl5m.process_klines(df5m)
bis5m = cl5m.get_bis()

print("=== BTC5m ding from bi[3]: incremental _xd_cal_line_xlfx ===")
for n in range(3, min(25, len(bis5m) - 3)):
    lines = bis5m[3:3+n]
    result = cl5m._xd_cal_line_xlfx(lines, fx_type='ding', bh_type='no_bh')
    tzxls, xlfxs = result
    if xlfxs:
        info = []
        for fx in xlfxs:
            xl_bis = [l.index for l in fx.xl.lines]
            info.append(f"bi[{xl_bis}] is_bad={fx.is_line_bad} xl_bad={fx.xl.line_bad}")
        print(f"  n={n} lines[3..{2+n}]: {len(xlfxs)} FXes: {'; '.join(info)}")

# BTC60
print("\n\n=== BTC60 di from bi[28]: incremental _xd_cal_line_xlfx ===")
df60 = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cl60 = CL_Pyarmor("test", "60m", config)
cl60.process_klines(df60)
bis60 = cl60.get_bis()

for n in range(3, min(25, len(bis60) - 28)):
    lines = bis60[28:28+n]
    result = cl60._xd_cal_line_xlfx(lines, fx_type='di', bh_type='no_bh')
    tzxls, xlfxs = result
    if xlfxs:
        info = []
        for fx in xlfxs:
            xl_bis = [l.index for l in fx.xl.lines]
            info.append(f"bi[{xl_bis}] is_bad={fx.is_line_bad} xl_bad={fx.xl.line_bad}")
        print(f"  n={n} lines[28..{27+n}]: {len(xlfxs)} FXes: {'; '.join(info)}")

# BTC60 ding from bi[39]
print("\n\n=== BTC60 ding from bi[39]: incremental _xd_cal_line_xlfx ===")
for n in range(3, min(20, len(bis60) - 39)):
    lines = bis60[39:39+n]
    result = cl60._xd_cal_line_xlfx(lines, fx_type='ding', bh_type='no_bh')
    tzxls, xlfxs = result
    if xlfxs:
        info = []
        for fx in xlfxs:
            xl_bis = [l.index for l in fx.xl.lines]
            info.append(f"bi[{xl_bis}] is_bad={fx.is_line_bad} xl_bad={fx.xl.line_bad}")
        print(f"  n={n} lines[39..{38+n}]: {len(xlfxs)} FXes: {'; '.join(info)}")

# ETH60 ding from bi[31]
print("\n\n=== ETH60 ding from bi[31]: incremental _xd_cal_line_xlfx ===")
dfeth = pd.read_parquet("tests/test_data/ETH_USDT_60m_1000.parquet")
cleth = CL_Pyarmor("test", "60m", config)
cleth.process_klines(dfeth)
biseth = cleth.get_bis()

for n in range(3, min(25, len(biseth) - 31)):
    lines = biseth[31:31+n]
    result = cleth._xd_cal_line_xlfx(lines, fx_type='ding', bh_type='no_bh')
    tzxls, xlfxs = result
    if xlfxs:
        info = []
        for fx in xlfxs:
            xl_bis = [l.index for l in fx.xl.lines]
            info.append(f"bi[{xl_bis}] is_bad={fx.is_line_bad} xl_bad={fx.xl.line_bad}")
        print(f"  n={n} lines[31..{30+n}]: {len(xlfxs)} FXes: {'; '.join(info)}")
