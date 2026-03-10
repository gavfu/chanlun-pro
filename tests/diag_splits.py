"""Investigate BTCd xd[1] and check all pyarmor split types"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P
from chanlun.cl_interface import *

CL_CONFIG = {
    "bi_type": "bi_type_old", "fx_qj": "fx_qj_k", "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes", "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1", "xd_bzh": "xd_bzh_no",
}

TEST_DATA = {
    "BTCd":  "tests/test_data/BTC_USDT_d_500.parquet",
    "ETH60": "tests/test_data/ETH_USDT_60m_1000.parquet",
    "BTC60": "tests/test_data/BTC_USDT_60m_1000.parquet",
    "BTC5m": "tests/test_data/BTC_USDT_5m_1000.parquet",
    "ETH5m": "tests/test_data/ETH_USDT_5m_1000.parquet",
}

for name, path in TEST_DATA.items():
    df = pd.read_parquet(path)
    cl_p = CL_P("test", "test", config=CL_CONFIG)
    cl_p.process_klines(df)
    xds = cl_p.get_xds()
    
    splits = [(i, xd) for i, xd in enumerate(xds) if xd.is_split]
    if splits:
        print(f"\n=== {name}: Pyarmor segments with is_split ===")
        for i, xd in splits:
            print(f"  xd[{i}] {xd.type} bi[{xd.start.index}→{xd.end.index}] split='{xd.is_split}'")
    else:
        print(f"\n=== {name}: No splits ===")

# Detailed BTCd investigation
print(f"\n{'='*60}")
print(f"=== BTCd xd[1] detailed investigation ===")
df = pd.read_parquet(TEST_DATA["BTCd"])

# cl_open
cl = CL_O("test", "test", config=CL_CONFIG)
cl.process_klines(df)
bis_o = cl.get_bis()
xds_o = cl.get_xds()

# pyarmor
cl_p = CL_P("test", "test", config=CL_CONFIG)
cl_p.process_klines(df)
bis_p = cl_p.get_bis()
xds_p = cl_p.get_xds()

print(f"  cl_open  xd[1]: {xds_o[1].type} bi[{xds_o[1].start.index}→{xds_o[1].end.index}]")
print(f"  pyarmor  xd[1]: {xds_p[1].type} bi[{xds_p[1].start.index}→{xds_p[1].end.index}]")

# Check pyarmor xd[1]'s di_fx and ding_fx
xd1_p = xds_p[1]
print(f"\n  Pyarmor xd[1] di_fx:")
if xd1_p.di_fx and xd1_p.di_fx.xl:
    print(f"    xl.lines={[l.index for l in xd1_p.di_fx.xl.lines]} xl.line_bad={xd1_p.di_fx.xl.line_bad}")
    print(f"    is_line_bad={xd1_p.di_fx.is_line_bad}")
    if xd1_p.di_fx.xls:
        for j, xle in enumerate(xd1_p.di_fx.xls):
            if xle:
                print(f"    xls[{j}]: max={xle.max:.1f} min={xle.min:.1f} bad={xle.line_bad} lines={[l.index for l in xle.lines]}")

# Check cl_open xd[1]'s FX
xd1_o = xds_o[1]
print(f"\n  cl_open xd[1] di_fx:")
if xd1_o.di_fx and xd1_o.di_fx.xl:
    print(f"    xl.lines={[l.index for l in xd1_o.di_fx.xl.lines]} xl.line_bad={xd1_o.di_fx.xl.line_bad}")
    print(f"    is_line_bad={xd1_o.di_fx.is_line_bad}")
    if xd1_o.di_fx.xls:
        for j, xle in enumerate(xd1_o.di_fx.xls):
            if xle:
                print(f"    xls[{j}]: max={xle.max:.1f} min={xle.min:.1f} bad={xle.line_bad} lines={[l.index for l in xle.lines]}")

# Check pyarmor TZXL for xd[1] if available
print(f"\n  Pyarmor xd[1] TZXL:")
if hasattr(xd1_p, 'tzxl') and xd1_p.tzxl:
    for i, t in enumerate(xd1_p.tzxl):
        print(f"    TZXL[{i}]: max={t.max:.1f} min={t.min:.1f} bad={t.line_bad} lines={[l.index for l in t.lines]}")
else:
    print("    No tzxl attribute")
