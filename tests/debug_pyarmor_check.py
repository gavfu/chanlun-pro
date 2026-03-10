"""Check pyarmor TZXL construction and FX selection for BTC60 and BTC5m"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl import CL as CL_P  # pyarmor

CL_CONFIG = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
}

print("=== BTC60 Pyarmor ===")
df = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cd = CL_P("test", "test", config=CL_CONFIG)
cd.process_klines(df)
xds = cd.get_xds()
print("Segments around bi[28]:")
for i, xd in enumerate(xds):
    si = xd.start_line.index
    ei = xd.end_line.index
    if si >= 13 and si <= 50:
        print(f"  xd[{i}] {xd.type:>4s} bi[{si}->{ei}] done={xd.done} split=[{xd.is_split}]")

print("\n=== BTC5m Pyarmor ===")
df = pd.read_parquet("tests/test_data/BTC_USDT_5m_1000.parquet")
cd = CL_P("test", "test", config=CL_CONFIG)
cd.process_klines(df)
xds = cd.get_xds()
print("Segments around bi[46]:")
for i, xd in enumerate(xds):
    si = xd.start_line.index
    ei = xd.end_line.index
    if si >= 37:
        print(f"  xd[{i}] {xd.type:>4s} bi[{si}->{ei}] done={xd.done} split=[{xd.is_split}]")
