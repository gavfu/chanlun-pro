"""Trace pyarmor's _find_xd_end calls to understand its segment chain"""
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

# Try to trace pyarmor's _find_xd_end
original = CL_P._find_xd_end

def trace(self, bis, start_bi_idx, xd_type):
    result = original(self, bis, start_bi_idx, xd_type)
    if result is not None:
        end_bi_idx = result[0]
        print(f"  pyarmor _find_xd_end({xd_type:>4s}, bi[{start_bi_idx}]) → end={end_bi_idx}")
    else:
        print(f"  pyarmor _find_xd_end({xd_type:>4s}, bi[{start_bi_idx}]) → None")
    return result

CL_P._find_xd_end = trace

print("=== ETH60 Pyarmor chain ===")
df = pd.read_parquet("tests/test_data/ETH_USDT_60m_1000.parquet")
cd = CL_P("test", "test", config=CL_CONFIG)
cd.process_klines(df)

print("\nSegments:")
for i, xd in enumerate(cd.get_xds()):
    print(f"  xd[{i}] {xd.type:>4s} bi[{xd.start_line.index}->{xd.end_line.index}] split=[{xd.is_split}]")

CL_P._find_xd_end = original

print("\n" + "="*70)

CL_P._find_xd_end = trace
print("\n=== BTC60 Pyarmor chain ===")
df = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cd = CL_P("test", "test", config=CL_CONFIG)
cd.process_klines(df)

print("\nSegments:")
for i, xd in enumerate(cd.get_xds()):
    print(f"  xd[{i}] {xd.type:>4s} bi[{xd.start_line.index}->{xd.end_line.index}] split=[{xd.is_split}]")

CL_P._find_xd_end = original

print("\n" + "="*70)

CL_P._find_xd_end = trace
print("\n=== BTC5m Pyarmor chain ===")
df = pd.read_parquet("tests/test_data/BTC_USDT_5m_1000.parquet")
cd = CL_P("test", "test", config=CL_CONFIG)
cd.process_klines(df)

print("\nSegments:")
for i, xd in enumerate(cd.get_xds()):
    si = xd.start_line.index
    ei = xd.end_line.index
    if si >= 37:
        print(f"  xd[{i}] {xd.type:>4s} bi[{si}->{ei}] split=[{xd.is_split}]")

CL_P._find_xd_end = original
