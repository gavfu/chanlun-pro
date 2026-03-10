"""Debug ETH60 vs BTC60 split point difference"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

CL_CONFIG = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
}

# ETH60 - pyarmor splits at bi[33], not bi[31]
df = pd.read_parquet("tests/test_data/ETH_USDT_60m_1000.parquet")
cd_o = CL_O("test", "test", config=CL_CONFIG)
cd_o.process_klines(df)
cd_p = CL_P("test", "test", config=CL_CONFIG)
cd_p.process_klines(df)

bis = cd_o.get_bis()

print("=== ETH60 BIs around bi[31→41] ===")
for i in range(31, min(42, len(bis))):
    b = bis[i]
    print(f"  bi[{b.index}] {b.type:>4s} high={b.high:.1f} low={b.low:.1f}")

print("\n=== ETH60 BI pohuai analysis (UP segment, checking DOWN BIs going lower) ===")
for i in range(33, 42, 2):  # check DOWN bis: 32, 34, 36, 38, 40
    if i >= len(bis):
        break
    b = bis[i]
    if b.type == "down":
        # Find previous down bi 
        prev = None
        for k in range(i-2, 30, -1):
            if bis[k].type == "down":
                prev = bis[k]
                break
        if prev:
            pohuai = b.low < prev.low
            print(f"  bi[{b.index}] DOWN low={b.low:.1f} vs prev bi[{prev.index}] low={prev.low:.1f} {'** POHUAI **' if pohuai else 'no'}")

# BTC60 - pyarmor splits at bi[13]
df2 = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cd_o2 = CL_O("test2", "test2", config=CL_CONFIG)
cd_o2.process_klines(df2)
bis2 = cd_o2.get_bis()

print("\n=== BTC60 BIs around bi[13→27] ===")
for i in range(13, min(28, len(bis2))):
    b = bis2[i]
    print(f"  bi[{b.index}] {b.type:>4s} high={b.high:.1f} low={b.low:.1f}")

print("\n=== BTC60 BI pohuai analysis (UP segment, checking DOWN BIs going lower) ===")
for i in range(14, 28):
    if i >= len(bis2):
        break
    b = bis2[i]
    if b.type == "down":
        prev = None
        for k in range(i-2, 12, -1):
            if bis2[k].type == "down":
                prev = bis2[k]
                break
        if prev:
            pohuai = b.low < prev.low
            print(f"  bi[{b.index}] DOWN low={b.low:.1f} vs prev bi[{prev.index}] low={prev.low:.1f} {'** POHUAI **' if pohuai else 'no'}")

# Show pyarmor segments for comparison
print("\n=== ETH60 pyarmor segments around split ===")
for xd in cd_p.get_xds():
    si = xd.start_line.index
    ei = xd.end_line.index
    if si >= 28 and si <= 42:
        print(f"  xd {xd.type:>4s} bi[{si}->{ei}] done={xd.done} split=[{xd.is_split}]")
