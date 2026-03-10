"""Compare raw (pre-split) vs split BIs between open and pyarmor.
Key: does pyarmor also split up 299→330 into 3 sub-BIs?
If so, does it produce different split points?"""
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

# BTC 60m
df = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")

# Open - raw (no split)
cd_raw = CL_O("test", "test", config=CL_CONFIG)
cd_raw.bi_split_k_cross_nums = 0
cd_raw.process_klines(df)
bis_raw = cd_raw.get_bis()

# Open - with split
cd_split = CL_O("test", "test", config=CL_CONFIG)
cd_split.process_klines(df)
bis_split = cd_split.get_bis()

# Pyarmor
cd_p = CL_P("test", "test", config=CL_CONFIG)
cd_p.process_klines(df)
bis_p = cd_p.get_bis()

print(f"BTC 60m: raw={len(bis_raw)}, split={len(bis_split)}, pyarmor={len(bis_p)}")

print(f"\n=== RAW BIs (no split) ===")
for j, b in enumerate(bis_raw):
    print(f"  [{j}] {b.type:4s} {b.start.k.k_index:>4d}→{b.end.k.k_index:>4d}")

print(f"\n=== SPLIT BIs ===")
for j, b in enumerate(bis_split):
    # Mark divergences from pyarmor
    if j < len(bis_p):
        bp = bis_p[j]
        match = (b.type == bp.type and b.start.k.k_index == bp.start.k.k_index 
                 and b.end.k.k_index == bp.end.k.k_index)
        marker = "✅" if match else "❌"
    else:
        marker = "❌"
    print(f"  [{j}] {b.type:4s} {b.start.k.k_index:>4d}→{b.end.k.k_index:>4d} {marker}")

print(f"\n=== PYARMOR BIs ===")
for j, b in enumerate(bis_p):
    print(f"  [{j}] {b.type:4s} {b.start.k.k_index:>4d}→{b.end.k.k_index:>4d}")

# Show which raw BIs were split
print(f"\n=== Split analysis ===")
split_idx = 0
for j, b_raw in enumerate(bis_raw):
    # Count how many split BIs cover this raw BI
    sub_bis = []
    while split_idx < len(bis_split):
        b_s = bis_split[split_idx]
        if b_s.start.k.k_index >= b_raw.start.k.k_index and \
           b_s.end.k.k_index <= b_raw.end.k.k_index:
            sub_bis.append(b_s)
            split_idx += 1
        else:
            break
    if len(sub_bis) > 1:
        print(f"  raw[{j}] {b_raw.type} {b_raw.start.k.k_index}→{b_raw.end.k.k_index} "
              f"SPLIT into {len(sub_bis)}:")
        for sb in sub_bis:
            print(f"    {sb.type} {sb.start.k.k_index}→{sb.end.k.k_index}")
    elif len(sub_bis) == 1:
        pass  # no split
