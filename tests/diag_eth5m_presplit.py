# -*- coding: utf-8 -*-
"""Compare ETH5m pre-split BIs between cl_open and pyarmor"""
import sys, os, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor
from chanlun.cl_interface import FX, BI

DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')

df = pd.read_parquet(os.path.join(DATA_DIR, 'ETH_USDT_5m_1000.parquet'))

# cl_open with pre-split capture
cd_open = CL_Open("ETH/USDT", "5m", config={})
open_presplit = []
orig_fn = cd_open._bi_special_bi_split.__func__
def cap_open(self, bis):
    open_presplit.extend(bis)
    return orig_fn(self, bis)
cd_open._bi_special_bi_split = types.MethodType(cap_open, cd_open)
cd_open.process_klines(df)

# pyarmor
cd_py = CL_Pyarmor("ETH/USDT", "5m", config={})
cd_py.process_klines(df)

# Reconstruct pyarmor pre-split BIs
py_bis = list(cd_py.bis)
py_presplit = []
i = 0
while i < len(py_bis):
    bi = py_bis[i]
    if getattr(bi, 'is_split', False):
        # merge 3 sub-BIs into 1 pre-split BI
        merged = BI(start=bi.start, end=py_bis[i+2].end, _type=bi.type,
                     index=len(py_presplit), default_zs_type=bi.default_zs_type)
        py_presplit.append(merged)
        i += 3
    else:
        py_presplit.append(bi)
        i += 1

print(f"Pre-split counts: open={len(open_presplit)} pyarmor={len(py_presplit)}")
print(f"Final counts: open={len(cd_open.bis)} pyarmor={len(cd_py.bis)}")

# Side-by-side comparison
max_len = max(len(open_presplit), len(py_presplit))
print(f"\n{'#':>3} {'Open':>20} {'Pyarmor':>20} {'Match':>6}")
print("-" * 55)
for i in range(max_len):
    o = f"{open_presplit[i].type[0]}[{open_presplit[i].start.k.index}→{open_presplit[i].end.k.index}]" if i < len(open_presplit) else "---"
    p = f"{py_presplit[i].type[0]}[{py_presplit[i].start.k.index}→{py_presplit[i].end.k.index}]" if i < len(py_presplit) else "---"
    match = "✅" if o == p else "❌"
    if o != p:
        print(f"{i:>3} {o:>20} {p:>20} {match:>6}")

# Check which pyarmor BIs are split and whether they exist in cl_open pre-split
print("\n\nPyarmor splits:")
for i, bi in enumerate(py_bis):
    if getattr(bi, 'is_split', False):
        merged_end = py_bis[i+2].end.k.index
        print(f"  Split #{bi.index}: {bi.type}[{bi.start.k.index}→{merged_end}] → "
              f"{bi.type}[{bi.start.k.index}→{bi.end.k.index}], "
              f"{py_bis[i+1].type}[{py_bis[i+1].start.k.index}→{py_bis[i+1].end.k.index}], "
              f"{py_bis[i+2].type}[{py_bis[i+2].start.k.index}→{py_bis[i+2].end.k.index}]")

        # Check if this exists in cl_open
        found = False
        for ob in open_presplit:
            if ob.start.k.index == bi.start.k.index and ob.end.k.index == merged_end:
                found = True
                break
        print(f"    In cl_open pre-split: {'YES' if found else 'NO'}")
        if not found:
            # Find closest
            for ob in open_presplit:
                if abs(ob.start.k.index - bi.start.k.index) < 10 or abs(ob.end.k.index - merged_end) < 10:
                    print(f"    Nearby: {ob.type}[{ob.start.k.index}→{ob.end.k.index}]")
