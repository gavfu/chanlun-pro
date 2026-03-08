# -*- coding: utf-8 -*-
"""Toggle bi_fx_cgd in pyarmor and compare strokes"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from chanlun.cl_pyarmor import CL as CL_Pyarmor

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "test_data", "BTC_USDT_60m_500.parquet"))

# Test 1: default config (bi_fx_cgd_yes)
cd1 = CL_Pyarmor("BTC/USDT", "60m")
cd1.process_klines(df)

# Test 2: bi_fx_cgd_no
cd2 = CL_Pyarmor("BTC/USDT", "60m", config={"bi_fx_cgd": "bi_fx_cgd_no"})
cd2.process_klines(df)

# Test 3: no strict 
cd3 = CL_Pyarmor("BTC/USDT", "60m", config={"allow_bi_fx_strict": 0})
cd3.process_klines(df)

# Test 4: bi_fx_cgd_no AND no strict
cd4 = CL_Pyarmor("BTC/USDT", "60m", config={"bi_fx_cgd": "bi_fx_cgd_no", "allow_bi_fx_strict": 0})
cd4.process_klines(df)

print(f"Test1 (default, cgd_yes+strict): {len(cd1.get_bis())} strokes")
print(f"Test2 (cgd_no+strict):           {len(cd2.get_bis())} strokes")
print(f"Test3 (cgd_yes+no_strict):       {len(cd3.get_bis())} strokes")
print(f"Test4 (cgd_no+no_strict):        {len(cd4.get_bis())} strokes")

# Compare Test1 vs Test2 (effect of cgd)
bis1 = cd1.get_bis()
bis2 = cd2.get_bis()
print(f"\n=== Test1(cgd_yes) vs Test2(cgd_no) ===")
for i in range(max(len(bis1), len(bis2))):
    b1 = bis1[i] if i < len(bis1) else None
    b2 = bis2[i] if i < len(bis2) else None
    if b1 and b2:
        match = "✅" if (b1.start.k.index == b2.start.k.index and b1.end.k.index == b2.end.k.index) else "❌"
        s1 = f"{b1.type:4s} {b1.start.k.index:3d}->{b1.end.k.index:3d}"
        s2 = f"{b2.type:4s} {b2.start.k.index:3d}->{b2.end.k.index:3d}"
        print(f"  bi[{i:2d}] {match} cgd_yes: {s1} | cgd_no: {s2}")
    elif b1:
        print(f"  bi[{i:2d}] ❌ cgd_yes: {b1.type:4s} {b1.start.k.index:3d}->{b1.end.k.index:3d} | cgd_no: (none)")
    elif b2:
        print(f"  bi[{i:2d}] ❌ cgd_yes: (none) | cgd_no: {b2.type:4s} {b2.start.k.index:3d}->{b2.end.k.index:3d}")

# Compare Test3 vs Test4 (effect of cgd when no strict)
bis4 = cd4.get_bis()
print(f"\n=== Test3(no_strict+cgd_yes) vs Test4(no_strict+cgd_no) ===")
for i in range(max(len(bis3), len(bis4))):
    b3 = bis3[i] if i < len(bis3) else None
    b4 = bis4[i] if i < len(bis4) else None
    if b3 and b4:
        match = "✅" if (b3.start.k.index == b4.start.k.index and b3.end.k.index == b4.end.k.index) else "❌"
        s3 = f"{b3.type:4s} {b3.start.k.index:3d}->{b3.end.k.index:3d}"
        s4 = f"{b4.type:4s} {b4.start.k.index:3d}->{b4.end.k.index:3d}"
        print(f"  bi[{i:2d}] {match} cgd_yes: {s3} | cgd_no: {s4}")
    elif b3:
        print(f"  bi[{i:2d}] ❌ cgd_yes: {b3.type:4s} {b3.start.k.index:3d}->{b3.end.k.index:3d} | cgd_no: (none)")
    elif b4:
        print(f"  bi[{i:2d}] ❌ cgd_yes: (none) | cgd_no: {b4.type:4s} {b4.start.k.index:3d}->{b4.end.k.index:3d}")
