# -*- coding: utf-8 -*-
"""Quick comparison of Test3 vs Test4"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from chanlun.cl_pyarmor import CL

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "test_data", "BTC_USDT_60m_500.parquet"))

cd3 = CL("BTC/USDT", "60m", config={"allow_bi_fx_strict": 0})
cd3.process_klines(df)
cd4 = CL("BTC/USDT", "60m", config={"bi_fx_cgd": "bi_fx_cgd_no", "allow_bi_fx_strict": 0})
cd4.process_klines(df)

b3 = cd3.get_bis()
b4 = cd4.get_bis()
print(f"Test3 (cgd_yes, no_strict): {len(b3)} strokes")
print(f"Test4 (cgd_no, no_strict):  {len(b4)} strokes")
n = max(len(b3), len(b4))
for i in range(n):
    s3 = f"{b3[i].type:4s} {b3[i].start.k.index:3d}->{b3[i].end.k.index:3d}" if i < len(b3) else "(none)"
    s4 = f"{b4[i].type:4s} {b4[i].start.k.index:3d}->{b4[i].end.k.index:3d}" if i < len(b4) else "(none)"
    match = "✅" if i < len(b3) and i < len(b4) and b3[i].start.k.index == b4[i].start.k.index and b3[i].end.k.index == b4[i].end.k.index else "❌"
    print(f"  bi[{i:2d}] {match} cgd_yes: {s3} | cgd_no: {s4}")
