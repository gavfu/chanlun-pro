# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from chanlun.cl_pyarmor import CL

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "test_data", "BTC_USDT_60m_500.parquet"))
cd = CL("BTC/USDT", "60m")
cd.process_klines(df)

for i, bi in enumerate(cd.bis):
    direction = "up" if bi.type == "up" else "down"
    start_idx = bi.start.k.index
    end_idx = bi.end.k.index
    print(f"bi[{i:2d}] {direction:4s} {start_idx:3d} -> {end_idx:3d}")
