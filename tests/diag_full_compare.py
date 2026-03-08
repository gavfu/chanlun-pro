# -*- coding: utf-8 -*-
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "test_data", "BTC_USDT_60m_500.parquet"))
cd_o = CL_O("BTC/USDT", "60m")
cd_o.process_klines(df)
cd_p = CL_P("BTC/USDT", "60m")
cd_p.process_klines(df)

print(f"FXs: ours={len(cd_o.fxs)}, pyarmor={len(cd_p.fxs)}")
print(f"BIs: ours={len(cd_o.bis)}, pyarmor={len(cd_p.bis)}")
print(f"XDs: ours={len(cd_o.xds)}, pyarmor={len(cd_p.xds)}")
print(f"CLKlines: ours={len(cd_o.cl_klines)}, pyarmor={len(cd_p.cl_klines)}")

o_set = {fx.k.index for fx in cd_o.fxs}
p_set = {fx.k.index for fx in cd_p.fxs}
only_p = p_set - o_set
only_o = o_set - p_set
print(f"Only in pyarmor: {sorted(only_p)}")
print(f"Only in ours: {sorted(only_o)}")

for fx in cd_p.fxs:
    if fx.k.index in only_p:
        print(f"Extra pyarmor FX: {fx.type}@{fx.k.index} val={fx.val:.1f} k_idx={fx.k.k_index}")

# Show CLKlines around the extra one
for idx in sorted(only_p):
    print(f"\nAround CLK[{idx}]:")
    for i in range(max(0, idx-2), min(len(cd_o.cl_klines), idx+3)):
        ck = cd_o.cl_klines[i]
        print(f"  Our CLK[{i}] h={ck.h:.1f} l={ck.l:.1f} klines={[k.index for k in ck.klines]}")
    for i in range(max(0, idx-2), min(len(cd_p.cl_klines), idx+3)):
        ck = cd_p.cl_klines[i]
        print(f"  Pya CLK[{i}] h={ck.h:.1f} l={ck.l:.1f} klines={[k.index for k in ck.klines]}")

# Check XD comparison too
print(f"\n=== XD comparison ===")
max_n = max(len(cd_o.xds), len(cd_p.xds))
for i in range(max_n):
    o = cd_o.xds[i] if i < len(cd_o.xds) else None
    p = cd_p.xds[i] if i < len(cd_p.xds) else None
    o_str = f"{o.type:4s} {o.start.k.index:3d} -> {o.end.k.index:3d}" if o else "---"
    p_str = f"{p.type:4s} {p.start.k.index:3d} -> {p.end.k.index:3d}" if p else "---"
    match = o and p and o.type == p.type and o.start.k.index == p.start.k.index and o.end.k.index == p.end.k.index
    flag = "OK" if match else "DIFF"
    print(f"xd[{i:2d}] {o_str:20s}  {p_str:20s}  {flag}")
