"""Compare specific divergence points between open and pyarmor for 500k dataset"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
import pandas as pd
from chanlun.cl_open import CL as CLOpen
from chanlun.cl_pyarmor import CL as CLPya

df = pd.read_parquet(pathlib.Path(__file__).parent / "test_data" / "BTC_USDT_60m_500.parquet")

c_open = CLOpen("BTC/USDT", "60m", {})
c_open.process_klines(df)

c_pya = CLPya("BTC/USDT", "60m", {})
c_pya.process_klines(df)

print(f"Open BIs: {len(c_open.bis)}, Pyarmor BIs: {len(c_pya.bis)}")
print()

print("=== Open BIs ===")
for b in c_open.bis:
    sg = b.end.k.index - b.start.k.index
    kg = b.end.k.k_index - b.start.k.k_index
    print(f"  bi[{b.index}]: FX{b.start.index}({b.start.type})->FX{b.end.index}({b.end.type}) cl_gap={sg} k_gap={kg} h={b.high} l={b.low}")

print()
print("=== Pyarmor BIs ===")
for b in c_pya.bis:
    sg = b.end.k.index - b.start.k.index
    kg = b.end.k.k_index - b.start.k.k_index
    print(f"  bi[{b.index}]: FX{b.start.index}({b.start.type})->FX{b.end.index}({b.end.type}) cl_gap={sg} k_gap={kg} h={b.high} l={b.low}")

print()
print("=== Diff (extra in open, not in pyarmor) ===")
pya_endpoints = {(b.start.index, b.end.index) for b in c_pya.bis}
for b in c_open.bis:
    if (b.start.index, b.end.index) not in pya_endpoints:
        sg = b.end.k.index - b.start.k.index
        kg = b.end.k.k_index - b.start.k.k_index
        print(f"  EXTRA bi[{b.index}]: FX{b.start.index}({b.start.type})->FX{b.end.index}({b.end.type}) cl_gap={sg} k_gap={kg}")

print()
print("=== Diff (in pyarmor, not in open) ===")
open_endpoints = {(b.start.index, b.end.index) for b in c_open.bis}
for b in c_pya.bis:
    if (b.start.index, b.end.index) not in open_endpoints:
        sg = b.end.k.index - b.start.k.index
        kg = b.end.k.k_index - b.start.k.k_index
        print(f"  MISSING bi[{b.index}]: FX{b.start.index}({b.start.type})->FX{b.end.index}({b.end.type}) cl_gap={sg} k_gap={kg}")
