"""
Understand what conditions differentiate accepted vs rejected BIs in pyarmor
when k_gap >= 4 but open would reject (cl_gap < 4).

We know:
- BTC 60m 1000: FX99->FX100 (cl_gap=1, k_gap=5) → pyarmor ACCEPTS
- BTC 60m 500: open=36 BIs (too many), pyarmor=27 BIs

Want to find: what conditions make open accept a BI that pyarmor rejects,
or vice versa?
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import pandas as pd
from chanlun.cl_open import CL as CLOpen
from chanlun.cl_pyarmor import CL as CLPya

# Test with 500 klines first
df = pd.read_parquet(pathlib.Path(__file__).parent / "test_data" / "BTC_USDT_60m_500.parquet")

c_open = CLOpen("BTC/USDT", "60m", {})
c_open.process_klines(df)
c_pya = CLPya("BTC/USDT", "60m", {})
c_pya.process_klines(df)

open_bis = c_open.get_bis()
pya_bis = c_pya.get_bis()
open_fxs = c_open.get_fxs()

print(f"500klines: Open={len(open_bis)} BIs, Pya={len(pya_bis)} BIs")

# Show ALL open BIs with their FX cl_gap and k_gap
print("\n--- All Open BIs ---")
for i, b in enumerate(open_bis):
    cl_gap = b.end.k.index - b.start.k.index
    k_gap = b.end.k.k_index - b.start.k.k_index
    # find in pyarmor
    pya_match = any(
        pb.start.index == b.start.index and pb.end.index == b.end.index
        for pb in pya_bis
    )
    pya_idx = next(
        (j for j, pb in enumerate(pya_bis)
         if pb.start.index == b.start.index and pb.end.index == b.end.index),
        -1
    )
    marker = "" if pya_idx >= 0 else " ❌EXTRA"
    print(f"  bi[{i}]: {b.type} h={b.high:.2f} l={b.low:.2f}  "
          f"FX {b.start.index}→{b.end.index}  cl_gap={cl_gap} k_gap={k_gap}{marker}")

print("\n--- Pyarmor BIs that open MISSING ---")
for j, p in enumerate(pya_bis):
    open_match = any(
        ob.start.index == p.start.index and ob.end.index == p.end.index
        for ob in open_bis
    )
    if not open_match:
        cl_gap = p.end.k.index - p.start.k.index
        k_gap = p.end.k.k_index - p.start.k.k_index
        print(f"  pya bi[{j}]: {p.type} h={p.high:.2f} l={p.low:.2f}  "
              f"FX {p.start.index}→{p.end.index}  cl_gap={cl_gap} k_gap={k_gap}  ❌MISSING")
