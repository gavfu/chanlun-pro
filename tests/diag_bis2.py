"""
Diagnose BI divergence: FX index comparison
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import pandas as pd
from chanlun.cl_open import CL as CLOpen
from chanlun.cl_pyarmor import CL as CLPya

df = pd.read_parquet(pathlib.Path(__file__).parent / "test_data" / "BTC_USDT_60m_1000.parquet")

c_open = CLOpen("BTC/USDT", "60m", {})
c_open.process_klines(df)
c_pya = CLPya("BTC/USDT", "60m", {})
c_pya.process_klines(df)

open_bis = c_open.get_bis()
pya_bis = c_pya.get_bis()
open_fxs = c_open.get_fxs()
pya_fxs = c_pya.get_fxs()

print(f"Open: {len(open_bis)} BIs, {len(open_fxs)} FXs")
print(f"Pya:  {len(pya_bis)} BIs, {len(pya_fxs)} FXs")

# Show bi[13..19] with FX indices
print("\n--- Open bi[13..20] with FX indices ---")
for i in range(13, min(len(open_bis), 21)):
    b = open_bis[i]
    print(f"  bi[{i}]: {b.type} h={b.high:.2f} l={b.low:.2f}  FX start={b.start.index} FX end={b.end.index}")

print("\n--- Pyarmor bi[13..20] with FX indices ---")
for i in range(13, min(len(pya_bis), 21)):
    b = pya_bis[i]
    print(f"  bi[{i}]: {b.type} h={b.high:.2f} l={b.low:.2f}  FX start={b.start.index} FX end={b.end.index}")

# Both should have same FX at index 99 (di at 67250)
# Look at FXs 97-105 in both
print("\n--- Open FXs 97-107 ---")
for j in range(97, min(len(open_fxs), 108)):
    fx = open_fxs[j]
    print(f"  FX[{j}]: {fx.type} val={fx.val:.2f} k_date={fx.k.date}")

print("\n--- Pyarmor FXs 97-107 ---")
for j in range(97, min(len(pya_fxs), 108)):
    fx = pya_fxs[j]
    print(f"  FX[{j}]: {fx.type} val={fx.val:.2f} k_date={fx.k.date}")
