"""
Check pyarmor's _bi_fx_valid result for FX99->FX100 directly.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import pandas as pd
from chanlun.cl_open import CL as CLOpen
from chanlun.cl_pyarmor import CL as CLPya

# 1000-kline test
df1000 = pd.read_parquet(pathlib.Path(__file__).parent / "test_data" / "BTC_USDT_60m_1000.parquet")

c_open = CLOpen("BTC/USDT", "60m", {})
c_open.process_klines(df1000)
c_pya = CLPya("BTC/USDT", "60m", {})
c_pya.process_klines(df1000)

pya_fxs = c_pya.get_fxs()
fx99_p = pya_fxs[99]
fx100_p = pya_fxs[100]

# Call pyarmor's own _bi_fx_valid
print(f"Pyarmor _bi_fx_valid(FX99, FX100): {c_pya._bi_fx_valid(fx99_p, fx100_p)}")

# Also test with open's FXes (should be same)
open_fxs = c_open.get_fxs()
fx99_o = open_fxs[99]
fx100_o = open_fxs[100]
print(f"Open   _bi_fx_valid(FX99, FX100): {c_open._bi_fx_valid(fx99_o, fx100_o)}")

# Now let's look at a case that pyarmor REJECTS: 500k FX0->FX1
df500 = pd.read_parquet(pathlib.Path(__file__).parent / "test_data" / "BTC_USDT_60m_500.parquet")
c_pya500 = CLPya("BTC/USDT", "60m", {})
c_pya500.process_klines(df500)
c_open500 = CLOpen("BTC/USDT", "60m", {})
c_open500.process_klines(df500)

pya500_fxs = c_pya500.get_fxs()
open500_fxs = c_open500.get_fxs()

print(f"\nPyarmor 500k _bi_fx_valid(FX0, FX1): {c_pya500._bi_fx_valid(pya500_fxs[0], pya500_fxs[1])}")
print(f"Open    500k _bi_fx_valid(FX0, FX1): {c_open500._bi_fx_valid(open500_fxs[0], open500_fxs[1])}")

# FX0 and FX1 in 500k:
fx0 = pya500_fxs[0]
fx1 = pya500_fxs[1]
cl_gap = fx1.k.index - fx0.k.index
k_gap = fx1.k.k_index - fx0.k.k_index
print(f"\n500k FX0: {fx0.type} val={fx0.val:.2f} k.index={fx0.k.index} k.k_index={fx0.k.k_index}")
print(f"500k FX1: {fx1.type} val={fx1.val:.2f} k.index={fx1.k.index} k.k_index={fx1.k.k_index}")
print(f"cl_gap={cl_gap} k_gap={k_gap}")

# Show klines in FX0
print("\nFX0 klines:")
for ck in fx0.klines:
    if ck is not None:
        for rk in ck.klines:
            print(f"  raw k: {rk.date} h={rk.h:.2f} l={rk.l:.2f}")

print("\nFX1 klines:")
for ck in fx1.klines:
    if ck is not None:
        for rk in ck.klines:
            print(f"  raw k: {rk.date} h={rk.h:.2f} l={rk.l:.2f}")

# FX0->FX5 (pyarmor accepts)
print(f"\nPyarmor 500k _bi_fx_valid(FX0, FX5):")
fx5 = pya500_fxs[5]
cl_gap5 = fx5.k.index - fx0.k.index
k_gap5 = fx5.k.k_index - fx0.k.k_index
print(f"  FX5: {fx5.type} val={fx5.val:.2f} k.index={fx5.k.index} k.k_index={fx5.k.k_index}")
print(f"  cl_gap={cl_gap5} k_gap={k_gap5}")
print(f"  valid: {c_pya500._bi_fx_valid(pya500_fxs[0], pya500_fxs[5])}")
