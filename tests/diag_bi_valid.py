"""
Diagnose: check why FX99->FX100 doesn't form a BI in open but does in pyarmor
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

open_fxs = c_open.get_fxs()
pya_fxs = c_pya.get_fxs()

# FX[99] and FX[100] in open:
fx99 = open_fxs[99]  # di val=67250
fx100 = open_fxs[100]  # ding val=70000

print("Open FX[99]:")
print(f"  type={fx99.type} val={fx99.val:.2f}")
print(f"  k.date={fx99.k.date} k.index={fx99.k.index} k.k_index={fx99.k.k_index} k.n={fx99.k.n}")

print("\nOpen FX[100]:")
print(f"  type={fx100.type} val={fx100.val:.2f}")
print(f"  k.date={fx100.k.date} k.index={fx100.k.index} k.k_index={fx100.k.k_index} k.n={fx100.k.n}")

cl_gap = fx100.k.index - fx99.k.index
k_gap = fx100.k.k_index - fx99.k.k_index
print(f"\ncl_gap (index diff) = {cl_gap}")
print(f"k_gap (k_index diff) = {k_gap}")

# Show open's bi_type and bi_fx_check settings
print(f"\nOpen config:")
print(f"  bi_type = {c_open.bi_type!r}")
print(f"  fx_check_k_nums = {c_open.fx_check_k_nums}")

# same for pyarmor
pfx99 = pya_fxs[99]
pfx100 = pya_fxs[100]
print(f"\nPyarmor FX[99]:")
print(f"  type={pfx99.type} val={pfx99.val:.2f}")
print(f"  k.date={pfx99.k.date} k.index={pfx99.k.index} k.k_index={pfx99.k.k_index}")
print(f"\nPyarmor FX[100]:")
print(f"  type={pfx100.type} val={pfx100.val:.2f}")
print(f"  k.date={pfx100.k.date} k.index={pfx100.k.index} k.k_index={pfx100.k.k_index}")
pcl_gap = pfx100.k.index - pfx99.k.index
pk_gap = pfx100.k.k_index - pfx99.k.k_index
print(f"\nPyarmor cl_gap = {pcl_gap}, k_gap = {pk_gap}")

# Call open's _bi_fx_valid directly
valid_open = c_open._bi_fx_valid(fx99, fx100)
print(f"\nOpen _bi_fx_valid(FX99, FX100) = {valid_open}")
