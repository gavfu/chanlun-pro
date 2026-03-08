"""
Check pyarmor's fx_qj, fx_qy settings and compare bi_fx_valid behavior
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

print("Open config:")
print(f"  bi_type = {c_open.bi_type!r}")
print(f"  fx_qj = {c_open.fx_qj!r}")
print(f"  fx_qy = {c_open.fx_qy!r}")
print(f"  fx_check_k_nums = {c_open.fx_check_k_nums}")
print(f"  allow_bi_fx_strict = {c_open.allow_bi_fx_strict}")

print("\nPyarmor config:")
print(f"  bi_type = {c_pya.bi_type!r}")
print(f"  fx_qj = {c_pya.fx_qj!r}")
print(f"  fx_qy = {c_pya.fx_qy!r}")
print(f"  fx_check_k_nums = {c_pya.fx_check_k_nums}")
print(f"  allow_bi_fx_strict = {c_pya.allow_bi_fx_strict}")

# Now test pyarmor's _bi_fx_valid with FX99 and FX100
open_fxs = c_open.get_fxs()
pya_fxs = c_pya.get_fxs()

fx99_p = pya_fxs[99]
fx100_p = pya_fxs[100]
print(f"\nPyarmor FX99: {fx99_p.type} val={fx99_p.val:.2f} k.index={fx99_p.k.index} k.k_index={fx99_p.k.k_index}")
print(f"Pyarmor FX100: {fx100_p.type} val={fx100_p.val:.2f} k.index={fx100_p.k.index} k.k_index={fx100_p.k.k_index}")
print(f"Pyarmor _bi_fx_valid(FX99, FX100) = {c_pya._bi_fx_valid(fx99_p, fx100_p)}")

# Also show FX99 klines for range check
print(f"\nPyarmor FX99 klines info:")
print(f"  k.h={fx99_p.k.h:.2f} k.l={fx99_p.k.l:.2f}")
for ck in fx99_p.klines:
    if ck is not None:
        print(f"  ck: h={ck.h:.2f} l={ck.l:.2f}")

# Show strict check manually with pyarmor's settings
qj = c_pya.fx_qj
qy = c_pya.fx_qy
sfh = fx99_p.high(qj, qy) if hasattr(fx99_p.high, '__call__') else fx99_p.high
efh = fx100_p.high(qj, qy) if hasattr(fx100_p.high, '__call__') else fx100_p.high
sfl = fx99_p.low(qj, qy) if hasattr(fx99_p.low, '__call__') else fx99_p.low
efl = fx100_p.low(qj, qy) if hasattr(fx100_p.low, '__call__') else fx100_p.low
print(f"\nWith pyarmor qj={qj!r} qy={qy!r}:")
print(f"  start(di) high={sfh:.2f} low={sfl:.2f}")
print(f"  end(ding) high={efh:.2f} low={efl:.2f}")
print(f"  start_high > end_high → invalid: {sfh > efh}")
print(f"  end_low < start_low → invalid: {efl < sfl}")
