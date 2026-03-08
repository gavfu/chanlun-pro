"""
Diagnostic: CL kline differences around src k.index=60-70
"""
import sys
sys.path.insert(0, 'src')
import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')
config = {}
cd_open = CL_Open('BTC/USDT', '60m', config=config)
cd_open.process_klines(df)
cd_pyarmor = CL_Pyarmor('BTC/USDT', '60m', config=config)
cd_pyarmor.process_klines(df)

print("=== CL klines around k.index=60-70 ===")
print("  py cl_klines:")
py_cls = [k for k in cd_pyarmor.get_cl_klines() if 55 <= k.index <= 75]
op_cls = [k for k in cd_open.get_cl_klines() if 55 <= k.index <= 75]

print(f"\n  {'k_idx':>5} {'k.idx':>6} {'high':>10} {'low':>10} {'n':>4}")
print("  pyarmor:")
for k in py_cls:
    print(f"    {k.k_index:>5} {k.index:>6} {k.h:>10.2f} {k.l:>10.2f} {k.n:>4}")

print("  open:")
for k in op_cls:
    print(f"    {k.k_index:>5} {k.index:>6} {k.h:>10.2f} {k.l:>10.2f} {k.n:>4}")

print("\n  Total cl_klines: py={}, op={}".format(len(cd_pyarmor.get_cl_klines()), len(cd_open.get_cl_klines())))
