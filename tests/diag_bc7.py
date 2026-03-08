"""
Diagnostic: Raw src klines around k.index=60-70 to understand containment differences
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

# Show src klines in range
print("=== Source klines around index=55-70 ===")
src_klines = cd_open.src_klines
for k in src_klines:
    if 55 <= k.index <= 72:
        print(f"  src[{k.index:3d}] h={k.h:.2f} l={k.l:.2f}")

# Now show containment details for cl klines with n > 0
print("\n=== pyarmor CL klines with n>0 (k.index 55-75) ===")
for k in cd_pyarmor.get_cl_klines():
    if 55 <= k.index <= 75 and k.n > 0:
        # Show which src klines it contains
        klines_info = [(kk.index, kk.h, kk.l) for kk in k.klines]
        print(f"  py cl[k_idx={k.k_index}, k.index={k.index}, n={k.n}]: {klines_info}")

print("\n=== open CL klines with n>0 (k.index 55-75) ===")
for k in cd_open.get_cl_klines():
    if 55 <= k.index <= 75 and k.n > 0:
        klines_info = [(kk.index, kk.h, kk.l) for kk in k.klines]
        print(f"  op cl[k_idx={k.k_index}, k.index={k.index}, n={k.n}]: {klines_info}")
