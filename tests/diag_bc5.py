"""
Diagnostic: Why bi[6] has different end FX in pyarmor vs open.
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

# Check bi[5] through bi[8] FX details
for bi_idx in range(4, 9):
    py_bi = cd_pyarmor.bis[bi_idx]
    op_bi = cd_open.bis[bi_idx]
    
    print(f"\nbi[{bi_idx}] {py_bi.type}:")
    print(f"  py: start_fx.k.k_idx={py_bi.start.k.k_index} end_fx.k.k_idx={py_bi.end.k.k_index}")
    print(f"     start.k.index={py_bi.start.k.index} end.k.index={py_bi.end.k.index}")
    print(f"     high={py_bi.high}, low={py_bi.low}")
    print(f"  op: start_fx.k.k_idx={op_bi.start.k.k_index} end_fx.k.k_idx={op_bi.end.k.k_index}")
    print(f"     start.k.index={op_bi.start.k.index} end.k.index={op_bi.end.k.index}")
    print(f"     high={op_bi.high}, low={op_bi.low}")
    
    if py_bi.start.k.k_index != op_bi.start.k.k_index or py_bi.end.k.k_index != op_bi.end.k.k_index:
        print(f"  !!! RANGE DIFFERENCE !!!")

# Look at FXs around ck_index 77-80
print("\n=== FXs in cl_kline range 66-90 ===")
py_fxs_range = [fx for fx in cd_pyarmor.fxs if 66 <= fx.k.k_index <= 90]
op_fxs_range = [fx for fx in cd_open.fxs if 66 <= fx.k.k_index <= 90]

print("  pyarmor FXs:")
for fx in py_fxs_range:
    print(f"    {fx.type} k_index={fx.k.k_index}(k.index={fx.k.index}) val={fx.val:.2f}")

print("  open FXs:")
for fx in op_fxs_range:
    print(f"    {fx.type} k_index={fx.k.k_index}(k.index={fx.k.index}) val={fx.val:.2f}")
