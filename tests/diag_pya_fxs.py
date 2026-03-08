"""Check pyarmor's FX19..FX26 and what _bi_fx_valid returns"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
import pandas as pd
from chanlun.cl_pyarmor import CL as CLPya

df = pd.read_parquet(pathlib.Path(__file__).parent / "test_data" / "BTC_USDT_60m_500.parquet")
c = CLPya("BTC/USDT", "60m", {})
c.process_klines(df)
fxs = c.fxs

print("Pyarmor FX19..FX26:")
for i in range(19, 27):
    fx = fxs[i]
    print(f"  FX{i}: {fx.type} val={fx.val} k.index={fx.k.index} k.k_index={fx.k.k_index}")

print()
print("Pyarmor BIs around that region:")
for b in c.bis:
    if b.start.index >= 14 and b.end.index <= 32:
        print(f"  bi[{b.index}]: FX{b.start.index}({b.start.type})->FX{b.end.index}({b.end.type}) h={b.high} l={b.low}")
