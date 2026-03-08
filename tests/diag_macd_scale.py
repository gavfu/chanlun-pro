"""Check talib MACD hist calculation"""
import sys
sys.path.insert(0, 'src')
import numpy as np
import pandas as pd
import talib

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')
print(df.columns.tolist())
closes = df['close'].values.astype(float)
dif, dea, hist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)

# Check first valid index (around 33)
first_valid = np.where(~np.isnan(hist))[0][0]
print(f"First valid idx: {first_valid}")
print(f"hist[{first_valid}] = {hist[first_valid]}")
print(f"dif[{first_valid}] = {dif[first_valid]}")
print(f"dea[{first_valid}] = {dea[first_valid]}")
print(f"dif-dea = {dif[first_valid]-dea[first_valid]}")
print(f"2*(dif-dea) = {2*(dif[first_valid]-dea[first_valid])}")
print()
print("Ratio hist/(dif-dea):", hist[first_valid] / (dif[first_valid]-dea[first_valid]))
print("So talib hist = 2*(dif-dea):", abs(hist[first_valid] - 2*(dif[first_valid]-dea[first_valid])) < 1e-10)
