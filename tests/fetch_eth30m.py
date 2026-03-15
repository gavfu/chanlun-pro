"""下载 ETH/USDT 30m 数据 (2025-01-01 ~ 2026-01-01)"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import ccxt

exchange = ccxt.binance()

symbol = 'ETH/USDT'
timeframe = '30m'
start_iso = '2025-01-01T00:00:00Z'
end_ts = exchange.parse8601('2026-01-01T00:00:00Z')
limit = 1000
all_data = []
since = exchange.parse8601(start_iso)

while True:
    data = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    if not data:
        break
    data = [d for d in data if d[0] < end_ts]
    all_data.extend(data)
    if len(data) < limit:
        break
    since = data[-1][0] + 1
    time.sleep(0.2)
    print(f"  Fetched batch, total so far: {len(all_data)}")

print(f'Got {len(all_data)} rows')

df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
df['code'] = symbol
df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
print(f'After dedup: {len(df)} rows')
print(f'Date range: {df["date"].min()} -> {df["date"].max()}')

path = os.path.join(os.path.dirname(__file__), 'test_data', 'ETH_USDT_30m_2025.parquet')
df.to_parquet(path, index=False)
print(f'Saved -> {path}')
