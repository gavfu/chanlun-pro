"""下载大量历史数据用于 ZSD/QSD 测试"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import ccxt

exchange = ccxt.binance()


def fetch_ohlcv_full(symbol, timeframe, bars=5000, start_iso="2017-01-01T00:00:00Z"):
    """从 start_iso 时刻起向后翻页，直到获取足够数量的 bars。"""
    limit = 1000
    all_data = []
    since = exchange.parse8601(start_iso)
    while len(all_data) < bars:
        data = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not data:
            break
        all_data.extend(data)
        since = data[-1][0] + 1
        if len(data) < limit:
            break
        time.sleep(0.2)
    # 只保留最新的 bars 条（正序结尾）
    return all_data[-bars:] if len(all_data) > bars else all_data


def save(rows, symbol, path):
    df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['code'] = symbol
    df.to_parquet(path, index=False)
    print(f"  Saved {len(df)} rows -> {path}")
    print(f"  Date range: {df['date'].min()} -> {df['date'].max()}")


tasks = [
    ('BTC/USDT', '4h',  5000, "2023-11-01T00:00:00Z", 'tests/test_data/BTC_USDT_4h_5000.parquet'),
    ('ETH/USDT', '4h',  5000, "2023-11-01T00:00:00Z", 'tests/test_data/ETH_USDT_4h_5000.parquet'),
    ('BTC/USDT', '1d',  3000, "2017-01-01T00:00:00Z", 'tests/test_data/BTC_USDT_d_3000.parquet'),
]

for symbol, tf, bars, start, path in tasks:
    print(f"\nFetching {symbol} {tf} {bars} bars from {start} ...")
    rows = fetch_ohlcv_full(symbol, tf, bars, start_iso=start)
    print(f"  Got {len(rows)} rows")
    save(rows, symbol, path)

print("\nDone.")
