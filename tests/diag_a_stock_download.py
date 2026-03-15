"""
下载A股日线K线数据并保存为 parquet 文件
用法: python tests/diag_a_stock_download.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from pytdx.hq import TdxHq_API
from chanlun.tools.tdx_best_ip import select_best_ip


def download_tdx_klines(code: str, market: int, pages: int = 12):
    """
    从通达信下载日线K线数据
    code: 股票代码，如 '600030'
    market: 0=深圳, 1=上海
    pages: 页数，每页700条，12页约8400天(~33年)
    """
    # 获取最优服务器
    print("正在查找最优通达信服务器...")
    ip_info = select_best_ip("stock")
    print(f"  使用服务器: {ip_info['ip']}:{ip_info['port']}")

    client = TdxHq_API(raise_exception=True, auto_retry=True)
    with client.connect(ip_info['ip'], ip_info['port']):
        # frequency_map: d=9
        dfs = []
        for i in range(1, pages + 1):
            df = client.to_df(
                client.get_security_bars(9, market, code, (i - 1) * 700, 700)
            )
            if len(df) == 0:
                break
            dfs.append(df)
            print(f"  第 {i} 页: {len(df)} 条")

        ks = pd.concat(dfs, axis=0, sort=False)
        ks['date'] = pd.to_datetime(ks['datetime'])
        ks.sort_values('date', inplace=True)
        ks = ks.drop_duplicates(['date'], keep='last')

        # 标准化列名
        ks = ks.rename(columns={'vol': 'volume'})
        ks['code'] = f"SH.{code}" if market == 1 else f"SZ.{code}"

        # 只保留需要的列
        ks = ks[['code', 'date', 'open', 'close', 'high', 'low', 'volume']].reset_index(drop=True)
        return ks


def main():
    # 中信证券 600030，上海交易所 market=1
    code = '600030'
    market = 1
    full_code = 'SH.600030'

    out_path = os.path.join(os.path.dirname(__file__), 'test_data', f'{full_code.replace(".", "_")}_d.parquet')

    print(f"\n下载 {full_code} 日线数据...")
    df = download_tdx_klines(code, market, pages=4)  # 4页 ~2800天 (~11年)

    # 只保留最近约2年 (~500个交易日)
    if len(df) > 500:
        df = df.tail(500).reset_index(drop=True)

    print(f"\n数据概览:")
    print(f"  总条数: {len(df)}")
    print(f"  日期范围: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
    print(f"  最新价格: open={df['open'].iloc[-1]:.2f} high={df['high'].iloc[-1]:.2f} "
          f"low={df['low'].iloc[-1]:.2f} close={df['close'].iloc[-1]:.2f}")

    # 保存
    df.to_parquet(out_path)
    print(f"\n已保存到: {out_path}")
    print(f"文件大小: {os.path.getsize(out_path) / 1024:.1f} KB")


if __name__ == '__main__':
    main()
