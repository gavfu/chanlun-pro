# -*- coding: utf-8 -*-
"""Count how often confirmation has a 'better' next endpoint, across all test cases"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_Open

DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
LIMITS = {
    "BTC/USDT_d": 500, "BTC/USDT_60m": 1000, "BTC/USDT_5m": 1000,
    "ETH/USDT_60m": 1000, "ETH/USDT_5m": 1000,
}

def load(symbol, freq):
    limit = LIMITS.get(f"{symbol}_{freq}", 1000)
    cache_key = f"{symbol.replace('/', '_')}_{freq}_{limit}"
    return pd.read_parquet(os.path.join(DATA_DIR, f"{cache_key}.parquet"))


for symbol, freq in [("BTC/USDT", "60m"), ("BTC/USDT", "5m"), ("BTC/USDT", "d"),
                      ("ETH/USDT", "60m"), ("ETH/USDT", "5m")]:
    df = load(symbol, freq)
    cd = CL_Open(symbol, freq, config={})
    cd.process_klines(df)

    fxs = list(cd.fxs)
    fx_idx = {fx.k.index: i for i, fx in enumerate(fxs)}

    # Simulate _build_bis and count look-ahead cases
    confirm_count = 0
    lookahead_count = 0
    lookahead_cases = []

    # Simple forward scan simulation
    i = 0
    while i < len(fxs) - 1:
        start_fx = fxs[i]
        end_fx = None
        j = i + 1
        while j < len(fxs):
            cur_fx = fxs[j]
            if end_fx is None:
                # Looking for end_fx
                if cur_fx.type != start_fx.type:
                    if cd._bi_fx_valid(start_fx, cur_fx):
                        end_fx = cur_fx
            else:
                if cur_fx.type == end_fx.type:
                    # Same type as end_fx: extension check
                    ext = False
                    if end_fx.type == "di" and cur_fx.val <= end_fx.val:
                        ext = True
                    elif end_fx.type == "ding" and cur_fx.val >= end_fx.val:
                        ext = True
                    if ext and cd._bi_fx_valid(start_fx, cur_fx):
                        end_fx = cur_fx
                else:
                    # Same type as start_fx: confirmation check
                    if cd._bi_fx_valid(end_fx, cur_fx):
                        confirm_count += 1

                        # Look-ahead: check if next FX of end_fx type is more extreme
                        has_better = False
                        if j + 1 < len(fxs):
                            next_fx = fxs[j + 1]
                            if next_fx.type == end_fx.type:
                                more_extreme = False
                                if end_fx.type == "di" and next_fx.val < end_fx.val:
                                    more_extreme = True
                                elif end_fx.type == "ding" and next_fx.val > end_fx.val:
                                    more_extreme = True
                                if more_extreme and cd._bi_fx_valid(start_fx, next_fx):
                                    has_better = True

                        if has_better:
                            lookahead_count += 1
                            next_fx = fxs[j + 1]
                            lookahead_cases.append({
                                'start': start_fx.k.index,
                                'end': end_fx.k.index,
                                'confirm': cur_fx.k.index,
                                'better': next_fx.k.index,
                                'end_val': end_fx.val,
                                'better_val': next_fx.val,
                            })

                        # BI confirmed, move to next
                        i = j - 1  # will be incremented
                        break
            j += 1
        else:
            break
        i += 1

    print(f"{symbol} {freq}: confirmations={confirm_count}, "
          f"with_better_next={lookahead_count}")
    for case in lookahead_cases:
        print(f"  start={case['start']} end={case['end']} "
              f"confirm={case['confirm']} better={case['better']} "
              f"end_val={case['end_val']:.2f} better_val={case['better_val']:.2f}")
