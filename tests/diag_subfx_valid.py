# -*- coding: utf-8 -*-
"""Check _bi_fx_valid for each sub-BI of pyarmor's split AND for alternative candidates"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor

DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
LIMITS = {
    "BTC/USDT_d": 500, "BTC/USDT_60m": 1000, "BTC/USDT_5m": 1000,
    "ETH/USDT_60m": 1000, "ETH/USDT_5m": 1000,
}


def load(symbol, freq):
    limit = LIMITS.get(f"{symbol}_{freq}", 1000)
    cache_key = f"{symbol.replace('/', '_')}_{freq}_{limit}"
    return pd.read_parquet(os.path.join(DATA_DIR, f"{cache_key}.parquet"))


def check_bi_valid(cd, start_fx, end_fx):
    """Check _bi_fx_valid and also return gap info"""
    cl_gap = end_fx.k.index - start_fx.k.index
    k_gap = end_fx.k.k_index - start_fx.k.k_index
    valid = cd._bi_fx_valid(start_fx, end_fx)
    return valid, cl_gap, k_gap


for symbol, freq in [("BTC/USDT", "60m"), ("BTC/USDT", "5m"), ("BTC/USDT", "d"),
                      ("ETH/USDT", "60m"), ("ETH/USDT", "5m")]:
    df = load(symbol, freq)
    cd_o = CL_Open(symbol, freq, config={})
    cd_o.process_klines(df)
    cd_p = CL_Pyarmor(symbol, freq, config={})
    cd_p.process_klines(df)

    p_bis = list(cd_p.bis)
    fxs = {fx.k.index: fx for fx in cd_o.fxs}

    i = 0
    while i < len(p_bis):
        bi = p_bis[i]
        if getattr(bi, 'is_split', False) and i + 2 < len(p_bis):
            parent = bi
            parent_start = bi.start.k.index
            parent_end = p_bis[i + 2].end.k.index
            parent_type = bi.type
            split1_idx = bi.end.k.index
            split2_idx = p_bis[i + 1].end.k.index

            start_fx = fxs.get(parent_start)
            end_fx = fxs.get(parent_end)
            split1_fx = fxs.get(split1_idx)
            split2_fx = fxs.get(split2_idx)

            if not all([start_fx, end_fx, split1_fx, split2_fx]):
                i += 3
                continue

            # Check sub-BI validity for pyarmor's selection
            v1, g1, k1 = check_bi_valid(cd_o, start_fx, split1_fx)
            v2, g2, k2 = check_bi_valid(cd_o, split1_fx, split2_fx)
            v3, g3, k3 = check_bi_valid(cd_o, split2_fx, end_fx)

            print(f"\n{symbol} {freq}: {parent_type}[{parent_start}→{parent_end}]")
            print(f"  Pyarmor splits: [{split1_idx}] [{split2_idx}]")
            print(f"  Sub-BI 1: start→[{split1_idx}]: valid={v1} cl_gap={g1} k_gap={k1}")
            print(f"  Sub-BI 2: [{split1_idx}]→[{split2_idx}]: valid={v2} cl_gap={g2} k_gap={k2}")
            print(f"  Sub-BI 3: [{split2_idx}]→end: valid={v3} cl_gap={g3} k_gap={k3}")

            # Check alternatives: first few FXs by position
            internal = [fx for fx in cd_o.fxs if parent_start < fx.k.index < parent_end]
            if parent_type == "down":
                first_type = "di"
                second_type = "ding"
            else:
                first_type = "ding"
                second_type = "di"

            alt_first = sorted([fx for fx in internal if fx.type == first_type], key=lambda f: f.k.index)
            alt_second = sorted([fx for fx in internal if fx.type == second_type], key=lambda f: f.k.index)

            print(f"  Alternative first candidates:")
            for fx in alt_first[:5]:
                v, g, k = check_bi_valid(cd_o, start_fx, fx)
                chosen = " <<<" if fx.k.index == split1_idx else ""
                print(f"    [{fx.k.index}] valid={v} cl_gap={g} k_gap={k} val={fx.val:.2f}{chosen}")

                # For each first candidate, show first few second candidates with gap_mid validity
                for sfx in alt_second:
                    if sfx.k.index <= fx.k.index:
                        continue
                    vm, gm, km = check_bi_valid(cd_o, fx, sfx)
                    ve, ge, ke = check_bi_valid(cd_o, sfx, end_fx)
                    ding_gt_di = (sfx.val > fx.val) if parent_type == "down" else (fx.val > sfx.val)
                    if gm >= 4 and ding_gt_di:
                        chosen2 = " <<<" if (fx.k.index == split1_idx and sfx.k.index == split2_idx) else ""
                        print(f"      →[{sfx.k.index}] mid_valid={vm} cl_gap_m={gm} k_gap_m={km} "
                              f"end_valid={ve} cl_gap_e={ge} k_gap_e={ke} ding>di={ding_gt_di}{chosen2}")
                        break  # just show first valid second candidate

            i += 3
        else:
            i += 1
