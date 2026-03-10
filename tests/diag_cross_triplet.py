# -*- coding: utf-8 -*-
"""Check which cross-counting triplet triggers and compare to split points"""
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


for symbol, freq in [("BTC/USDT", "60m"), ("BTC/USDT", "5m"), ("BTC/USDT", "d"),
                      ("ETH/USDT", "60m"), ("ETH/USDT", "5m")]:
    df = load(symbol, freq)
    cd_o = CL_Open(symbol, freq, config={})
    cd_o.process_klines(df)
    cd_p = CL_Pyarmor(symbol, freq, config={})
    cd_p.process_klines(df)

    qj, qy = cd_o.fx_qj, cd_o.fx_qy
    threshold = cd_o.bi_split_k_cross_nums
    tolerance = cd_o.bi_split_k_cross_tolerance

    p_bis = list(cd_p.bis)
    i = 0
    while i < len(p_bis):
        bi = p_bis[i]
        if getattr(bi, 'is_split', False) and i + 2 < len(p_bis):
            parent_type = bi.type
            parent_start = bi.start.k.index
            parent_end = p_bis[i + 2].end.k.index

            if parent_type == "down":
                split1_idx = bi.end.k.index
                split2_idx = p_bis[i + 1].end.k.index
            else:
                split1_idx = bi.end.k.index
                split2_idx = p_bis[i + 1].end.k.index

            # Find the matching pre-split BI in cl_open
            # Look for a BI with same start, type that spans the range
            o_bi = None
            for ob in cd_o.bis:
                if ob.start.k.index == parent_start and ob.type == parent_type:
                    if ob.end.k.index >= parent_end - 2:  # allow small difference
                        o_bi = ob
                        break
            if o_bi is None:
                # Try finding by range
                for ob in cd_o.bis:
                    if ob.start.k.index == parent_start and ob.type == parent_type:
                        o_bi = ob
                        break

            if o_bi is None:
                print(f"\n{symbol} {freq}: {parent_type}[{parent_start}→{parent_end}] - NO MATCHING BI FOUND")
                i += 3
                continue

            start_idx = o_bi.start.k.index
            end_idx = o_bi.end.k.index
            end_ki = o_bi.end.k.k_index

            internal_fxs = [fx for fx in cd_o.fxs
                           if start_idx < fx.k.index < end_idx]

            # Replay cross-counting to find the triggering triplet
            triggered_triplet = None
            for ti in range(len(internal_fxs) - 2):
                fx1 = internal_fxs[ti]
                fx2 = internal_fxs[ti + 1]
                fx3 = internal_fxs[ti + 2]

                h1, l1 = fx1.high(qj, qy), fx1.low(qj, qy)
                h2, l2 = fx2.high(qj, qy), fx2.low(qj, qy)
                h3, l3 = fx3.high(qj, qy), fx3.low(qj, qy)

                hit_count = 0
                miss_count = 0
                for ki in range(fx1.k.k_index, end_ki):
                    k = cd_o.src_klines[ki]
                    if (k.h >= l1 and k.l <= h1
                            and k.h >= l2 and k.l <= h2
                            and k.h >= l3 and k.l <= h3):
                        hit_count += 1
                        miss_count = 0
                    else:
                        miss_count += 1
                    if miss_count > tolerance:
                        break

                if hit_count >= threshold:
                    triggered_triplet = (ti, fx1, fx2, fx3, hit_count)
                    break

            if triggered_triplet:
                ti, fx1, fx2, fx3, hits = triggered_triplet
                print(f"\n{symbol} {freq}: {parent_type}[{parent_start}→{parent_end}]")
                print(f"  Pyarmor splits: [{split1_idx}] [{split2_idx}]")
                print(f"  Triggered triplet index={ti}: "
                      f"fx1={fx1.type}[{fx1.k.index}] "
                      f"fx2={fx2.type}[{fx2.k.index}] "
                      f"fx3={fx3.type}[{fx3.k.index}] "
                      f"hits={hits}")

                # Show relationship between triplet and split points
                if parent_type == "down":
                    # split1=di, split2=ding
                    print(f"  Triplet types: {fx1.type},{fx2.type},{fx3.type}")
                    print(f"  Split1(di)={split1_idx} vs triplet: fx1={fx1.k.index}, fx2={fx2.k.index}, fx3={fx3.k.index}")
                    print(f"  Split2(ding)={split2_idx} vs triplet")
                else:
                    # split1=ding, split2=di
                    print(f"  Triplet types: {fx1.type},{fx2.type},{fx3.type}")
                    print(f"  Split1(ding)={split1_idx} vs triplet")
                    print(f"  Split2(di)={split2_idx} vs triplet")

                # Show all triplets that pass threshold
                print(f"  All passing triplets:")
                for tj in range(len(internal_fxs) - 2):
                    fx_a = internal_fxs[tj]
                    fx_b = internal_fxs[tj + 1]
                    fx_c = internal_fxs[tj + 2]
                    ha, la = fx_a.high(qj, qy), fx_a.low(qj, qy)
                    hb, lb = fx_b.high(qj, qy), fx_b.low(qj, qy)
                    hc, lc = fx_c.high(qj, qy), fx_c.low(qj, qy)
                    hc2 = 0
                    mc = 0
                    for ki in range(fx_a.k.k_index, end_ki):
                        k = cd_o.src_klines[ki]
                        if (k.h >= la and k.l <= ha
                                and k.h >= lb and k.l <= hb
                                and k.h >= lc and k.l <= hc):
                            hc2 += 1
                            mc = 0
                        else:
                            mc += 1
                        if mc > tolerance:
                            break
                    if hc2 >= threshold:
                        marker = " <<<FIRST" if tj == ti else ""
                        print(f"    ti={tj}: {fx_a.type}[{fx_a.k.index}] "
                              f"{fx_b.type}[{fx_b.k.index}] "
                              f"{fx_c.type}[{fx_c.k.index}] hits={hc2}{marker}")

            i += 3
        else:
            i += 1
