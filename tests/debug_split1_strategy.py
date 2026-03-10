"""Test hypothesis: pyarmor selects split1 as MOST EXTREME value among triplet-only candidates
(lowest di for DOWN, highest ding for UP) with gap_ok."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

CL_CONFIG = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
}

TEST_DATA = {
    "BTCd":  "tests/test_data/BTC_USDT_d_500.parquet",
    "ETH60": "tests/test_data/ETH_USDT_60m_1000.parquet",
    "BTC60": "tests/test_data/BTC_USDT_60m_1000.parquet",
    "BTC5m": "tests/test_data/BTC_USDT_5m_1000.parquet",
    "ETH5m": "tests/test_data/ETH_USDT_5m_1000.parquet",
}

for name, path in TEST_DATA.items():
    df = pd.read_parquet(path)
    
    # Get raw BIs (no split)
    cd_raw = CL_O("test", "test", config=CL_CONFIG)
    cd_raw.bi_split_k_cross_nums = 0
    cd_raw.process_klines(df)
    bis_raw = cd_raw.get_bis()
    fxs = cd_raw.get_fxs()
    
    # Get pyarmor BIs
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    bis_p = cd_p.get_bis()
    
    # Find split BIs
    qj = "fx_qj_k"; qy = "fx_qy_three"
    for j, bi in enumerate(bis_raw):
        start_idx = bi.start.k.index
        end_idx = bi.end.k.index
        internal = [fx for fx in fxs if start_idx < fx.k.index < end_idx]
        
        if len(internal) < 3:
            continue
        
        # Check for trigger
        triggered_ti = -1
        for ti in range(len(internal) - 2):
            fx1, fx2, fx3 = internal[ti], internal[ti+1], internal[ti+2]
            h1, l1 = fx1.high(qj, qy), fx1.low(qj, qy)
            h2, l2 = fx2.high(qj, qy), fx2.low(qj, qy)
            h3, l3 = fx3.high(qj, qy), fx3.low(qj, qy)
            hit, miss = 0, 0
            for ki in range(fx1.k.k_index, bi.end.k.k_index):
                k = cd_raw.src_klines[ki]
                if (k.h>=l1 and k.l<=h1 and k.h>=l2 and k.l<=h2 and k.h>=l3 and k.l<=h3):
                    hit += 1; miss = 0
                else:
                    miss += 1
                if miss > 1: break
            if hit >= 20:
                triggered_ti = ti
                break
        
        if triggered_ti < 0:
            continue
        
        trip = (internal[triggered_ti], internal[triggered_ti+1], internal[triggered_ti+2])
        
        if bi.type == "down":
            split1_type = "di"
            split2_type = "ding"
        else:
            split1_type = "ding"
            split2_type = "di"
        
        cands = [fx for fx in trip if fx.type == split1_type]
        cands.sort(key=lambda f: f.k.index)
        gap_ok = [f for f in cands if (f.k.k_index - bi.start.k.k_index) >= 4]
        
        # Current: first gap_ok
        if gap_ok:
            split1_first = gap_ok[0]
        else:
            split1_first = cands[-1]
        
        # Hypothesis: most extreme gap_ok
        if gap_ok:
            if bi.type == "down":
                split1_extreme = min(gap_ok, key=lambda f: f.val)
            else:
                split1_extreme = max(gap_ok, key=lambda f: f.val)
        else:
            split1_extreme = cands[-1]
        
        # Hypothesis: LAST gap_ok
        if gap_ok:
            split1_last = gap_ok[-1]
        else:
            split1_last = cands[-1]
        
        # Find pyarmor's split1
        # Look for the pyarmor BI that starts at bi.start and find its end
        pyarmor_split1_k = None
        for bp in bis_p:
            if bp.start.k.k_index == bi.start.k.k_index and bp.end.k.k_index != bi.end.k.k_index:
                pyarmor_split1_k = bp.end.k.k_index
                break
        
        if pyarmor_split1_k is None:
            continue
        
        first_match = split1_first.k.k_index == pyarmor_split1_k
        extreme_match = split1_extreme.k.k_index == pyarmor_split1_k
        last_match = split1_last.k.k_index == pyarmor_split1_k
        
        print(f"{name} raw[{j}] {bi.type} {bi.start.k.k_index}→{bi.end.k.k_index}:")
        print(f"  triplet: {trip[0].type}@{trip[0].k.k_index}, {trip[1].type}@{trip[1].k.k_index}, {trip[2].type}@{trip[2].k.k_index}")
        print(f"  gap_ok candidates: {[(f.k.k_index, f.val) for f in gap_ok]}")
        print(f"  pyarmor split1 = {pyarmor_split1_k}")
        print(f"  first={split1_first.k.k_index}{'✓' if first_match else '✗'} "
              f"extreme={split1_extreme.k.k_index}{'✓' if extreme_match else '✗'} "
              f"last={split1_last.k.k_index}{'✓' if last_match else '✗'}")
        print()
