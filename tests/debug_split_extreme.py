"""Test hypothesis: pyarmor picks split pair where split1 has the most extreme value
among all gap_ok-valid pairs."""
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

qj = "fx_qj_k"; qy = "fx_qy_three"

for name, path in TEST_DATA.items():
    df = pd.read_parquet(path)
    
    cd_raw = CL_O("test", "test", config=CL_CONFIG)
    cd_raw.bi_split_k_cross_nums = 0
    cd_raw.process_klines(df)
    bis_raw = cd_raw.get_bis()
    fxs = cd_raw.get_fxs()
    
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    bis_p = cd_p.get_bis()
    
    for j, bi in enumerate(bis_raw):
        start_ci = bi.start.k.index
        end_ci = bi.end.k.index
        internal = [fx for fx in fxs if start_ci < fx.k.index < end_ci]
        
        if len(internal) < 3:
            continue
        
        # Check for trigger
        triggered = False
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
                triggered = True
                break
        
        if not triggered:
            continue
        
        # Find pyarmor's split points
        pyarmor_s1 = pyarmor_s2 = None
        for bp in bis_p:
            if bp.start.k.k_index == bi.start.k.k_index and bp.end.k.k_index != bi.end.k.k_index:
                pyarmor_s1 = bp.end.k.k_index
                # Find next BI
                idx = bis_p.index(bp)
                if idx + 1 < len(bis_p):
                    pyarmor_s2 = bis_p[idx + 1].end.k.k_index
                break
        
        if pyarmor_s1 is None:
            continue
        
        # Find ALL valid split pairs
        if bi.type == "down":
            s1_type, s2_type = "di", "ding"
        else:
            s1_type, s2_type = "ding", "di"
        
        s1_cands = [fx for fx in internal if fx.type == s1_type]
        s2_cands = [fx for fx in internal if fx.type == s2_type]
        
        valid_pairs = []
        for s1 in s1_cands:
            if not cd_raw._split_gap_ok(bi.start, s1):
                continue
            for s2 in s2_cands:
                if s2.k.index <= s1.k.index:
                    continue
                if not cd_raw._split_gap_ok(s1, s2):
                    continue
                if not cd_raw._split_gap_ok(s2, bi.end):
                    continue
                # Direction check
                if s2_type == "ding" and s2.val <= s1.val:
                    continue
                if s2_type == "di" and s1.val <= s2.val:
                    continue
                valid_pairs.append((s1, s2))
        
        if not valid_pairs:
            continue
        
        # Strategy: most extreme split1
        if bi.type == "down":
            best = min(valid_pairs, key=lambda p: p[0].val)
        else:
            best = max(valid_pairs, key=lambda p: p[0].val)
        
        match = best[0].k.k_index == pyarmor_s1 and best[1].k.k_index == pyarmor_s2
        
        if not match:
            print(f"{name} raw[{j}] {bi.type} {bi.start.k.k_index}→{bi.end.k.k_index}: "
                  f"extreme=({best[0].k.k_index},{best[1].k.k_index}) "
                  f"pyarmor=({pyarmor_s1},{pyarmor_s2}) ✗ MISMATCH")
            # Check what strategy would work
            # Try: most extreme split1, then most extreme split2
            s1_match = [p for p in valid_pairs if p[0].k.k_index == pyarmor_s1]
            if s1_match:
                s2_options = [p[1] for p in s1_match]
                print(f"    s2 options with pyarmor s1={pyarmor_s1}: {[(s.k.k_index, s.val) for s in s2_options]}")
        else:
            print(f"{name} raw[{j}] {bi.type} {bi.start.k.k_index}→{bi.end.k.k_index}: "
                  f"extreme=({best[0].k.k_index},{best[1].k.k_index}) ✓")
