"""Targeted hybrid: use cl_gap >= 4 normally, but when cl_gap < 4 AND k_gap >= 4,
apply additional condition to decide whether to accept.

The idea: pyarmor's gap check might be:
  if cl_gap < 4:
      if k_gap < 4: return False      # both fail → reject
      # cl_gap < 4 but k_gap >= 4 → conditional accept
      <some additional condition>
  
What could the condition be? Ideas:
1. Accept if k_gap >= some threshold (e.g., 5, 6, 7)
2. Accept only if the BI value difference exceeds some threshold
3. Accept only if both start and end are well-separated in raw K-lines
4. Accept but apply stricter strict check

Let me first check: across ALL 5 datasets, how many FX pairs have cl_gap<4 but k_gap>=4?
And which of these does pyarmor actually accept as BIs?"""
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

DATASETS = [
    ("BTCd", "tests/test_data/BTC_USDT_d_500.parquet"),
    ("ETH60", "tests/test_data/ETH_USDT_60m_1000.parquet"),
    ("BTC60", "tests/test_data/BTC_USDT_60m_1000.parquet"),
    ("BTC5m", "tests/test_data/BTC_USDT_5m_1000.parquet"),
    ("ETH5m", "tests/test_data/ETH_USDT_5m_1000.parquet"),
]

for name, path in DATASETS:
    df = pd.read_parquet(path)
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    bis_p = cd_p.get_bis()
    
    # Build set of pyarmor BI boundaries
    pya_bi_pairs = set()
    for bi in bis_p:
        pya_bi_pairs.add((bi.start.k.k_index, bi.end.k.k_index))
    
    # Also collect consecutive (end, next start) pairs as confirmation-BI pairs
    pya_confirm_pairs = set()
    for i in range(len(bis_p) - 1):
        # end of bi[i] = start of bi[i+1]
        # confirmation is: the start of bi[i+2] is the confirmation FX of bi[i]
        pass
    
    print(f"\n{'='*60}")
    print(f"=== {name}: Pyarmor has {len(bis_p)} BIs ===")
    
    # Check pyarmor BIs with small cl_gap
    small_cl = []
    for bi in bis_p:
        cl_gap = bi.end.k.index - bi.start.k.index
        k_gap = bi.end.k.k_index - bi.start.k.k_index
        if cl_gap < 4:
            small_cl.append((bi, cl_gap, k_gap))
            print(f"  bi[{bi.index}] {bi.type} {bi.start.k.k_index}→{bi.end.k.k_index} "
                  f"cl={cl_gap} k={k_gap}")
    
    if not small_cl:
        print("  No BIs with cl_gap < 4")
    
    # Count BIs with cl_gap < 4 per k_gap value
    by_kgap = {}
    for bi, cl, kg in small_cl:
        by_kgap.setdefault(kg, []).append(bi)
    for kg in sorted(by_kgap.keys()):
        print(f"  k_gap={kg}: {len(by_kgap[kg])} BIs")
    
    # Also track: for BIs with cl_gap < 4, what was the strict check result?
    cd_o = CL_O("test", "test", config=CL_CONFIG)
    cd_o.process_klines(df)
    fxs = cd_o.get_fxs()
    qj = cd_o.fx_qj
    qy = cd_o.fx_qy
    
    fx_by_k = {}
    for fx in fxs:
        fx_by_k[fx.k.k_index] = fx
    
    print(f"\n  Strict check for cl_gap<4 pyarmor BIs:")
    for bi, cl, kg in small_cl:
        sfx = fx_by_k.get(bi.start.k.k_index)
        efx = fx_by_k.get(bi.end.k.k_index)
        if sfx and efx:
            # Check strict
            strict = True
            if kg < 13:
                if sfx.type == "ding" and efx.type == "di":
                    if sfx.low(qj, qy) < efx.low(qj, qy): strict = False
                    elif efx.high(qj, qy) > sfx.high(qj, qy): strict = False
                elif sfx.type == "di" and efx.type == "ding":
                    if sfx.high(qj, qy) > efx.high(qj, qy): strict = False
                    elif efx.low(qj, qy) < sfx.low(qj, qy): strict = False
            print(f"    bi[{bi.index}] {bi.type} {bi.start.k.k_index}→{bi.end.k.k_index} "
                  f"cl={cl} k={kg} strict={strict}")
