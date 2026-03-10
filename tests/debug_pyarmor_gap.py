"""Check minimum k_gap and cl_gap in pyarmor BIs to determine the gap rule."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
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

datasets = [
    ("BTCd", "tests/test_data/BTC_USDT_d_500.parquet"),
    ("ETH60", "tests/test_data/ETH_USDT_60m_1000.parquet"),
    ("BTC60", "tests/test_data/BTC_USDT_60m_1000.parquet"),
    ("BTC5m", "tests/test_data/BTC_USDT_5m_1000.parquet"),
    ("ETH5m", "tests/test_data/ETH_USDT_5m_1000.parquet"),
]

for name, path in datasets:
    df = pd.read_parquet(path)
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    
    bis = cd_p.get_bis()
    
    min_k_gap = float('inf')
    min_cl_gap = float('inf')
    small_gap_bis = []
    
    for bi in bis:
        cl_gap = bi.end.k.index - bi.start.k.index
        k_gap = bi.end.k.k_index - bi.start.k.k_index
        if k_gap < min_k_gap:
            min_k_gap = k_gap
        if cl_gap < min_cl_gap:
            min_cl_gap = cl_gap
        if cl_gap < 4:
            small_gap_bis.append((bi.index, bi.type, bi.start.k.k_index, bi.end.k.k_index, cl_gap, k_gap))
    
    print(f"{name}: {len(bis)} BIs, min_cl_gap={min_cl_gap}, min_k_gap={min_k_gap}")
    if small_gap_bis:
        print(f"  BIs with cl_gap < 4:")
        for idx, tp, sk, ek, cg, kg in small_gap_bis:
            print(f"    bi[{idx}] {tp} k={sk}→{ek} cl_gap={cg} k_gap={kg}")
