"""Check ALL gap values across ALL datasets to find the TRUE gap threshold pyarmor uses."""
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

datasets = {
    "BTC60": "tests/test_data/BTC_USDT_60m_1000.parquet",
    "ETH60": "tests/test_data/ETH_USDT_60m_1000.parquet",
    "BTC5m": "tests/test_data/BTC_USDT_5m_1000.parquet",
    "ETH5m": "tests/test_data/ETH_USDT_5m_1000.parquet",
    "BTCd": "tests/test_data/BTC_USDT_d_1000.parquet",
}

for name, path in datasets.items():
    df = pd.read_parquet(path)
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    bis_p = cd_p.get_bis()
    
    print(f"\n{'='*60}")
    print(f"=== {name}: Pyarmor BI gap analysis ===")
    min_cl = 999
    min_k = 999
    gap_pairs = []
    for bi in bis_p:
        cl = bi.end.k.index - bi.start.k.index
        k = bi.end.k.k_index - bi.start.k.k_index
        gap_pairs.append((cl, k))
        if cl < min_cl: min_cl = cl
        if k < min_k: min_k = k
    
    print(f"  Total BIs: {len(bis_p)}")
    print(f"  Min cl_gap: {min_cl},  Min k_gap: {min_k}")
    
    # Show distribution
    cl_counts = {}
    k_counts = {}
    for cl, k in gap_pairs:
        cl_counts[cl] = cl_counts.get(cl, 0) + 1
        k_counts[k] = k_counts.get(k, 0) + 1
    
    print(f"  cl_gap distribution (low end):")
    for g in sorted(cl_counts.keys()):
        if g <= 6:
            print(f"    cl_gap={g}: {cl_counts[g]} BIs")
    
    print(f"  k_gap distribution (low end):")
    for g in sorted(k_counts.keys()):
        if g <= 8:
            print(f"    k_gap={g}: {k_counts[g]} BIs")
    
    # Show BIs with cl_gap < 4
    low_cl = [(i, cl, k) for i, (cl, k) in enumerate(gap_pairs) if cl < 4]
    if low_cl:
        print(f"  BIs with cl_gap < 4:")
        for i, cl, k in low_cl:
            bi = bis_p[i]
            print(f"    bi[{i}]: {bi.type:4s} {bi.start.k.k_index}→{bi.end.k.k_index} "
                  f"cl={cl} k={k}")
    
    # Show BIs with k_gap < 4
    low_k = [(i, cl, k) for i, (cl, k) in enumerate(gap_pairs) if k < 4]
    if low_k:
        print(f"  BIs with k_gap < 4:")
        for i, cl, k in low_k:
            bi = bis_p[i]
            print(f"    bi[{i}]: {bi.type:4s} {bi.start.k.k_index}→{bi.end.k.k_index} "
                  f"cl={cl} k={k}")
    else:
        print(f"  No BIs with k_gap < 4")
    
    # Check: how many BIs have k_gap >= 4 but cl_gap < 4?
    k_pass_cl_fail = [(i, cl, k) for i, (cl, k) in enumerate(gap_pairs) if k >= 4 and cl < 4]
    if k_pass_cl_fail:
        print(f"  BIs with k_gap>=4 but cl_gap<4: {len(k_pass_cl_fail)}")
        for i, cl, k in k_pass_cl_fail:
            bi = bis_p[i]
            print(f"    bi[{i}]: {bi.type:4s} {bi.start.k.k_index}→{bi.end.k.k_index} "
                  f"cl={cl} k={k}")
