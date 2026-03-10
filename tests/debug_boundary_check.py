"""Check exact BI boundary differences remaining after the split fix."""
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

for name, data_path in TEST_DATA.items():
    df = pd.read_parquet(data_path)
    cd_o = CL_O("test", "test", config=CL_CONFIG)
    cd_o.process_klines(df)
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    
    bis_o = cd_o.get_bis()
    bis_p = cd_p.get_bis()
    
    # Count boundary matches
    n = min(len(bis_o), len(bis_p))
    matches = sum(1 for i in range(n) 
                  if bis_o[i].start.k.k_index == bis_p[i].start.k.k_index
                  and bis_o[i].end.k.k_index == bis_p[i].end.k.k_index)
    
    print(f"\n{name}: BI={len(bis_o)}/{len(bis_p)} boundary={matches}/{n}")
    
    # Show differences
    for i in range(n):
        bo = bis_o[i]
        bp = bis_p[i]
        if bo.start.k.k_index != bp.start.k.k_index or bo.end.k.k_index != bp.end.k.k_index:
            print(f"  bi[{i}] open: {bo.type} {bo.start.k.k_index}→{bo.end.k.k_index} "
                  f"pyarmor: {bp.type} {bp.start.k.k_index}→{bp.end.k.k_index}")
    
    if len(bis_o) != len(bis_p):
        print(f"  BI count differs: open={len(bis_o)} pyarmor={len(bis_p)}")
