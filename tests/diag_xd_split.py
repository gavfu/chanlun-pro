"""
诊断：检查所有线段的 is_split 标记
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

TEST_DATA = {
    "BTCd":  "tests/test_data/BTC_USDT_d_500.parquet",
    "ETH60": "tests/test_data/ETH_USDT_60m_1000.parquet",
    "BTC60": "tests/test_data/BTC_USDT_60m_1000.parquet",
    "BTC5m": "tests/test_data/BTC_USDT_5m_1000.parquet",
    "ETH5m": "tests/test_data/ETH_USDT_5m_1000.parquet",
}

CL_CONFIG = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
}

for name, path in TEST_DATA.items():
    df = pd.read_parquet(path)
    cd_o = CL_O("test", "test", config=CL_CONFIG)
    cd_o.process_klines(df)
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    
    xds_p = cd_p.get_xds()
    xds_o = cd_o.get_xds()
    bis_p = cd_p.get_bis()
    
    print(f"\n=== {name} ===")
    print(f"  PYARMOR XDs ({len(xds_p)}):")
    for i, xd in enumerate(xds_p):
        ding_bad = xd.ding_fx.is_line_bad if xd.ding_fx else False
        di_bad = xd.di_fx.is_line_bad if xd.di_fx else False
        n_bis = xd.end_line.index - xd.start_line.index + 1
        print(f"    xd[{i:>2}] {xd.type:>4} bi[{xd.start_line.index:>2}→{xd.end_line.index:>2}] "
              f"({n_bis:>2} bis) done={xd.done:<5} is_split='{xd.is_split}' "
              f"ding_bad={ding_bad} di_bad={di_bad}")
    
    print(f"  OPEN XDs ({len(xds_o)}):")
    for i, xd in enumerate(xds_o):
        ding_bad = xd.ding_fx.is_line_bad if xd.ding_fx else False
        di_bad = xd.di_fx.is_line_bad if xd.di_fx else False
        n_bis = xd.end_line.index - xd.start_line.index + 1
        print(f"    xd[{i:>2}] {xd.type:>4} bi[{xd.start_line.index:>2}→{xd.end_line.index:>2}] "
              f"({n_bis:>2} bis) done={xd.done:<5} is_split='{xd.is_split}' "
              f"ding_bad={ding_bad} di_bad={di_bad}")
