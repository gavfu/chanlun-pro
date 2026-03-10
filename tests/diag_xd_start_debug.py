"""Debug _find_first_xd_start for all cases"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P
from chanlun.cl_interface import TZXL

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
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    
    bis_p = cd_p.process_klines(df).get_bis()
    
    # Show first few BIs
    print(f"\n=== {name} ===")
    print(f"  First 6 BIs:")
    for i in range(min(6, len(bis_p))):
        bi = bis_p[i]
        print(f"    bi[{i}] {bi.type:>4} h={bi.high:.1f} l={bi.low:.1f}")
    
    # Check pyarmor first segment
    xds_p = cd_p.get_xds()
    if xds_p:
        print(f"  Pyarmor first XD: {xds_p[0].type} bi[{xds_p[0].start_line.index}→{xds_p[0].end_line.index}]"
              f" ding_bad={xds_p[0].ding_fx.is_line_bad}")
    
    # Now trace the open algorithm
    cd_o.process_klines(df)
    bis = cd_o.get_bis()
    
    # Trace _find_first_tzxl_fx for DOWN BIs (ding)
    ding_result = cd_o._find_first_tzxl_fx(bis, "down", "up", "ding")
    print(f"  DOWN BI ding FX: {ding_result}")  # (bi_idx, is_bad)
    
    # Trace _find_first_tzxl_fx for UP BIs (di)
    di_result = cd_o._find_first_tzxl_fx(bis, "up", "down", "di")
    print(f"  UP BI di FX: {di_result}")
    
    # Show what _find_first_xd_start returns
    xd_type, start_idx = cd_o._find_first_xd_start(bis)
    print(f"  _find_first_xd_start → type={xd_type}, start_bi_idx={start_idx}")
    
    # Show open's first XD
    xds_o = cd_o.get_xds()
    if xds_o:
        print(f"  Open first XD: {xds_o[0].type} bi[{xds_o[0].start_line.index}→{xds_o[0].end_line.index}]")
