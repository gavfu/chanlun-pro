"""
检查 cl_pyarmor 的 zsds 行为
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_pyarmor import CL as CL_P
from chanlun.cl_open import CL as CL_O

CL_CONFIG = {
    'bi_type': 'bi_type_old', 'fx_qj': 'fx_qj_k', 'fx_qy': 'fx_qy_three',
    'bi_fx_cgd': 'bi_fx_cgd_yes', 'fx_check_k_nums': 13,
    'bi_split_k_cross_nums': '20,1', 'xd_bzh': 'xd_bzh_no',
}

TEST_DATA = {
    "BTCd":  "tests/test_data/BTC_USDT_d_500.parquet",
    "ETH60": "tests/test_data/ETH_USDT_60m_1000.parquet",
    "BTC60": "tests/test_data/BTC_USDT_60m_1000.parquet",
    "BTC5m": "tests/test_data/BTC_USDT_5m_1000.parquet",
    "ETH5m": "tests/test_data/ETH_USDT_5m_1000.parquet",
    "BTC60_500": "tests/test_data/BTC_USDT_60m_500.parquet",
}

for ds_name, path in TEST_DATA.items():
    df = pd.read_parquet(path)
    cd_p = CL_P('test', 'test', config=CL_CONFIG)
    cd_p.process_klines(df)
    cd_o = CL_O('test', 'test', config=CL_CONFIG)
    cd_o.process_klines(df)

    xds_p = cd_p.get_xds()
    xd_zss_p = cd_p.get_xd_zss()
    zsds_p = cd_p.get_zsds()
    zsds_o = cd_o.get_zsds()

    print(f"\n{'='*70}")
    print(f"{ds_name}: xds={len(xds_p)} xd_zss={len(xd_zss_p)} "
          f"zsds_pyarmor={len(zsds_p)} zsds_open={len(zsds_o)}")
    print(f"  XDs: ", end="")
    for xd in xds_p:
        print(f"{'↑' if xd.type=='up' else '↓'}[{xd.start_line.index}→{xd.end_line.index}]", end=" ")
    print()
    print(f"  xd_zss: ", end="")
    for zs in xd_zss_p:
        print(f"(zg={zs.zg:.2f},zd={zs.zd:.2f},n={zs.line_num})", end=" ")
    print()
    if zsds_p:
        print(f"  zsds_p: ", end="")
        for zsd in zsds_p:
            print(f"{'↑' if zsd.type=='up' else '↓'}[xd{zsd.start_line.index}→xd{zsd.end_line.index}]", end=" ")
        print()
    if zsds_o:
        print(f"  zsds_o: ", end="")
        for zsd in zsds_o:
            sli = zsd.start_line.index if hasattr(zsd.start_line, 'index') else '?'
            eli = zsd.end_line.index if hasattr(zsd.end_line, 'index') else '?'
            print(f"{'↑' if zsd.type=='up' else '↓'}[xd{sli}→xd{eli}] done={zsd.done}", end=" ")
        print()
