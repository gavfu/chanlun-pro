"""Check pyarmor XLFX details for each segment"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_pyarmor import CL as CL_P
from chanlun.cl_interface import *

TEST_DATA = {
    "BTCd":  "tests/test_data/BTC_USDT_d_500.parquet",
    "ETH60": "tests/test_data/ETH_USDT_60m_1000.parquet",
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

for name, data_path in TEST_DATA.items():
    df = pd.read_parquet(data_path)
    cl = CL_P("test", "test", config=CL_CONFIG)
    cl.process_klines(df)
    
    xds = cl.get_xds()
    print(f"\n{'='*80}")
    print(f"=== {name} ({len(xds)} segments) ===")
    
    for i, xd in enumerate(xds[:5]):  # Only first 5
        print(f"\n  xd[{i}] {xd.type} bi[{xd.start.index}→{xd.end.index}] is_split='{xd.is_split}'")
        if xd.ding_fx:
            fx = xd.ding_fx
            lines = [l.index for l in fx.xl.lines] if fx.xl else []
            print(f"    ding_fx: is_line_bad={fx.is_line_bad} xl.line_bad={fx.xl.line_bad if fx.xl else '-'} xl.lines={lines}")
            if fx.xls:
                for j, xle in enumerate(fx.xls):
                    if xle:
                        xl_lines = [l.index for l in xle.lines]
                        print(f"      xls[{j}]: max={xle.max:.1f} min={xle.min:.1f} bad={xle.line_bad} lines={xl_lines}")
        if xd.di_fx:
            fx = xd.di_fx
            lines = [l.index for l in fx.xl.lines] if fx.xl else []
            print(f"    di_fx: is_line_bad={fx.is_line_bad} xl.line_bad={fx.xl.line_bad if fx.xl else '-'} xl.lines={lines}")
            if fx.xls:
                for j, xle in enumerate(fx.xls):
                    if xle:
                        xl_lines = [l.index for l in xle.lines]
                        print(f"      xls[{j}]: max={xle.max:.1f} min={xle.min:.1f} bad={xle.line_bad} lines={xl_lines}")
