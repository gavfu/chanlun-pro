"""
Look at pyarmor's XD objects to see what ding_fx and di_fx are stored for each XD.
These tell us exactly which FX pyarmor chose.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_pyarmor import CL as CL_Pyarmor

config = {
    "bi_type": "bi_type_old", "fx_qj": "fx_qj_k", "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes", "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1", "xd_bzh": "xd_bzh_no",
    "xd_zs_max_lines_split": 11, "xd_allow_split_no_highlow": 1,
    "xd_allow_split_zs_kz": 0, "xd_allow_split_zs_more_line": 1,
    "xd_allow_split_zs_no_direction": 1,
}

datasets = [
    ("BTC60", "tests/test_data/BTC_USDT_60m_1000.parquet", "60m"),
    ("BTC5m", "tests/test_data/BTC_USDT_5m_1000.parquet", "5m"),
    ("ETH60", "tests/test_data/ETH_USDT_60m_1000.parquet", "60m"),
]

for name, path, freq in datasets:
    df = pd.read_parquet(path)
    cl = CL_Pyarmor("test", freq, config)
    cl.process_klines(df)
    xds = cl.get_xds()
    
    print(f"\n{'='*80}")
    print(f"{name}: {len(xds)} XDs")
    print(f"{'='*80}")
    
    for xd in xds:
        ding_info = "None"
        di_info = "None"
        
        if xd.ding_fx is not None:
            fx = xd.ding_fx
            fx_bis = ",".join(str(l.index) for l in fx.xl.lines)
            ding_info = f"bi[{fx_bis}] bad={fx.is_line_bad} xl_bad={fx.xl.line_bad} bh={fx.bh_type}"
        
        if xd.di_fx is not None:
            fx = xd.di_fx
            fx_bis = ",".join(str(l.index) for l in fx.xl.lines)
            di_info = f"bi[{fx_bis}] bad={fx.is_line_bad} xl_bad={fx.xl.line_bad} bh={fx.bh_type}"
        
        split = ""
        if hasattr(xd, 'is_split') and xd.is_split:
            split = " [SPLIT]"
        
        print(f"  xd[{xd.index}] {xd.type} bi[{xd.start_line.index}→{xd.end_line.index}]{split}")
        print(f"    ding: {ding_info}")
        print(f"    di:   {di_info}")
