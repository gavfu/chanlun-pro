"""
Check which pyarmor segments are pre-split vs split segments.
Then only care about comparing for the pre-split segments.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_pyarmor import CL as CL_Pyarmor
from chanlun.cl_open import CL as CL_Open

config = {
    "bi_type": "bi_type_old", "fx_qj": "fx_qj_k", "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes", "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1", "xd_bzh": "xd_bzh_no",
    "xd_zs_max_lines_split": 11, "xd_allow_split_no_highlow": 1,
    "xd_allow_split_zs_kz": 0, "xd_allow_split_zs_more_line": 1,
    "xd_allow_split_zs_no_direction": 1,
}

cases = [
    ("BTCd", "BTC_USDT_d_500.parquet", "d"),
    ("ETH60", "ETH_USDT_60m_1000.parquet", "60m"),
    ("BTC60", "BTC_USDT_60m_1000.parquet", "60m"),
    ("BTC5m", "BTC_USDT_5m_1000.parquet", "5m"),
]

for name, file, freq in cases:
    df = pd.read_parquet(f"tests/test_data/{file}")
    
    cl_p = CL_Pyarmor(name, freq, config)
    cl_p.process_klines(df)
    
    cl_o = CL_Open(name, freq, config)
    # Capture pre-split segments
    orig_split = cl_o._split_xds
    presplit = []
    def traced_split(xds, *a, **kw):
        for xd in xds:
            presplit.append((xd.type, xd.start_line.index, xd.end_line.index))
        return orig_split(xds, *a, **kw)
    cl_o._split_xds = traced_split
    cl_o.process_klines(df)
    
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Pre-split segments (our code):")
    for t, s, e in presplit:
        print(f"    {t} bi[{s}→{e}]")
    
    print(f"  Pyarmor segments:")
    for xd in cl_p.get_xds():
        split = f" split={xd.is_split}" if xd.is_split else ""
        print(f"    {xd.type} bi[{xd.start_line.index}→{xd.end_line.index}]{split}")
    
    presplit.clear()
