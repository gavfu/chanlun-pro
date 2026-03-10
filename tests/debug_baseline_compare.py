"""
Compare current (original) cl_open vs pyarmor, to see baseline.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor

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
    ("ETH5m", "ETH_USDT_5m_1000.parquet", "5m"),
]

for name, file, freq in cases:
    df = pd.read_parquet(f"tests/test_data/{file}")
    
    cl_o = CL_Open(name, freq, config)
    cl_o.process_klines(df)
    
    cl_p = CL_Pyarmor(name, freq, config)
    cl_p.process_klines(df)
    
    xds_o = cl_o.get_xds()
    xds_p = cl_p.get_xds()
    bis_o = cl_o.get_bis()
    bis_p = cl_p.get_bis()
    
    count_match = "✅" if len(xds_o) == len(xds_p) else "❌"
    
    print(f"\n{'='*80}")
    print(f"  {name}: BI={len(bis_o)}/{len(bis_p)}  XD={len(xds_o)}/{len(xds_p)} {count_match}")
    
    max_len = max(len(xds_o), len(xds_p))
    content_matches = 0
    for i in range(max_len):
        if i < len(xds_o) and i < len(xds_p):
            o = xds_o[i]
            p = xds_p[i]
            o_desc = f"{o.type} bi[{o.start_line.index}→{o.end_line.index}]"
            p_desc = f"{p.type} bi[{p.start_line.index}→{p.end_line.index}]"
            split_o = f" split={o.is_split}" if o.is_split else ""
            split_p = f" split={p.is_split}" if p.is_split else ""
            match = "✅" if (o.type == p.type and o.start_line.index == p.start_line.index and o.end_line.index == p.end_line.index) else "❌"
            if match == "✅":
                content_matches += 1
            print(f"  xd[{i:2d}] {match} open: {o_desc:25s}{split_o:30s}  pyarmor: {p_desc:25s}{split_p}")
        elif i < len(xds_o):
            o = xds_o[i]
            print(f"  xd[{i:2d}]    open: {o.type} bi[{o.start_line.index}→{o.end_line.index}]  pyarmor: MISSING")
        else:
            p = xds_p[i]
            print(f"  xd[{i:2d}]    open: MISSING  pyarmor: {p.type} bi[{p.start_line.index}→{p.end_line.index}]")
    
    total = min(len(xds_o), len(xds_p))
    print(f"  Content match: {content_matches}/{total}")
