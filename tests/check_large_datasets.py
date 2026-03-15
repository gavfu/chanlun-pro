"""快速检查大数据集的 XD/ZSD 数量，确认有非零值"""
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

DATASETS = {
    "BTC4h_5000": "tests/test_data/BTC_USDT_4h_5000.parquet",
    "ETH4h_5000": "tests/test_data/ETH_USDT_4h_5000.parquet",
    "BTCd_3000":  "tests/test_data/BTC_USDT_d_3000.parquet",
}

for name, path in DATASETS.items():
    df = pd.read_parquet(path)
    cl_o = CL_O("BTC/USDT", "4h", CL_CONFIG)
    cl_p = CL_P("BTC/USDT", "4h", CL_CONFIG)
    cl_o.process_klines(df)
    cl_p.process_klines(df)
    print(f"{name}: bis={len(cl_o.bis)}/{len(cl_p.bis)}  "
          f"xds={len(cl_o.xds)}/{len(cl_p.xds)}  "
          f"zsds={len(cl_o.zsds)}/{len(cl_p.zsds)}  "
          f"qsds={len(cl_o.qsds)}/{len(cl_p.qsds)}")
