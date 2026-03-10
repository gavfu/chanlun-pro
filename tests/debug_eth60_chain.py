"""Trace the full segment chain for ETH60 to see which _find_xd_end calls are made"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_interface import TZXL

CL_CONFIG = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
}

original_find_xd_end = CL_O._find_xd_end

calls = []

def trace_find_xd_end(self, bis, start_bi_idx, xd_type):
    result = original_find_xd_end(self, bis, start_bi_idx, xd_type)
    if result:
        end_bi_idx, ding_fx, di_fx, tzxls = result
        target_fx = ding_fx if xd_type == "up" else di_fx
        is_bad = target_fx.is_line_bad if target_fx else False
        calls.append((start_bi_idx, xd_type, end_bi_idx, is_bad))
    else:
        calls.append((start_bi_idx, xd_type, None, None))
    return result

CL_O._find_xd_end = trace_find_xd_end

df = pd.read_parquet("tests/test_data/ETH_USDT_60m_1000.parquet")
cd = CL_O("test", "test", config=CL_CONFIG)
cd.process_klines(df)

print("=== _find_xd_end calls for ETH60 ===")
for start, tp, end, bad in calls:
    print(f"  {tp:>4s} from bi[{start}] → end={end} bad={bad}")

print("\n=== Final segments ===")
xds = cd.get_xds()
for i, xd in enumerate(xds):
    print(f"  xd[{i}] {xd.type:>4s} bi[{xd.start_line.index}->{xd.end_line.index}] split=[{xd.is_split}]")
