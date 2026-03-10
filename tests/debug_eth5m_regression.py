"""
Debug ETH5m regression: 12→13 XDs.
Check pre-split vs post-split XD lists.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O

CL_CONFIG = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
}

df = pd.read_parquet("tests/test_data/ETH_USDT_5m_1000.parquet")
cd = CL_O("test", "test", config=CL_CONFIG)

# Monkey-patch to capture pre-split XDs
original_split_xds = cd._split_xds

def patched_split_xds(xds, bis):
    print("=== PRE-SPLIT XDs ===")
    for i, xd in enumerate(xds):
        print(f"  xd[{i}] {xd.type:>4} bi[{xd.start_line.index}→{xd.end_line.index}] done={xd.done}")
    
    result = original_split_xds(xds, bis)
    
    print("\n=== POST-SPLIT XDs ===")
    for i, xd in enumerate(result):
        marker = ""
        if hasattr(xd, 'split_reason') and xd.split_reason:
            marker = f" [{xd.split_reason}]"
        print(f"  xd[{i}] {xd.type:>4} bi[{xd.start_line.index}→{xd.end_line.index}] done={xd.done}{marker}")
    
    # Check consecutive same direction
    for i in range(1, len(result)):
        if result[i].type == result[i-1].type:
            print(f"\n  *** CONSECUTIVE SAME DIRECTION: xd[{i-1}] and xd[{i}] both '{result[i].type}' ***")
    
    return result

cd._split_xds = patched_split_xds
cd.process_klines(df)
