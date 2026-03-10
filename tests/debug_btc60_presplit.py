"""
Check what _build_xds produces BEFORE _split_xds for BTC60.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_open import CL

df = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
config = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
    "xd_zs_max_lines_split": 11,
    "xd_allow_split_no_highlow": 1,
    "xd_allow_split_zs_kz": 0,
    "xd_allow_split_zs_more_line": 1,
    "xd_allow_split_zs_no_direction": 1,
}

cl = CL("BTC60", "60m", config)

# Monkey-patch _split_xds to capture pre-split segments
orig_split_xds = cl._split_xds
pre_split_xds = []

def traced_split_xds(xds, *args, **kwargs):
    for xd in xds:
        pre_split_xds.append(f"  {xd.type} bi[{xd.start_line.index}→{xd.end_line.index}]")
    return orig_split_xds(xds, *args, **kwargs)

cl._split_xds = traced_split_xds

cl.process_klines(df)

print("=== Pre-split segments ===")
for s in pre_split_xds:
    print(s)

print(f"\n=== Post-split segments ===")
for i, xd in enumerate(cl.get_xds()):
    split = f" split={xd.is_split}" if xd.is_split else ""
    print(f"  xd[{i}] {xd.type} bi[{xd.start_line.index}→{xd.end_line.index}]{split}")
