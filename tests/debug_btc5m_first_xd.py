"""
Check: what is _determine_first_xd_start for BTC5m?
And what does _find_xd_end give for different starting positions?
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_interface import TZXL

config = {
    "bi_type": "bi_type_old", "fx_qj": "fx_qj_k", "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes", "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1", "xd_bzh": "xd_bzh_no",
    "xd_zs_max_lines_split": 11, "xd_allow_split_no_highlow": 1,
    "xd_allow_split_zs_kz": 0, "xd_allow_split_zs_more_line": 1,
    "xd_allow_split_zs_no_direction": 1,
}

df = pd.read_parquet("tests/test_data/BTC_USDT_5m_1000.parquet")
cl = CL_Open("BTC5m", "5m", config)
cl.process_klines(df)
bis = cl.get_bis()

# What does _determine_first_xd_start return?
first_start = cl._find_first_xd_start(bis)
print(f"_determine_first_xd_start: {first_start}")

# What are the first few BIs?
for i in range(min(15, len(bis))):
    b = bis[i]
    print(f"  bi[{i}] {b.type} high={b.high:.1f} low={b.low:.1f}")

# Try _find_xd_end from different starts
for start in range(0, 8):
    for xd_type in ["up", "down"]:
        result = cl._find_xd_end(bis, start, xd_type)
        if result:
            end = result[0]
            print(f"  _find_xd_end(bis, {start}, '{xd_type}') → end={end}")
