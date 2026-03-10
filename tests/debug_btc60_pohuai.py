"""Check bi_pohuai for BTC60 TZXL DI FX candidates"""
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

df = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cd = CL_O("test", "test", config=CL_CONFIG)
cd.process_klines(df)

bis = cd.get_bis()

start_bi = bis[28]
print(f"Segment start: bi[{start_bi.index}] high={start_bi.high} (start value for down)")

# TZXL[5] lines=[39], check _check_xd_bi_pohuai
tzxl5 = TZXL(bh_direction="down", line=bis[39], pre_line=bis[38], line_bad=True, done=True)
result5 = cd._check_xd_bi_pohuai(bis, 28, tzxl5, "down")
print(f"\nTZXL[5] (lines=[39]): bi_pohuai={result5}")
print(f"  Next BI after last line bi[39] is bi[40]:")
bi40 = bis[40]
print(f"  bi[40] type={bi40.type} high={bi40.high} low={bi40.low}")

# TZXL[8] lines=[45,47]
tzxl8 = TZXL(bh_direction="down", line=bis[45], pre_line=bis[44], line_bad=False, done=True)
tzxl8.lines.append(bis[47])
tzxl8.update_maxmin()
result8 = cd._check_xd_bi_pohuai(bis, 28, tzxl8, "down")
print(f"\nTZXL[8] (lines=[45,47]): bi_pohuai={result8}")
print(f"  Next BI after last line bi[47] is bi[48]:")
bi48 = bis[48]
print(f"  bi[48] type={bi48.type} high={bi48.high} low={bi48.low}")
print(f"  bi48.high({bi48.high}) > start_high({start_bi.high})? {bi48.high > start_bi.high}")
