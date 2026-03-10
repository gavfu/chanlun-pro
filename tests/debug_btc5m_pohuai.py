"""Check bi_pohuai for BTC5m TZXL DI FX candidates"""
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

df = pd.read_parquet("tests/test_data/BTC_USDT_5m_1000.parquet")
cd = CL_O("test", "test", config=CL_CONFIG)
cd.process_klines(df)

bis = cd.get_bis()

# Check bi_pohuai for down segment starting at bi[46]
# For DOWN, bi_pohuai checks if the BI AFTER the TZXL's last line breaks the start value
# Start value for down bi[46] is bi[46].high = 68524.9

start_bi = bis[46]
print(f"Segment start: bi[{start_bi.index}] high={start_bi.high} (start value for down)")

# TZXL[1] last line is bi[51], check bi[52]
# For DOWN, pohuai means the next UP bi's high > start_high
print(f"\nTZXL[1] last line: bi[51]. Check bi[52]:")
bi52 = bis[52]
print(f"  bi[52] type={bi52.type} high={bi52.high} low={bi52.low}")
print(f"  bi52.high({bi52.high}) > start_high({start_bi.high})? {bi52.high > start_bi.high}")

# TZXL[3] last line is bi[61], check bi[62]
print(f"\nTZXL[3] last line: bi[61]. Check bi[62]:")
bi62 = bis[62]
print(f"  bi[62] type={bi62.type} high={bi62.high} low={bi62.low}")
print(f"  bi62.high({bi62.high}) > start_high({start_bi.high})? {bi62.high > start_bi.high}")

# TZXL[5] last line is bi[65], check bi[66]
print(f"\nTZXL[5] last line: bi[65]. Check bi[66]:")
bi66 = bis[66]
print(f"  bi[66] type={bi66.type} high={bi66.high} low={bi66.low}")
print(f"  bi66.high({bi66.high}) > start_high({start_bi.high})? {bi66.high > start_bi.high}")

# Also check actual _check_xd_bi_pohuai
from chanlun.cl_interface import TZXL
# Simulate TZXL for bi_pohuai check
print("\n--- Checking _check_xd_bi_pohuai ---")
# For BTC5m down bi[46], TZXL[1] lines=[49,51]
tzxl1 = TZXL(bh_direction="down", line=bis[49], pre_line=bis[48], line_bad=False, done=True)
tzxl1.lines.append(bis[51])
tzxl1.update_maxmin()
result1 = cd._check_xd_bi_pohuai(bis, 46, tzxl1, "down")
print(f"TZXL[1] (lines=[49,51]): bi_pohuai={result1}")

# TZXL[3] lines=[57,59,61]
tzxl3 = TZXL(bh_direction="down", line=bis[57], pre_line=bis[56], line_bad=False, done=True)
tzxl3.lines.append(bis[59])
tzxl3.lines.append(bis[61])
tzxl3.update_maxmin()
result3 = cd._check_xd_bi_pohuai(bis, 46, tzxl3, "down")
print(f"TZXL[3] (lines=[57,59,61]): bi_pohuai={result3}")

# TZXL[5] lines=[65]
tzxl5 = TZXL(bh_direction="down", line=bis[65], pre_line=bis[64], line_bad=False, done=True)
result5 = cd._check_xd_bi_pohuai(bis, 46, tzxl5, "down")
print(f"TZXL[5] (lines=[65]): bi_pohuai={result5}")
