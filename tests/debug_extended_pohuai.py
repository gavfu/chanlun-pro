"""
Test: does an EXTENDED bi-pohuai check (looking at more BIs after the FX)
explain why BTC5m bi[6] is skipped?

Hypothesis: pyarmor checks not just the FIRST BI after the FX, but looks
at the next BI of the CORRECT TYPE (for up segment: the next DOWN BI).
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

def check_extended_bi_pohuai(bis, start_bi_idx, target_xl, xd_type, check_count=3):
    """Extended bi-pohuai: check up to check_count BIs after the target."""
    start_bi = bis[start_bi_idx]
    last_line_idx = target_xl.lines[-1].index
    
    checked = 0
    for bi_idx in range(last_line_idx + 1, len(bis)):
        bi = bis[bi_idx]
        if xd_type == "up":
            if bi.type == "down" and bi.low < start_bi.low:
                return True, bi_idx, bi.low, start_bi.low
        else:
            if bi.type == "up" and bi.high > start_bi.high:
                return True, bi_idx, bi.high, start_bi.high
        checked += 1
        if checked >= check_count:
            break
    return False, None, None, None


# BTC5m
df5m = pd.read_parquet("tests/test_data/BTC_USDT_5m_1000.parquet")
cl5m = CL_Open("test", "5m", config)
cl5m.process_klines(df5m)
bis5m = cl5m.get_bis()

print("=== BTC5m up[3]: bi-pohuai analysis ===")
start_bi = bis5m[3]
print(f"  start_bi = bis[3]: type={start_bi.type} high={start_bi.high:.1f} low={start_bi.low:.1f}")

# Show BIs around the region
for i in range(3, 15):
    bi = bis5m[i]
    print(f"  bi[{i}]: {bi.type} h={bi.high:.1f} l={bi.low:.1f}")

# Current check for bi[6]
print(f"\n  Current bi-pohuai for FX@bi[6] (checks only bi[7]):")
bi6_tz = TZXL(bh_direction="up", line=bis5m[6], pre_line=bis5m[5], line_bad=True, done=True)
result = cl5m._check_xd_bi_pohuai(bis5m, 3, bi6_tz, "up")
print(f"    result = {result}")

# Extended check for bi[6]
print(f"\n  Extended bi-pohuai for FX@bi[6] (checks multiple BIs):")
for count in [1, 2, 3, 5]:
    ph, bi_idx, val, start_val = check_extended_bi_pohuai(bis5m, 3, bi6_tz, "up", count)
    print(f"    check_count={count}: pohuai={ph} "
          f"{'at bi['+str(bi_idx)+'] val='+str(val)+' start='+str(start_val) if ph else ''}")

# Also check bi[10]
print(f"\n  Extended bi-pohuai for FX@bi[10]:")
bi10_tz = TZXL(bh_direction="up", line=bis5m[10], pre_line=bis5m[9], line_bad=False, done=True)
for count in [1, 2, 3, 5]:
    ph, bi_idx, val, start_val = check_extended_bi_pohuai(bis5m, 3, bi10_tz, "up", count)
    print(f"    check_count={count}: pohuai={ph} "
          f"{'at bi['+str(bi_idx)+'] val='+str(val)+' start='+str(start_val) if ph else ''}")


# BTC60
print("\n\n=== BTC60 down[28]: bi-pohuai analysis ===")
df60 = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cl60 = CL_Open("test", "60m", config)
cl60.process_klines(df60)
bis60 = cl60.get_bis()

start_bi28 = bis60[28]
print(f"  start_bi = bis[28]: type={start_bi28.type} high={start_bi28.high:.1f} low={start_bi28.low:.1f}")

for i in range(28, 45):
    bi = bis60[i]
    print(f"  bi[{i}]: {bi.type} h={bi.high:.1f} l={bi.low:.1f}")

# Extended check for bi[39]
print(f"\n  Extended bi-pohuai for FX@bi[39]:")
bi39_tz = TZXL(bh_direction="down", line=bis60[39], pre_line=bis60[38], line_bad=True, done=True)
for count in [1, 2, 3, 5]:
    ph, bi_idx, val, start_val = check_extended_bi_pohuai(bis60, 28, bi39_tz, "down", count)
    print(f"    check_count={count}: pohuai={ph} "
          f"{'at bi['+str(bi_idx)+'] val='+str(val)+' start='+str(start_val) if ph else ''}")

# BTC60 up[39]
print("\n\n=== BTC60 up[39]: bi-pohuai analysis ===")
start_bi39 = bis60[39]
print(f"  start_bi = bis[39]: type={start_bi39.type} high={start_bi39.high:.1f} low={start_bi39.low:.1f}")

# Extended check for bi[42]
print(f"\n  Extended bi-pohuai for FX@bi[42]:")
bi42_tz = TZXL(bh_direction="up", line=bis60[42], pre_line=bis60[41], line_bad=True, done=True)
for count in [1, 2, 3, 5]:
    ph, bi_idx, val, start_val = check_extended_bi_pohuai(bis60, 39, bi42_tz, "up", count)
    print(f"    check_count={count}: pohuai={ph} "
          f"{'at bi['+str(bi_idx)+'] val='+str(val)+' start='+str(start_val) if ph else ''}")


# What about "next same-type BI" check?
print("\n\n=== Alternative: check NEXT SAME-TYPE BI after FX ===")
def check_next_same_type_pohuai(bis, start_bi_idx, target_xl, xd_type):
    """Check the next BI of the correct type for pohuai."""
    start_bi = bis[start_bi_idx]
    last_line_idx = target_xl.lines[-1].index
    
    for bi_idx in range(last_line_idx + 1, len(bis)):
        bi = bis[bi_idx]
        if xd_type == "up":
            if bi.type == "down":
                return bi.low < start_bi.low, bi_idx, bi.low, start_bi.low
        else:
            if bi.type == "up":
                return bi.high > start_bi.high, bi_idx, bi.high, start_bi.high
    return False, None, None, None

print("BTC5m up[3] FX@bi[6] (next down BI):")
ph, bi_idx, val, sv = check_next_same_type_pohuai(bis5m, 3, bi6_tz, "up")
print(f"  pohuai={ph} bi[{bi_idx}] val={val} start={sv}")

print("BTC5m up[3] FX@bi[10] (next down BI):")
ph, bi_idx, val, sv = check_next_same_type_pohuai(bis5m, 3, bi10_tz, "up")
print(f"  pohuai={ph} bi[{bi_idx}] val={val} start={sv}")

print("BTC60 down[28] FX@bi[39] (next up BI):")
ph, bi_idx, val, sv = check_next_same_type_pohuai(bis60, 28, bi39_tz, "down")
print(f"  pohuai={ph} bi[{bi_idx}] val={val} start={sv}")

print("BTC60 up[39] FX@bi[42] (next down BI):")
ph, bi_idx, val, sv = check_next_same_type_pohuai(bis60, 39, bi42_tz, "up")
print(f"  pohuai={ph} bi[{bi_idx}] val={val} start={sv}")
