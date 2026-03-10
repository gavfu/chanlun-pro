"""
Test hypothesis: pyarmor uses "always first FX" but _build_xd_fx_result returns
None for some FXes (segment too short), causing them to be skipped.

Check: for BTC5m up[3], does _build_xd_fx_result return None for bi[6]?
  bi[6] is a down BI. For up segment, end_bi = max(curr_xl.lines, key=h)
  bi[6]: high=73060.0. end_bi_idx = 6.
  Since bi[6].type == "down" and 6 > 0: end_bi_idx = 5
  Check: 5 - 3 = 2 >= 2 → result is NOT None

Hmm, so it wouldn't be None. But let me verify empirically by tracing
_find_xd_end for all key cases.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor
from chanlun.cl_interface import XLFX, TZXL

config = {
    "bi_type": "bi_type_old", "fx_qj": "fx_qj_k", "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes", "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1", "xd_bzh": "xd_bzh_no",
    "xd_zs_max_lines_split": 11, "xd_allow_split_no_highlow": 1,
    "xd_allow_split_zs_kz": 0, "xd_allow_split_zs_more_line": 1,
    "xd_allow_split_zs_no_direction": 1,
}

# BTC5m
df5m = pd.read_parquet("tests/test_data/BTC_USDT_5m_1000.parquet")
cl_o = CL_Open("test", "5m", config)
cl_o.process_klines(df5m)
bis_o = cl_o.get_bis()

print("=== BTC5m up[3]: trace _find_xd_end behavior ===")
start_bi_idx = 3
xd_type = "up"
tzxl_bi_type = "down"
bh_direction = "up"
target_fx_type = "ding"

# Build TZXLs
tzxl_bis = [bi for bi in bis_o[start_bi_idx:] if bi.type == tzxl_bi_type]
tzxls = []
for bi in tzxl_bis:
    pre_line = bis_o[bi.index - 1] if bi.index > 0 else bi
    done = bi.is_done()
    new_tzxl = TZXL(bh_direction=bh_direction, line=bi, pre_line=pre_line, line_bad=False, done=done)
    if len(tzxls) == 0:
        tzxls.append(new_tzxl)
        continue
    last_tzxl = tzxls[-1]
    old_contains_new = last_tzxl.max >= new_tzxl.max and last_tzxl.min <= new_tzxl.min
    new_contains_old = new_tzxl.max >= last_tzxl.max and new_tzxl.min <= last_tzxl.min
    if old_contains_new:
        last_tzxl.lines.append(bi)
        last_tzxl.done = done
        last_tzxl.line_bad = False
        last_tzxl.update_maxmin()
    elif new_contains_old:
        new_tzxl.line_bad = True
        tzxls.append(new_tzxl)
    else:
        tzxls.append(new_tzxl)

print(f"  {len(tzxls)} TZXLs total")
for i in range(min(8, len(tzxls))):
    tz = tzxls[i]
    bi_indices = [l.index for l in tz.lines]
    print(f"    [{i}] bis={bi_indices} bad={tz.line_bad} max={tz.max:.1f} min={tz.min:.1f}")

# Check FXes
print(f"\n  FX scan:")
for i in range(1, len(tzxls) - 1):
    curr_xl = tzxls[i]
    prev_xl = tzxls[i - 1]
    next_xl = tzxls[i + 1]
    is_fx = curr_xl.max > prev_xl.max and curr_xl.max > next_xl.max
    if is_fx:
        bi_indices = [l.index for l in curr_xl.lines]
        # Check bi-pohuai
        pohuai = cl_o._check_xd_bi_pohuai(bis_o, start_bi_idx, curr_xl, xd_type)
        
        # Check build result
        end_bi = max(curr_xl.lines, key=lambda l: l.high)
        end_bi_idx = end_bi.index
        if bis_o[end_bi_idx].type == "down" and end_bi_idx > 0:
            end_bi_idx -= 1
        result_valid = (end_bi_idx - start_bi_idx) >= 2
        
        print(f"    FX at TZXL[{i}] bis={bi_indices} bad={curr_xl.line_bad} "
              f"max={curr_xl.max:.1f} pohuai={pohuai} "
              f"end_bi_idx={end_bi_idx} valid={result_valid}")
        if i >= 8:
            break

# BTC60
print("\n\n=== BTC60 down[28]: trace _find_xd_end behavior ===")
df60 = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cl_o60 = CL_Open("test", "60m", config)
cl_o60.process_klines(df60)
bis_o60 = cl_o60.get_bis()

start_bi_idx = 28
xd_type = "down"
tzxl_bi_type = "up"
bh_direction = "down"
target_fx_type = "di"

tzxl_bis = [bi for bi in bis_o60[start_bi_idx:] if bi.type == tzxl_bi_type]
tzxls = []
for bi in tzxl_bis:
    pre_line = bis_o60[bi.index - 1] if bi.index > 0 else bi
    done = bi.is_done()
    new_tzxl = TZXL(bh_direction=bh_direction, line=bi, pre_line=pre_line, line_bad=False, done=done)
    if len(tzxls) == 0:
        tzxls.append(new_tzxl)
        continue
    last_tzxl = tzxls[-1]
    old_contains_new = last_tzxl.max >= new_tzxl.max and last_tzxl.min <= new_tzxl.min
    new_contains_old = new_tzxl.max >= last_tzxl.max and new_tzxl.min <= last_tzxl.min
    if old_contains_new:
        last_tzxl.lines.append(bi)
        last_tzxl.done = done
        last_tzxl.line_bad = False
        last_tzxl.update_maxmin()
    elif new_contains_old:
        new_tzxl.line_bad = True
        tzxls.append(new_tzxl)
    else:
        tzxls.append(new_tzxl)

print(f"  {len(tzxls)} TZXLs total")
for i in range(min(12, len(tzxls))):
    tz = tzxls[i]
    bi_indices = [l.index for l in tz.lines]
    print(f"    [{i}] bis={bi_indices} bad={tz.line_bad} max={tz.max:.1f} min={tz.min:.1f}")

print(f"\n  FX scan:")
for i in range(1, len(tzxls) - 1):
    curr_xl = tzxls[i]
    prev_xl = tzxls[i - 1]
    next_xl = tzxls[i + 1]
    is_fx = curr_xl.min < prev_xl.min and curr_xl.min < next_xl.min
    if is_fx:
        bi_indices = [l.index for l in curr_xl.lines]
        pohuai = cl_o60._check_xd_bi_pohuai(bis_o60, start_bi_idx, curr_xl, xd_type)
        
        end_bi = min(curr_xl.lines, key=lambda l: l.low)
        end_bi_idx = end_bi.index
        if bis_o60[end_bi_idx].type == "up" and end_bi_idx > 0:
            end_bi_idx -= 1
        result_valid = (end_bi_idx - start_bi_idx) >= 2
        
        print(f"    FX at TZXL[{i}] bis={bi_indices} bad={curr_xl.line_bad} "
              f"min={curr_xl.min:.1f} pohuai={pohuai} "
              f"end_bi_idx={end_bi_idx} valid={result_valid}")

# BTC60 up[39]
print("\n\n=== BTC60 up[39]: trace _find_xd_end behavior ===")
start_bi_idx = 39
xd_type = "up"
tzxl_bi_type = "down"
bh_direction = "up"
target_fx_type = "ding"

tzxl_bis = [bi for bi in bis_o60[start_bi_idx:] if bi.type == tzxl_bi_type]
tzxls = []
for bi in tzxl_bis:
    pre_line = bis_o60[bi.index - 1] if bi.index > 0 else bi
    done = bi.is_done()
    new_tzxl = TZXL(bh_direction=bh_direction, line=bi, pre_line=pre_line, line_bad=False, done=done)
    if len(tzxls) == 0:
        tzxls.append(new_tzxl)
        continue
    last_tzxl = tzxls[-1]
    old_contains_new = last_tzxl.max >= new_tzxl.max and last_tzxl.min <= new_tzxl.min
    new_contains_old = new_tzxl.max >= last_tzxl.max and new_tzxl.min <= last_tzxl.min
    if old_contains_new:
        last_tzxl.lines.append(bi)
        last_tzxl.done = done
        last_tzxl.line_bad = False
        last_tzxl.update_maxmin()
    elif new_contains_old:
        new_tzxl.line_bad = True
        tzxls.append(new_tzxl)
    else:
        tzxls.append(new_tzxl)

print(f"  {len(tzxls)} TZXLs total")
for i in range(min(12, len(tzxls))):
    tz = tzxls[i]
    bi_indices = [l.index for l in tz.lines]
    print(f"    [{i}] bis={bi_indices} bad={tz.line_bad} max={tz.max:.1f} min={tz.min:.1f}")

print(f"\n  FX scan:")
for i in range(1, len(tzxls) - 1):
    curr_xl = tzxls[i]
    prev_xl = tzxls[i - 1]
    next_xl = tzxls[i + 1]
    is_fx = curr_xl.max > prev_xl.max and curr_xl.max > next_xl.max
    if is_fx:
        bi_indices = [l.index for l in curr_xl.lines]
        pohuai = cl_o60._check_xd_bi_pohuai(bis_o60, start_bi_idx, curr_xl, xd_type)
        
        end_bi = max(curr_xl.lines, key=lambda l: l.high)
        end_bi_idx = end_bi.index
        if bis_o60[end_bi_idx].type == "down" and end_bi_idx > 0:
            end_bi_idx -= 1
        result_valid = (end_bi_idx - start_bi_idx) >= 2
        
        print(f"    FX at TZXL[{i}] bis={bi_indices} bad={curr_xl.line_bad} "
              f"max={curr_xl.max:.1f} pohuai={pohuai} "
              f"end_bi_idx={end_bi_idx} valid={result_valid}")
