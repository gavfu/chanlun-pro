"""Try merge using raw BI high/low instead of direction-processed max/min"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_interface import TZXL, BI

CL_CONFIG = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
}

df = pd.read_parquet("tests/test_data/ETH_USDT_60m_1000.parquet")
cd = CL_O("test", "test", config=CL_CONFIG)
cd.process_klines(df)
bis = cd.get_bis()

# ETH60 down from bi[28]
# UP BIs: bi[29], bi[31], bi[33], ...
start_bi = 28
tzxl_bi_type = "up"
bh_direction = "down"

tzxl_bis = [bi for bi in bis[start_bi:] if bi.type == tzxl_bi_type]

print(f"=== ETH60 down from bi[{start_bi}] ===")
print("Merge comparison: using TZXL max/min vs raw BI high/low")

# Method 1: Current (TZXL max/min)
tzxls_current = []
for bi in tzxl_bis:
    pre_line = bis[bi.index - 1] if bi.index > 0 else bi
    new_t = TZXL(bh_direction=bh_direction, line=bi, pre_line=pre_line,
                line_bad=False, done=bi.is_done())
    if not tzxls_current:
        tzxls_current.append(new_t)
        continue
    last = tzxls_current[-1]
    old_new = last.max >= new_t.max and last.min <= new_t.min
    new_old = new_t.max >= last.max and new_t.min <= last.min
    if old_new:
        last.lines.append(bi)
        last.done = bi.is_done()
        last.line_bad = False
        last.update_maxmin()
    elif new_old:
        new_t.line_bad = True
        tzxls_current.append(new_t)
    else:
        tzxls_current.append(new_t)

print(f"\nMethod 1 (current TZXL max/min) - {len(tzxls_current)} TZXLs:")
for i, xl in enumerate(tzxls_current):
    print(f"  [{i}]: max={xl.max:.1f} min={xl.min:.1f} bad={xl.line_bad} lines=[{','.join(str(l.index) for l in xl.lines)}]")

# Method 2: Using raw BI high/low for containment check
tzxls_raw = []
for bi in tzxl_bis:
    pre_line = bis[bi.index - 1] if bi.index > 0 else bi
    new_t = TZXL(bh_direction=bh_direction, line=bi, pre_line=pre_line,
                line_bad=False, done=bi.is_done())
    if not tzxls_raw:
        tzxls_raw.append(new_t)
        continue
    last = tzxls_raw[-1]
    # Use raw BI high/low for containment
    last_raw_high = max(l.high for l in last.lines)
    last_raw_low = min(l.low for l in last.lines)
    new_raw_high = bi.high
    new_raw_low = bi.low
    old_new = last_raw_high >= new_raw_high and last_raw_low <= new_raw_low
    new_old = new_raw_high >= last_raw_high and new_raw_low <= last_raw_low
    if old_new:
        last.lines.append(bi)
        last.done = bi.is_done()
        last.line_bad = False
        last.update_maxmin()
    elif new_old:
        new_t.line_bad = True
        tzxls_raw.append(new_t)
    else:
        tzxls_raw.append(new_t)

print(f"\nMethod 2 (raw BI high/low) - {len(tzxls_raw)} TZXLs:")
for i, xl in enumerate(tzxls_raw):
    print(f"  [{i}]: max={xl.max:.1f} min={xl.min:.1f} bad={xl.line_bad} lines=[{','.join(str(l.index) for l in xl.lines)}]")

# Check DI FXs for both
print("\nDI FX comparison:")
for method, tzxls in [("current", tzxls_current), ("raw", tzxls_raw)]:
    fxs = []
    for i in range(1, len(tzxls) - 1):
        xl = tzxls[i]
        if xl.min < tzxls[i-1].min and xl.min < tzxls[i+1].min:
            fxs.append((i, xl.min, xl.line_bad))
    print(f"  {method}: {fxs}")
