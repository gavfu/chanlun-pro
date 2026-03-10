"""Trace TZXL for BTCd UP from bi[11] to understand bad FX handling"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_interface import *

CL_CONFIG = {
    "bi_type": "bi_type_old", "fx_qj": "fx_qj_k", "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes", "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1", "xd_bzh": "xd_bzh_no",
}

df = pd.read_parquet("tests/test_data/BTC_USDT_d_500.parquet")
cl = CL_O("test", "test", config=CL_CONFIG)
cl.process_klines(df)
bis = cl.get_bis()

print(f"BIs from 11 onwards:")
for i in range(11, min(32, len(bis))):
    b = bis[i]
    print(f"  bi[{i:2d}] {b.type:4s} h={b.high:10.1f} l={b.low:10.1f}")

# Build TZXL for UP from bi[11]
print(f"\n--- TZXL for UP from bi[11] (DOWN BIs, bh=up) ---")
start = 11
tzxl_bis = [bi for bi in bis if bi.index >= start and bi.type == "down"]
print(f"DOWN BIs: {[(b.index, round(b.high,1), round(b.low,1)) for b in tzxl_bis]}")

tzxls = []
for bi in tzxl_bis:
    pre_line = bis[bi.index - 1] if bi.index > 0 else bi
    new_tzxl = TZXL(bh_direction="up", line=bi, pre_line=pre_line, line_bad=False, done=bi.is_done())
    if len(tzxls) == 0:
        tzxls.append(new_tzxl)
        print(f"  TZXL[0]: bi[{bi.index}] max={new_tzxl.max:.1f} min={new_tzxl.min:.1f}")
        continue

    last = tzxls[-1]
    old_new = last.max >= new_tzxl.max and last.min <= new_tzxl.min
    new_old = new_tzxl.max >= last.max and new_tzxl.min <= last.min
    
    if old_new:
        last.lines.append(bi)
        last.done = bi.is_done()
        last.line_bad = False
        last.update_maxmin()
        print(f"  MERGE bi[{bi.index}] into TZXL[{len(tzxls)-1}] bad={last.line_bad}")
    elif new_old:
        new_tzxl.line_bad = True
        tzxls.append(new_tzxl)
        print(f"  TZXL[{len(tzxls)-1}]: bi[{bi.index}] max={new_tzxl.max:.1f} min={new_tzxl.min:.1f} BAD")
    else:
        tzxls.append(new_tzxl)
        print(f"  TZXL[{len(tzxls)-1}]: bi[{bi.index}] max={new_tzxl.max:.1f} min={new_tzxl.min:.1f}")

print(f"\nTotal: {len(tzxls)} elements")

# FX detection
print(f"\n--- FX detection (DING) ---")
for i in range(1, len(tzxls) - 1):
    curr = tzxls[i]
    prev = tzxls[i-1]
    nxt = tzxls[i+1]
    is_ding = curr.max > prev.max and curr.max > nxt.max
    if is_ding:
        bad_str = "BAD" if curr.line_bad else "   "
        prev_bad = "prev_BAD" if prev.line_bad else "prev_OK "
        print(f"  DING at TZXL[{i}]: max={curr.max:.1f} {bad_str} {prev_bad} lines={[l.index for l in curr.lines]}")
