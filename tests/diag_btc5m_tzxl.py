"""Detailed trace of _find_xd_end for BTC5m UP from bi[3]"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P
from chanlun.cl_interface import *

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

# Open
cl = CL_O("test", "test", config=CL_CONFIG)
cl.process_klines(df)
bis = cl.get_bis()

# Pyarmor
cl_p = CL_P("test", "test", config=CL_CONFIG)
cl_p.process_klines(df)
bis_p = cl_p.get_bis()
xds_p = cl_p.get_xds()

print(f"Pyarmor first XD: {xds_p[0].type} bi[{xds_p[0].start.index}→{xds_p[0].end.index}]")
print(f"Pyarmor xds[0] start h={xds_p[0].start.val:.1f} end h={xds_p[0].end.val:.1f}")
print(f"Pyarmor xds[0] ding_fx: {xds_p[0].ding_fx}")
if xds_p[0].ding_fx and xds_p[0].ding_fx.xl:
    print(f"  ding_fx.xl lines: {[(l.index, l.high, l.low) for l in xds_p[0].ding_fx.xl.lines]}")
    print(f"  ding_fx.xl.line_bad: {xds_p[0].ding_fx.xl.line_bad}")
    print(f"  ding_fx.is_line_bad: {xds_p[0].ding_fx.is_line_bad}")

print(f"\nBTC5m BIs around start:")
for i in range(min(20, len(bis))):
    b = bis[i]
    print(f"  bi[{i:2d}] {b.type:4s} h={b.high:10.1f} l={b.low:10.1f}")

# Trace TZXL for UP from bi[3]
print(f"\n--- TZXL for UP from bi[3] (DOWN BIs, bh=up) ---")
start_bi_idx = 3
xd_type = "up"
tzxl_bi_type = "down"  # opposite
bh_direction = "up"

tzxl_bis = [bi for bi in bis if bi.index >= start_bi_idx and bi.type == tzxl_bi_type]
print(f"DOWN BIs from bi[3]: {[(b.index, b.high, b.low) for b in tzxl_bis[:15]]}")

# Build TZXL manually
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
        print(f"  MERGE(OLD⊃NEW) bi[{bi.index}] into TZXL[{len(tzxls)-1}] → max={last.max:.1f} min={last.min:.1f}")
    elif new_old:
        new_tzxl.line_bad = True
        tzxls.append(new_tzxl)
        print(f"  TZXL[{len(tzxls)-1}]: bi[{bi.index}] max={new_tzxl.max:.1f} min={new_tzxl.min:.1f} BAD=True (NEW⊃OLD)")
    else:
        tzxls.append(new_tzxl)
        print(f"  TZXL[{len(tzxls)-1}]: bi[{bi.index}] max={new_tzxl.max:.1f} min={new_tzxl.min:.1f}")

print(f"\nTotal TZXL elements: {len(tzxls)}")

# Check FX
print(f"\n--- FX detection ---")
for i in range(1, len(tzxls) - 1):
    curr = tzxls[i]
    prev = tzxls[i-1]
    nxt = tzxls[i+1]
    is_ding = curr.max > prev.max and curr.max > nxt.max
    is_di = curr.min < prev.min and curr.min < nxt.min
    fx = "DING" if is_ding else ("DI" if is_di else "---")
    bad_str = "BAD" if curr.line_bad else "   "
    print(f"  TZXL[{i}]: max={curr.max:.1f} min={curr.min:.1f} {bad_str} → {fx}  lines={[l.index for l in curr.lines]}")

# Now also check what pyarmor's TZXL looks like
print(f"\n--- Pyarmor TZXL for xd[0] ---")
if hasattr(xds_p[0], 'tzxl') and xds_p[0].tzxl:
    for i, t in enumerate(xds_p[0].tzxl):
        print(f"  TZXL[{i}]: max={t.max:.1f} min={t.min:.1f} bad={t.line_bad} lines={[l.index for l in t.lines]}")
else:
    print("  No TZXL attribute found")

# Also check what _find_xd_end from cl_open returns differently
print(f"\n--- cl_open _find_xd_end(bi[3], 'up') ---")
result = cl._find_xd_end(bis, 3, "up")
if result:
    end_bi, ding_fx, di_fx, ret_tzxls = result
    print(f"  end_bi={end_bi}")
    print(f"  ding_fx.xl.lines={[l.index for l in ding_fx.xl.lines]}")
    print(f"  ding_fx.xl.line_bad={ding_fx.xl.line_bad}")
    print(f"  ding_fx.is_line_bad={ding_fx.is_line_bad}")
