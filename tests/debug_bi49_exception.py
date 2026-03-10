"""
Almost perfect pattern: is_line_bad = (fx_type == 'ding'), except BTC60 di@bi[49].

But wait - maybe the issue isn't about fx_type directly but about the DIRECTION
of the "bad" containment relative to what the FX is looking for.

For ding FX (searching for local max among down BIs):
  TZXL max = min(highs), min = min(lows) [bh_direction="down"]
  Bad = NEW⊃OLD at BI level; new down BI has higher high AND lower low
  The bad TZXL has a new EXTREME (max = new_bi.high which is > old's)
  → bad means the FX's extreme comes from the NEW containent direction

For di FX (searching for local min among up BIs):
  TZXL max = max(highs), min = max(lows) [bh_direction="up"]  
  Bad = NEW⊃OLD at BI level: new up BI has higher high AND lower low
  The bad TZXL has a new EXTREME (min = new_bi.low which is < old's)
  → bad means the FX's extreme comes from the NEW containment direction

Hmm, actually let me reconsider from scratch.

For bh_direction="down" (used in ding FX):
  TZXL takes the min of highs, min of lows
  Containment: if cur > prev in terms of TZXL (max >= max AND min <= min)
    → OLD⊃NEW: merge (old's range includes new)
    → NEW⊃OLD: separate with bad=True (new's range includes old)

Wait - I need to re-read the containment logic in cl_open.py!
"""
import sys
sys.path.insert(0, "src")

# First, let me read the actual containment code in cl_open.py
# to understand EXACTLY when line_bad=True is set

# For now, let me check a different hypothesis:
# Maybe is_line_bad depends on the TZXL index within the sequence,
# specifically whether the bad TZXL is the SECOND element (position 1)

# Actually, the simplest thing: since the FX has 3 TZXLs (xls[0], xls[1]=xl, xls[2]),
# the bad TZXL is always xls[1] (the middle).
# Maybe is_line_bad depends on whether xls[0] (left neighbor) is ALSO bad?

import pandas as pd
from chanlun.cl_pyarmor import CL as CL_Pyarmor

config = {
    "bi_type": "bi_type_old", "fx_qj": "fx_qj_k", "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes", "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1", "xd_bzh": "xd_bzh_no",
    "xd_zs_max_lines_split": 11, "xd_allow_split_no_highlow": 1,
    "xd_allow_split_zs_kz": 0, "xd_allow_split_zs_more_line": 1,
    "xd_allow_split_zs_no_direction": 1,
}

# Check if BTC60 di FX@bi[49] is actually created with different start positions
# Maybe it only appears as is_line_bad=TRUE from certain start positions?
df60 = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cl60 = CL_Pyarmor("test", "60m", config)
cl60.process_klines(df60)
bis60 = cl60.get_bis()

print("=== BTC60 di FX@bi[49] from different starts ===")
for start in range(30, 50):
    lines = bis60[start:]
    if len(lines) < 3:
        continue
    result = cl60._xd_cal_line_xlfx(lines, fx_type='di', bh_type='no_bh')
    tzxls, xlfxs = result
    for fx in xlfxs:
        xl_bis = [l.index for l in fx.xl.lines]
        if 49 in xl_bis and fx.xl.line_bad:
            xls_info = [(
                [l.index for l in x.lines], 
                x.line_bad, 
                f"max={x.max:.1f}", 
                f"min={x.min:.1f}"
            ) for x in fx.xls]
            print(f"  start={start}: is_line_bad={fx.is_line_bad} xls={xls_info}")

# Compare with bi[39] which is always FALSE
print("\n=== BTC60 di FX@bi[39] from different starts ===")
for start in range(20, 40):
    lines = bis60[start:]
    if len(lines) < 3:
        continue  
    result = cl60._xd_cal_line_xlfx(lines, fx_type='di', bh_type='no_bh')
    tzxls, xlfxs = result
    for fx in xlfxs:
        xl_bis = [l.index for l in fx.xl.lines]
        if 39 in xl_bis and fx.xl.line_bad:
            xls_info = [(
                [l.index for l in x.lines],
                x.line_bad,
                f"max={x.max:.1f}",
                f"min={x.min:.1f}"
            ) for x in fx.xls]
            print(f"  start={start}: is_line_bad={fx.is_line_bad} xls={xls_info}")

# KEY QUESTION: does the predecessor TZXL being MERGED vs SINGLE matter?
# bi[49] predecessor: always [45,47] (merged) or sometimes [47] (single)?
print("\n\n=== BTC60 TZXLs around bi[49] from start=28 ===")
lines = bis60[28:]
result = cl60._xd_cal_line_xlfx(lines, fx_type='di', bh_type='no_bh')
tzxls, xlfxs = result
for i, tz in enumerate(tzxls):
    bi_indices = [l.index for l in tz.lines]
    if max(bi_indices) >= 43 and min(bi_indices) <= 53:
        print(f"  [{i}] bis={bi_indices} bad={tz.line_bad} max={tz.max:.1f} min={tz.min:.1f}")

# What if we start from bi[46] instead? Will bi[45,47] still be merged?
print("\n=== BTC60 TZXLs around bi[49] from start=44 ===") 
lines = bis60[44:]
result = cl60._xd_cal_line_xlfx(lines, fx_type='di', bh_type='no_bh')
tzxls, xlfxs = result
for i, tz in enumerate(tzxls):
    bi_indices = [l.index for l in tz.lines]
    if max(bi_indices) >= 43 and min(bi_indices) <= 53:
        print(f"  [{i}] bis={bi_indices} bad={tz.line_bad} max={tz.max:.1f} min={tz.min:.1f}")
for fx in xlfxs:
    xl_bis = [l.index for l in fx.xl.lines]
    if 49 in xl_bis:
        xls_info = [([l.index for l in x.lines], x.line_bad) for x in fx.xls]
        print(f"  FX@bi[{xl_bis}] is_line_bad={fx.is_line_bad} xl_bad={fx.xl.line_bad} xls={xls_info}")

# From start=46: bi[45] is not in the lines, so containment changes
print("\n=== BTC60 TZXLs around bi[49] from start=46 ===")
lines = bis60[46:]
result = cl60._xd_cal_line_xlfx(lines, fx_type='di', bh_type='no_bh')
tzxls, xlfxs = result
for i, tz in enumerate(tzxls):
    bi_indices = [l.index for l in tz.lines]
    if max(bi_indices) >= 45 and min(bi_indices) <= 53:
        print(f"  [{i}] bis={bi_indices} bad={tz.line_bad} max={tz.max:.1f} min={tz.min:.1f}")
for fx in xlfxs:
    xl_bis = [l.index for l in fx.xl.lines]
    if 49 in xl_bis:
        xls_info = [([l.index for l in x.lines], x.line_bad) for x in fx.xls]
        print(f"  FX@bi[{xl_bis}] is_line_bad={fx.is_line_bad} xl_bad={fx.xl.line_bad} xls={xls_info}")
