"""
Investigate pyarmor's _xd_cal_line_xlfx internal logic by tracing
the TZXL construction step by step using monkey-patching.

We know pyarmor merges bi[37] and bi[39] while our code doesn't.
Let's trace the actual containment check inside pyarmor.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl import CL
from chanlun.cl_interface import TZXL

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
cl.process_klines(df)
bis = cl.get_bis()

# Trace _xd_line_to_tzxl to see how pyarmor creates TZXL from each line
original_line_to_tzxl = cl._xd_line_to_tzxl
call_count = [0]

def traced_line_to_tzxl(base_lines, bh_direction, _l):
    call_count[0] += 1
    result = original_line_to_tzxl(base_lines, bh_direction, _l)
    if _l.index in [37, 39, 41] and bh_direction == 'down':
        print(f"  _xd_line_to_tzxl(bh_direction={bh_direction}, line[{_l.index}]):")
        print(f"    input line: type={_l.type}, high={_l.high}, low={_l.low}")
        print(f"    result: max={result.max}, min={result.min}")
        print(f"    result.line: idx={result.line.index}, type={result.line.type}, high={result.line.high}, low={result.line.low}")
        if hasattr(result, 'pre_line') and result.pre_line:
            pl = result.pre_line
            print(f"    result.pre_line: idx={pl.index}, type={pl.type}, high={pl.high}, low={pl.low}")
        print(f"    result.lines: {[l.index for l in result.lines]}")
        print(f"    result.line_bad: {result.line_bad}")
        print(f"    result.bh_direction: {result.bh_direction}")
    return result

cl._xd_line_to_tzxl = traced_line_to_tzxl

# Call _xd_cal_line_xlfx with the subset
print("=== Tracing _xd_cal_line_xlfx with bi[37:40] ===")
subset = bis[37:40]
tzxls, xlfxs = cl._xd_cal_line_xlfx(subset, 'di', 'bh')
print(f"\nResult TZXLs ({len(tzxls)}):")
for i, t in enumerate(tzxls):
    print(f"  [{i}]: max={t.max} min={t.min} bad={t.line_bad} lines={[l.index for l in t.lines]}")

# Now let's see what _xd_cal_line_xlfx does differently:
# Look at its source code structure by examining import/inspect
print()
print("=== Examining _xd_cal_line_xlfx internals ===")
import inspect
# Try to get source - will fail for pyarmor but let's try
try:
    source = inspect.getsource(cl._xd_cal_line_xlfx)
    print("Got source!")
    print(source[:2000])
except Exception as e:
    print(f"Cannot get source: {e}")
    
# Instead, let's monkey-patch TZXL's __init__ and update_maxmin to trace
print()
print("=== Tracing TZXL operations ===")

original_init = TZXL.__init__
original_update = TZXL.update_maxmin

def traced_init(self, **kwargs):
    original_init(self, **kwargs)
    line = kwargs.get('line')
    if line and hasattr(line, 'index') and line.index in [37, 39, 41]:
        print(f"  TZXL.__init__(line[{line.index}], bh_dir={kwargs.get('bh_direction')}):")
        print(f"    max={self.max}, min={self.min}, bad={self.line_bad}")

def traced_update(self):
    old_max, old_min = self.max, self.min
    original_update(self)
    if any(l.index in [37, 39, 41] for l in self.lines):
        print(f"  TZXL.update_maxmin(lines={[l.index for l in self.lines]}):")
        print(f"    before: max={old_max}, min={old_min}")
        print(f"    after:  max={self.max}, min={self.min}")

TZXL.__init__ = traced_init
TZXL.update_maxmin = traced_update

# Re-run with the trace
print()
print("=== Re-tracing with TZXL monkey-patch ===")
call_count[0] = 0
subset = bis[37:40]
tzxls, xlfxs = cl._xd_cal_line_xlfx(subset, 'di', 'bh')
print(f"\nResult TZXLs ({len(tzxls)}):")
for i, t in enumerate(tzxls):
    print(f"  [{i}]: max={t.max} min={t.min} bad={t.line_bad} lines={[l.index for l in t.lines]} bh_dir={t.bh_direction}")

# Restore
TZXL.__init__ = original_init
TZXL.update_maxmin = original_update

# Also: what if pyarmor's merge criterion uses the PRE_LINE?
# In pyarmor, TZXL has a pre_line. Maybe the containment check uses pre_line high/low?
print()
print("=== Pre-line analysis ===")
# bi[37].pre_line = bi[36] (down)
# bi[39].pre_line = bi[38] (down)
bi36 = bis[36]
bi38 = bis[38]
bi37 = bis[37]
bi39 = bis[39]
print(f"  bi[36] (down): high={bi36.high}, low={bi36.low}")
print(f"  bi[37] (up):   high={bi37.high}, low={bi37.low}")
print(f"  bi[38] (down): high={bi38.high}, low={bi38.low}")
print(f"  bi[39] (up):   high={bi39.high}, low={bi39.low}")

# For bh_direction="down", maybe containment uses:
# TZXL max = pre_line.high, TZXL min = line.low ?
# bi[37]: max = bi[36].high = 68438.0, min = bi[37].low = 65826.1
# bi[39]: max = bi[38].high = 67299.4, min = bi[39].low = 65595.7
# OLD⊃NEW: 68438.0 >= 67299.4 AND 65826.1 <= 65595.7 → True AND False → No
# NEW⊃OLD: 67299.4 >= 68438.0 → No
print()
print("  Using pre_line.high + line.low:")
print(f"    TZXL(37): max={bi36.high}, min={bi37.low}")
print(f"    TZXL(39): max={bi38.high}, min={bi39.low}")
print(f"    OLD⊃NEW: {bi36.high >= bi38.high} AND {bi37.low <= bi39.low} = {bi36.high >= bi38.high and bi37.low <= bi39.low}")

# What about pre_line HIGH for containment, line LOW for containment?
# Check combining: max = max(pre_line.high, line.high), min = min(pre_line.low, line.low)?
print()
print("  Using max(pre.h, line.h) and min(pre.l, line.l):")
m37_max = max(bi36.high, bi37.high)
m37_min = min(bi36.low, bi37.low)
m39_max = max(bi38.high, bi39.high)
m39_min = min(bi38.low, bi39.low)
print(f"    TZXL(37): max={m37_max}, min={m37_min}")
print(f"    TZXL(39): max={m39_max}, min={m39_min}")
print(f"    OLD⊃NEW: {m37_max >= m39_max} AND {m37_min <= m39_min} = {m37_max >= m39_max and m37_min <= m39_min}")
print(f"    NEW⊃OLD: {m39_max >= m37_max} AND {m39_min <= m37_min} = {m39_max >= m37_max and m39_min <= m37_min}")
# This would be: max(68438,67299.4)=68438.0 >= max(67299.4,68283.7)=68283.7? YES
# AND min(65826.1, 65826.1)=65826.1 <= min(65595.7, 65595.7)=65595.7? NO

print()
print("  Using line.high and pre_line.low:")
print(f"    TZXL(37): max={bi37.high}, min={bi36.low}")
print(f"    TZXL(39): max={bi39.high}, min={bi38.low}")
print(f"    OLD⊃NEW: {bi37.high >= bi39.high} AND {bi36.low <= bi38.low} = {bi37.high >= bi39.high and bi36.low <= bi38.low}")
print(f"    NEW⊃OLD: {bi39.high >= bi37.high} AND {bi38.low <= bi36.low} = {bi39.high >= bi37.high and bi38.low <= bi36.low}")

print()
print("  Using pre_line.high and pre_line.low:")
print(f"    TZXL(37): max={bi36.high}, min={bi36.low}")
print(f"    TZXL(39): max={bi38.high}, min={bi38.low}")
print(f"    OLD⊃NEW: {bi36.high >= bi38.high} AND {bi36.low <= bi38.low} = {bi36.high >= bi38.high and bi36.low <= bi38.low}")
print(f"    NEW⊃OLD: {bi38.high >= bi36.high} AND {bi38.low <= bi36.low} = {bi38.high >= bi36.high and bi38.low <= bi36.low}")
