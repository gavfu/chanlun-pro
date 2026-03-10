"""
Track down the mysterious max=68115.9 value in pyarmor's TZXL for bi[41].

68115.9 exists in a kline. Let's find WHICH kline and its position relative to bi[41].
Also check if pyarmor's TZXL max/min might come from the PREVIOUS TZXL merged values
or from a different BI representation.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl import CL

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
klines = cl.get_cl_klines()

# Find ALL klines that have 68115.9
print("=== Finding kline with value 68115.9 ===")
for k in klines:
    if abs(k.h - 68115.9) < 0.2 or abs(k.l - 68115.9) < 0.2:
        print(f"  kline idx={k.index}: h={k.h}, l={k.l}, o={k.o}, c={k.c}")

# Show all klines within bi[40] and bi[41]
print()
print("=== Klines in bi[40] (down) ===")
bi40 = bis[40]
bi41 = bis[41]
print(f"  bi[40]: start.index={bi40.start.index}, end.index={bi40.end.index}, type={bi40.type}")
for k in klines:
    if bi40.start.index <= k.index <= bi40.end.index:
        print(f"    kline[{k.index}]: h={k.h}, l={k.l}")

print()
print("=== Klines in bi[41] (up) ===")
print(f"  bi[41]: start.index={bi41.start.index}, end.index={bi41.end.index}, type={bi41.type}")
for k in klines:
    if bi41.start.index <= k.index <= bi41.end.index:
        print(f"    kline[{k.index}]: h={k.h}, l={k.l}")

# Maybe pyarmor's _xd_line_to_tzxl works differently
# Let's check what _xd_line_to_tzxl does with bi[41]
print()
print("=== Testing pyarmor's _xd_line_to_tzxl with bi[41] ===")
# Monkey-patch to trace what _xd_line_to_tzxl returns
original_line_to_tzxl = cl._xd_line_to_tzxl
def traced_line_to_tzxl(base_lines, bh_direction, _l):
    result = original_line_to_tzxl(base_lines, bh_direction, _l)
    if _l.index in [37, 39, 41]:
        print(f"  _xd_line_to_tzxl(bh_direction={bh_direction}, line[{_l.index}]):")
        print(f"    line: type={_l.type}, high={_l.high}, low={_l.low}")
        print(f"    result: max={result.max}, min={result.min}, lines={[l.index for l in result.lines]}")
        print(f"    result.line: type={result.line.type}, high={result.line.high}, low={result.line.low}")
        if hasattr(result, 'pre_line') and result.pre_line:
            print(f"    result.pre_line: type={result.pre_line.type}, high={result.pre_line.high}, low={result.pre_line.low}")
    return result

cl._xd_line_to_tzxl = traced_line_to_tzxl

# Need to re-run to see the trace
# Actually, let's just call _xd_line_to_tzxl directly
print()
print("=== Direct call to _xd_line_to_tzxl ===")
tzxl37 = original_line_to_tzxl(bis, "down", bis[37])
print(f"  TZXL(bi[37]): max={tzxl37.max}, min={tzxl37.min}, lines={[l.index for l in tzxl37.lines]}")
if hasattr(tzxl37, 'pre_line'):
    print(f"    pre_line: idx={tzxl37.pre_line.index if hasattr(tzxl37.pre_line, 'index') else 'N/A'}")

tzxl39 = original_line_to_tzxl(bis, "down", bis[39])
print(f"  TZXL(bi[39]): max={tzxl39.max}, min={tzxl39.min}, lines={[l.index for l in tzxl39.lines]}")

tzxl41 = original_line_to_tzxl(bis, "down", bis[41])
print(f"  TZXL(bi[41]): max={tzxl41.max}, min={tzxl41.min}, lines={[l.index for l in tzxl41.lines]}")

# Now let's also try calling _xd_cal_line_xlfx directly with a smaller set
print()
print("=== Calling _xd_cal_line_xlfx on smaller subsets ===")

# With just bi[37, 38, 39]  
subset = bis[37:40]
print(f"  Calling with lines indexes: {[b.index for b in subset]}")
tzxls, xlfxs = cl._xd_cal_line_xlfx(subset, 'di', 'bh')
print(f"  TZXLs ({len(tzxls)}):")
for i, t in enumerate(tzxls):
    print(f"    [{i}]: max={t.max} min={t.min} bad={t.line_bad} lines={[l.index for l in t.lines]}")

# With bi[35, 36, 37, 38, 39, 40, 41]
subset = bis[35:42]
print(f"\n  Calling with lines indexes: {[b.index for b in subset]}")
tzxls, xlfxs = cl._xd_cal_line_xlfx(subset, 'di', 'bh')
print(f"  TZXLs ({len(tzxls)}):")
for i, t in enumerate(tzxls):
    print(f"    [{i}]: max={t.max} min={t.min} bad={t.line_bad} lines={[l.index for l in t.lines]}")
print(f"  XLFXs ({len(xlfxs)}):")
for i, f in enumerate(xlfxs):
    print(f"    [{i}]: type={f.type} bad={f.is_line_bad} done={f.done} xl_lines={[l.index for l in f.xl.lines]}")

# With bi[29..41]
subset = bis[29:42]
print(f"\n  Calling with lines indexes: {[b.index for b in subset]}")
tzxls, xlfxs = cl._xd_cal_line_xlfx(subset, 'di', 'bh')
print(f"  TZXLs ({len(tzxls)}):")
for i, t in enumerate(tzxls):
    print(f"    [{i}]: max={t.max} min={t.min} bad={t.line_bad} lines={[l.index for l in t.lines]}")
print(f"  XLFXs ({len(xlfxs)}):")
for i, f in enumerate(xlfxs):
    print(f"    [{i}]: type={f.type} bad={f.is_line_bad} done={f.done} xl_lines={[l.index for l in f.xl.lines]}")
