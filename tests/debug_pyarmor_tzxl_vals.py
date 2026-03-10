"""
Deeper investigation into pyarmor's TZXL merge criterion.

Key observation: pyarmor's TZXL[4] has lines=[37,39] with max=67299.4, min=65595.7
This means bi[39] was MERGED INTO bi[37]'s TZXL.

In our code: bi[39] (max=68283.7, min=65595.7) NEW⊃OLD bi[37] (max=67299.4, min=65826.1)
This is because: 68283.7 >= 67299.4 AND 65595.7 <= 65826.1

But pyarmor MERGED them. Let me look at the pyarmor TZXL list more carefully.
Notice pyarmor also has max=68115.9 for bi[41], while our code gets max=68687.0.

Wait - pyarmor's TZXL[5] is max=68115.9 but ours is max=68687.0 for bi[41].
bi[41]: high=68687.0, low=66915.0
For bh_direction="down": max = min(highs) = 68687.0, min = min(lows) = 66915.0

So where does pyarmor get max=68115.9 for TZXL[5]?

Hypothesis: pyarmor uses the PRE_LINE high/low for TZXL max/min values!
bi[41].pre_line = bi[40]: high=68283.7, low=66915.0

Wait, that gives 68283.7, not 68115.9...

Let me check if pyarmor's TZXL uses a different value source entirely.
Maybe it uses overlapping price ranges or candle values.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_interface import TZXL
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

print("=== Examining pyarmor's mysterious max=68115.9 for bi[41] ===")
print(f"  bi[41]: high={bis[41].high}, low={bis[41].low}")
print(f"  bi[40]: high={bis[40].high}, low={bis[40].low}")
print()

# Check if 68115.9 appears as any BI's value
for bi in bis:
    if abs(bi.high - 68115.9) < 0.1 or abs(bi.low - 68115.9) < 0.1:
        print(f"  Found 68115.9 in bi[{bi.index}]: high={bi.high}, low={bi.low}")

# Check the klines 
klines = cl.get_cl_klines()
for k in klines:
    if abs(k.h - 68115.9) < 0.1 or abs(k.l - 68115.9) < 0.1:
        print(f"  Found 68115.9 in kline: h={k.h}, l={k.l}, o={k.o}, c={k.c}")

# Check start/end points of BIs
for bi in bis[39:44]:
    print(f"  bi[{bi.index}]: type={bi.type}, high={bi.high}, low={bi.low}")
    print(f"    start: idx={bi.start.index if bi.start else 'N/A'}, val={bi.start.val if bi.start else 'N/A'}")
    print(f"    end: idx={bi.end.index if bi.end else 'N/A'}, val={bi.end.val if bi.end else 'N/A'}")

print()
print("=== Maybe pyarmor's TZXL uses BI start/end values instead of high/low? ===")
# For a DOWN BI: start is top (high), end is bottom (low)
# For an UP BI: start is bottom (low), end is top (high)
for bi in [bis[i] for i in [29, 31, 33, 35, 37, 39, 41] if i < len(bis)]:
    start_val = bi.start.val if bi.start else None
    end_val = bi.end.val if bi.end else None
    print(f"  bi[{bi.index}] ({bi.type}): high={bi.high}, low={bi.low}, start.val={start_val}, end.val={end_val}")
    
    # For UP BI: start is DI FX (bottom), end is DING FX (top)
    # TZXL max/min for bh_direction="down":
    #   Using BI high/low: max=high, min=low
    #   Using BI start/end: ???
    if bi.type == 'up':
        print(f"    TZXL(down) using high/low: max={bi.high}, min={bi.low}")
        if start_val and end_val:
            print(f"    TZXL(down) using end/start: max={end_val}, min={start_val}")

print()
print("=== Let's also check pyarmor's TZXLs more carefully ===")
print("Pyarmor trace shows these TZXLs when lines[28..41]:")
print("  [0]: max=70514.6 min=69212.2 lines=[29]")
print("  [1]: max=70110.9 min=68029.6 lines=[31]") 
print("  [2]: max=69033.0 min=67785.4 lines=[33]")
print("  [3]: max=68438.0 min=66588.0 lines=[35]")
print("  [4]: max=67299.4 min=65595.7 lines=[37,39]")
print("  [5]: max=68115.9 min=66915.0 lines=[41]")
print()
print("Our TZXLs:")
for bi_idx in [29, 31, 33, 35, 37, 39, 41]:
    bi = bis[bi_idx]
    print(f"  bi[{bi_idx}]: TZXL(down) max={bi.high}, min={bi.low}")

print()
print("=== Check if pyarmor uses FX val instead of BI high/low ===")
# bi[41] high=68687.0 but pyarmor max=68115.9
# If pyarmor uses end.val (DING FX val for UP BI)
bi41 = bis[41]
print(f"  bi[41] end.val = {bi41.end.val if bi41.end else 'N/A'}")
print(f"  bi[41] start.val = {bi41.start.val if bi41.start else 'N/A'}")

# Check ALL the UP BIs to see if pyarmor values match FX vals
print()
print("=== Systematic check: BI high/low vs FX end.val/start.val ===")
for bi_idx in [29, 31, 33, 35, 37, 39, 41]:
    bi = bis[bi_idx]
    if bi.end and bi.start:
        # UP BI: end = DING (top), start = DI (bottom)
        top_val = bi.end.val  # ding FX val
        bot_val = bi.start.val  # di FX val
        print(f"  bi[{bi_idx}]: high={bi.high}, end.val={top_val}, low={bi.low}, start.val={bot_val}")
        print(f"    pyarmor_max matches high? {abs(bi.high - top_val) < 0.01}")
        print(f"    pyarmor_max matches end.val? {'(same)' if abs(bi.high - top_val) < 0.01 else '(DIFFERENT!)'}")
