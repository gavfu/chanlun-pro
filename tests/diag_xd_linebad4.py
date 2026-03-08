"""Verify: merged TZXL elements get line_bad reset to False.

The flow is:
1. bi[15] unmerged: line_bad = True (line 4133, because NEW⊃OLD with bi[13])
2. bi[15] + bi[17] merge (OLD⊃NEW): line_bad = False (line 4125)

So the rule is:
- When containment is NEW⊃OLD: DON'T merge, SET line_bad=True on new element
- When containment is OLD⊃NEW: DO merge, SET line_bad=False on merged element

Let me also check bi[9] - it gets line_bad=True and is NOT merged (no bi[9,11] exists)
because bi[9] vs bi[11]: bi[9] h=68283.7,l=65595.7; bi[11] h=68687,l=66915. No containment.

Let me also verify bi[12] - it gets line_bad=True in the UP sequence for XD[1]
"""
import sys; sys.path.insert(0, 'src')
import pandas as pd
from chanlun.cl_pyarmor import CL as CL_P
from chanlun.cl_interface import TZXL

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')

# Just check: after all processing, what are the line_bad states?
cd_p = CL_P('BTC/USDT', '60m')
cd_p.process_klines(df)
xds = cd_p.xds

print("=== XD[0] TZXL ===")
for i, t in enumerate(xds[0].tzxls):
    bis_str = ','.join(str(l.index) for l in t.lines)
    merged = len(t.lines) > 1
    print(f"  [{i}] bi[{bis_str}] bad={t.line_bad} merged={merged} max={t.max:.1f} min={t.min:.1f}")

print("\n=== XD[1] TZXL ===")
for i, t in enumerate(xds[1].tzxls):
    bis_str = ','.join(str(l.index) for l in t.lines)
    merged = len(t.lines) > 1
    print(f"  [{i}] bi[{bis_str}] bad={t.line_bad} merged={merged} max={t.max:.1f} min={t.min:.1f}")

if len(xds) > 2:
    print("\n=== XD[2] TZXL ===")
    for i, t in enumerate(xds[2].tzxls):
        bis_str = ','.join(str(l.index) for l in t.lines)
        merged = len(t.lines) > 1
        print(f"  [{i}] bi[{bis_str}] bad={t.line_bad} merged={merged} max={t.max:.1f} min={t.min:.1f}")

# Now let's verify the rule: 
# Check which containment pairs exist and whether they're merged or not
print("\n=== Containment Analysis for XD[0] (DOWN, uses UP strokes) ===")
tzxls = xds[0].tzxls
for i in range(1, len(tzxls)):
    curr = tzxls[i]
    prev = tzxls[i-1]
    # Containment check on max/min (TZXL processed values)
    o_c_n = prev.max >= curr.max and prev.min <= curr.min
    n_c_o = curr.max >= prev.max and curr.min <= prev.min
    if o_c_n or n_c_o:
        direction = "OLD⊃NEW" if o_c_n else "NEW⊃OLD"
        merged_str = "MERGED" if len(curr.lines) > 1 else "SEPARATE"
        print(f"  [{i-1}]→[{i}]: {direction} → {merged_str}, bad={curr.line_bad}")
    else:
        print(f"  [{i-1}]→[{i}]: no containment")
