"""Verify the end_bi_idx calculation for XD[0] DI fractal.

DI fractal at tzxls [6,7,8] where:
  tzxl[6] = bi[13] min=64232.8
  tzxl[7] = bi[15,17] min=62401.7  (the low point)
  tzxl[8] = bi[19] min=62979.5

For a DOWN XD, the DI fractal means the segment ends.
The end bi should be: the lowest bi in tzxl[7]'s lines.
tzxl[7].lines = [bi[15], bi[17]]
bi[15] is UP, bi[17] is UP
For a DOWN XD, the end should be at the lowest point.
Actually - the end_line of a DOWN XD should be the lowest low.
Since these are UP strokes, their lows are the "bottoms" they start from.

Let me check what pyarmor uses as end_bi_idx.
"""
import sys; sys.path.insert(0, 'src')
import pandas as pd
from chanlun.cl_pyarmor import CL as CL_P

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')
cd_p = CL_P('BTC/USDT', '60m')
cd_p.process_klines(df)
xds = cd_p.xds
bis = cd_p.bis

print("=== XD[0] end details ===")
xd0 = xds[0]
print(f"  end_line = bi[{xd0.end_line.index}] type={xd0.end_line.type}")
print(f"  end_line h={xd0.end_line.high:.1f} l={xd0.end_line.low:.1f}")
print(f"  start_line = bi[{xd0.start_line.index}] type={xd0.start_line.type}")

# TZXL[7] = bi[15,17] - these are UP strokes
t7 = xd0.tzxls[7]
for l in t7.lines:
    print(f"  tzxl[7] line bi[{l.index}] type={l.type} h={l.high:.1f} l={l.low:.1f}")
# bi[15] h=69999.0 l=62401.7 - this has the lowest low
# bi[17] h=68188.8 l=66462.0

# For DOWN XD: end at the lowest, which is bi[15] (l=62401.7)
# But bi[15] is an UP stroke. The end_line should be the DOWN stroke before bi[15]
# which is bi[14] (type=down)
print(f"\n  bi[14] type={bis[14].type} h={bis[14].high:.1f} l={bis[14].low:.1f}")

# So: XD[0] DOWN from bi[2] (h=70110.9, type=down) to bi[14] (type=down)
# end_bi_idx = 14

print(f"\n=== XD[0] di_fx details ===")
di = xd0.di_fx
print(f"  type={di.type}")
print(f"  xl = bi[{','.join(str(l.index) for l in di.xl.lines)}]")
print(f"  is_line_bad={di.is_line_bad}")
print(f"  done={di.done}")

# Check what the XLFX.xls looks like
for j, xl in enumerate(di.xls):
    if xl is not None:
        lines_str = ','.join(str(l.index) for l in xl.lines)
        print(f"  xls[{j}] bi[{lines_str}] max={xl.max:.1f} min={xl.min:.1f} bad={xl.line_bad}")
    else:
        print(f"  xls[{j}] None")

print(f"\n=== XD[1] start/end ===")
xd1 = xds[1]
print(f"  type={xd1.type}")
print(f"  start_line = bi[{xd1.start_line.index}] type={xd1.start_line.type}")
print(f"  end_line = bi[{xd1.end_line.index}] type={xd1.end_line.type}")

# XD[1] UP: start_line=bi[15], end_line=bi[23]
# start_line is the UP stroke at the bottom (bi[15] is the lowest UP stroke)
# end_line is the UP stroke at the top

print(f"\n=== XD[2] start/end ===")
if len(xds) > 2:
    xd2 = xds[2]
    print(f"  type={xd2.type}")
    print(f"  start_line = bi[{xd2.start_line.index}] type={xd2.start_line.type}")
    print(f"  end_line = bi[{xd2.end_line.index}] type={xd2.end_line.type}")
