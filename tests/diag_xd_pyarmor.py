"""Extract pyarmor's final TZXL state for the first XD."""
import sys; sys.path.insert(0, 'src')
import pandas as pd
from chanlun.cl_pyarmor import CL as CL_P

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')
cd_p = CL_P('BTC/USDT', '60m')
cd_p.process_klines(df)

# The first XD is stored in cd_p.xds[0]
xd0 = cd_p.xds[0]
print(f"XD[0]: {xd0.type} start_line=bi[{xd0.start_line.index}] end_line=bi[{xd0.end_line.index}]")
print(f"  start: {xd0.start.type}@{xd0.start.k.index} val={xd0.start.val:.1f}")
print(f"  end: {xd0.end.type}@{xd0.end.k.index} val={xd0.end.val:.1f}")
print(f"  done={xd0.done}")

# Check if XD has tzxls attribute
print(f"  has tzxls: {hasattr(xd0, 'tzxls')}")
if hasattr(xd0, 'tzxls') and xd0.tzxls:
    print(f"  tzxls count: {len(xd0.tzxls)}")
    for i, xl in enumerate(xd0.tzxls):
        lines_str = [l.index for l in xl.lines] if xl.lines else []
        print(f"    [{i}] dir={xl.bh_direction} max={xl.max:.1f} min={xl.min:.1f} lines=bi{lines_str} bad={xl.line_bad}")
else:
    print("  No tzxls stored")

# Check ding_fx and di_fx
print(f"\n  ding_fx: {xd0.ding_fx}")
if xd0.ding_fx:
    dfx = xd0.ding_fx
    print(f"    type={dfx.type} high={dfx.high:.1f} low={dfx.low:.1f}")
    for j, xl in enumerate(dfx.xls):
        if xl:
            lines = [l.index for l in xl.lines] if xl.lines else []
            print(f"    xls[{j}]: max={xl.max:.1f} min={xl.min:.1f} lines=bi{lines}")
        else:
            print(f"    xls[{j}]: None")
            
print(f"\n  di_fx: {xd0.di_fx}")
if xd0.di_fx:
    dfx = xd0.di_fx
    print(f"    type={dfx.type} high={dfx.high:.1f} low={dfx.low:.1f}")
    for j, xl in enumerate(dfx.xls):
        if xl:
            lines = [l.index for l in xl.lines] if xl.lines else []
            print(f"    xls[{j}]: max={xl.max:.1f} min={xl.min:.1f} lines=bi{lines}")
        else:
            print(f"    xls[{j}]: None")

# Also check the second XD
print(f"\n{'='*60}")
xd1 = cd_p.xds[1]
print(f"XD[1]: {xd1.type} start_line=bi[{xd1.start_line.index}] end_line=bi[{xd1.end_line.index}]")
print(f"  has tzxls: {hasattr(xd1, 'tzxls')}")
if hasattr(xd1, 'tzxls') and xd1.tzxls:
    print(f"  tzxls count: {len(xd1.tzxls)}")
    for i, xl in enumerate(xd1.tzxls):
        lines_str = [l.index for l in xl.lines] if xl.lines else []
        print(f"    [{i}] dir={xl.bh_direction} max={xl.max:.1f} min={xl.min:.1f} lines=bi{lines_str} bad={xl.line_bad}")
        
if xd1.ding_fx:
    dfx = xd1.ding_fx
    print(f"\n  ding_fx: type={dfx.type} high={dfx.high:.1f} low={dfx.low:.1f}")
    for j, xl in enumerate(dfx.xls):
        if xl:
            lines = [l.index for l in xl.lines] if xl.lines else []
            print(f"    xls[{j}]: max={xl.max:.1f} min={xl.min:.1f} lines=bi{lines}")
        else:
            print(f"    xls[{j}]: None")

if xd1.di_fx:
    dfx = xd1.di_fx
    print(f"\n  di_fx: type={dfx.type} high={dfx.high:.1f} low={dfx.low:.1f}")
    for j, xl in enumerate(dfx.xls):
        if xl:
            lines = [l.index for l in xl.lines] if xl.lines else []
            print(f"    xls[{j}]: max={xl.max:.1f} min={xl.min:.1f} lines=bi{lines}")
        else:
            print(f"    xls[{j}]: None")
