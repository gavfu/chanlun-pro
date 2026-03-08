"""Debug BI ZS differences.

ZS[0]: Open ZG=69230.0 ZD=68112.2 lines=[0→6]
       Pyarmor ZG=69033.0 ZD=68112.2 lines=[0→6]

The ZG difference is 69230 vs 69033. 
bi[0] h=69230.0, bi[3] h=69033.0

ZS is formed by overlapping region of lines.
For ZS from lines [0→6], the ZG should be min of the highs of lines 1 and 2:
- bi[1] h=70110.9, bi[2] h=70110.9 → ZG = min(70110.9, 70110.9) = 70110.9? No...

Actually ZS definition: ZG = min(max of entering lines), ZD = max(min of entering lines)
Let me check how ZS is defined properly.
"""
import sys; sys.path.insert(0, 'src')
import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')
cd_o = CL_O('BTC/USDT', '60m')
cd_o.process_klines(df)
cd_p = CL_P('BTC/USDT', '60m')
cd_p.process_klines(df)

print("=== BI ZS details ===")
for label, cd in [("Open", cd_o), ("Pyarmor", cd_p)]:
    print(f"\n--- {label} ---")
    for i, zs in enumerate(cd.bi_zss):
        lines_idx = [l.index for l in zs.lines]
        print(f"  ZS[{i}]: ZG={zs.zg:.1f} ZD={zs.zd:.1f} GG={zs.gg:.1f} DD={zs.dd:.1f}")
        print(f"         type={zs.zs_type} level={zs.level}")
        print(f"         lines={lines_idx}")
        print(f"         line_num={zs.line_num} real={zs.real}")
        for l in zs.lines:
            print(f"           bi[{l.index}] {l.type} h={l.high:.1f} l={l.low:.1f}")

# Also check BI high/low values
print("\n=== BI high/low comparison ===")
for i in range(min(8, len(cd_o.bis))):
    o = cd_o.bis[i]
    p = cd_p.bis[i]
    match_h = "✓" if abs(o.high - p.high) < 0.01 else "✗"
    match_l = "✓" if abs(o.low - p.low) < 0.01 else "✗"
    print(f"  bi[{i}] {o.type}: O(h={o.high:.1f},l={o.low:.1f})  P(h={p.high:.1f},l={p.low:.1f})  {match_h}{match_l}")

# Check XD high/low
print("\n=== XD high/low comparison ===")
for i in range(min(len(cd_o.xds), len(cd_p.xds))):
    o = cd_o.xds[i]
    p = cd_p.xds[i]
    match_h = "✓" if abs(o.high - p.high) < 0.01 else "✗"
    match_l = "✓" if abs(o.low - p.low) < 0.01 else "✗"
    print(f"  XD[{i}] {o.type}: O(h={o.high:.1f},l={o.low:.1f})  P(h={p.high:.1f},l={p.low:.1f})  {match_h}{match_l}")
