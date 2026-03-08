"""Analyze containment direction for all TZXL pairs in XD[0]."""
import sys; sys.path.insert(0, 'src')
import pandas as pd
from chanlun.cl_pyarmor import CL as CL_P

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')
cd_p = CL_P('BTC/USDT', '60m')
cd_p.process_klines(df)

bis = cd_p.bis

# For XD[0] (down), the characteristic seq uses UP strokes
# bh_direction = down
up_bis = [bi for bi in bis if bi.type == "up"]

print("=== UP strokes (used for DOWN XD characteristic sequence) ===")
for bi in up_bis:
    print(f"  bi[{bi.index}] h={bi.high:.1f} l={bi.low:.1f}")

print("\n=== Pairwise containment analysis ===")
for i in range(len(up_bis) - 1):
    a = up_bis[i]
    b = up_bis[i + 1]
    
    # Check containment using raw high/low
    a_contains_b = a.high >= b.high and a.low <= b.low
    b_contains_a = b.high >= a.high and b.low <= a.low
    
    has_contain = a_contains_b or b_contains_a
    direction = ""
    if a_contains_b and b_contains_a:
        direction = "EQUAL"
    elif a_contains_b:
        direction = "OLD contains NEW"
    elif b_contains_a:
        direction = "NEW contains OLD"
    
    if has_contain:
        print(f"  bi[{a.index}] vs bi[{b.index}]: {direction}")
        print(f"    a: h={a.high:.1f} l={a.low:.1f}")
        print(f"    b: h={b.high:.1f} l={b.low:.1f}")

# Now look at pyarmor's actual xd[0].tzxls to see which ones got merged
print("\n=== Pyarmor XD[0] tzxls (actual result) ===")
xd0 = cd_p.xds[0]
for i, xl in enumerate(xd0.tzxls):
    lines = [l.index for l in xl.lines]
    merged = len(lines) > 1
    print(f"  [{i}] max={xl.max:.1f} min={xl.min:.1f} lines=bi{lines} "
          f"bad={xl.line_bad} {'MERGED' if merged else ''}")

# Now check: which up_bis pairs had containment?
# bi[7] vs bi[9]: b_contains_a → NOT merged in pyarmor, line_bad=True (on bi[9])
# bi[15] vs bi[17]: a_contains_b → MERGED in pyarmor
# bi[23] vs bi[25]: ??? 

print("\n=== Pyarmor XD[1] tzxls ===")
xd1 = cd_p.xds[1]
for i, xl in enumerate(xd1.tzxls):
    lines = [l.index for l in xl.lines]
    merged = len(lines) > 1
    print(f"  [{i}] max={xl.max:.1f} min={xl.min:.1f} lines=bi{lines} "
          f"bad={xl.line_bad} {'MERGED' if merged else ''}")
