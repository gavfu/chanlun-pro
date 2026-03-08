"""Test different ZG/ZD calculation hypotheses against pyarmor output.

Pyarmor ZS[0]: ZG=69033.0 ZD=68112.2 GG=70110.9 DD=66588.0 lines=[0→6]
"""
import sys; sys.path.insert(0, 'src')
import pandas as pd
from chanlun.cl_pyarmor import CL as CL_P

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')
cd_p = CL_P('BTC/USDT', '60m')
cd_p.process_klines(df)
bis = cd_p.bis

# ZS[0] expected: ZG=69033.0 ZD=68112.2 GG=70110.9 DD=66588.0

bi_vals = [(bi.index, bi.type, bi.high, bi.low) for bi in bis[:7]]
for idx, typ, h, l in bi_vals:
    print(f"  bi[{idx}] {typ:4s} h={h:.1f} l={l:.1f}")

print()

# Hypothesis 1: min(h0,h1,h2), max(l0,l1,l2) - first 3
h1 = min(bi_vals[0][2], bi_vals[1][2], bi_vals[2][2])
l1 = max(bi_vals[0][3], bi_vals[1][3], bi_vals[2][3])
print(f"H1 - first 3: ZG={h1:.1f} ZD={l1:.1f}")

# Hypothesis 2: min(h1,h2), max(l1,l2) - 2nd and 3rd only
h2 = min(bi_vals[1][2], bi_vals[2][2])
l2 = max(bi_vals[1][3], bi_vals[2][3])
print(f"H2 - 2nd,3rd: ZG={h2:.1f} ZD={l2:.1f}")

# Hypothesis 3: min(h1,h2,h3), max(l1,l2,l3) - 2nd,3rd,4th
h3 = min(bi_vals[1][2], bi_vals[2][2], bi_vals[3][2])
l3 = max(bi_vals[1][3], bi_vals[2][3], bi_vals[3][3])
print(f"H3 - 2nd,3rd,4th: ZG={h3:.1f} ZD={l3:.1f}")

# Hypothesis 4: use pairs of same direction
# DOWN: bi[0],bi[2],bi[4]; UP: bi[1],bi[3],bi[5]
# ZG = min(bi[1].h, bi[3].h) and ZD = max(bi[1].l, bi[3].l)?
h4 = min(bi_vals[1][2], bi_vals[3][2])
l4 = max(bi_vals[1][3], bi_vals[3][3])
print(f"H4 - bi[1],bi[3] (UP pairs): ZG={h4:.1f} ZD={l4:.1f}")

# Hypothesis 5: DOWN pairs bi[2],bi[4] excluding first
h5 = min(bi_vals[2][2], bi_vals[4][2])
l5 = max(bi_vals[2][3], bi_vals[4][3])
print(f"H5 - bi[2],bi[4] (DOWN pairs): ZG={h5:.1f} ZD={l5:.1f}")

# Hypothesis 6: Cross-pair: ZG = min(DOWN_high_1, DOWN_high_2) = min(bi[2].h, bi[4].h)
# and ZD = max(UP_low_1, UP_low_2) = max(bi[1].l, bi[3].l)
h6 = min(bi_vals[2][2], bi_vals[4][2])
l6 = max(bi_vals[1][3], bi_vals[3][3])
print(f"H6 - cross-pair: ZG={h6:.1f} ZD={l6:.1f}")

# Hypothesis 7: ZG = min of 2nd high and 3rd high (lines within ZS)
# where "entering" = bi[0], "within" = bi[1],bi[2], "leaving" = bi[3]
# So ZG = min(h1, h2), ZD = max(l1, l2)  
# Already tested as H2

# Hypothesis 8: use lines[2] and lines[3] (skipping first 2)
h8 = min(bi_vals[2][2], bi_vals[3][2])
l8 = max(bi_vals[2][3], bi_vals[3][3])
print(f"H8 - bi[2],bi[3]: ZG={h8:.1f} ZD={l8:.1f}")

# Hypothesis 9: use 3rd,4th,5th  
h9 = min(bi_vals[2][2], bi_vals[3][2], bi_vals[4][2])
l9 = max(bi_vals[2][3], bi_vals[3][3], bi_vals[4][3])
print(f"H9 - bi[2],bi[3],bi[4]: ZG={h9:.1f} ZD={l9:.1f}")

print(f"\nExpected: ZG=69033.0 ZD=68112.2")
# H3 matches! ZG=69033.0 ZD=68112.2 using bi[1],bi[2],bi[3]

# Now verify ZS[1]: expected ZG=67299.4 ZD=65826.1
print("\n=== ZS[1] verification ===")
# ZS[1] starts at bi[6]
# H3 hypothesis: use lines[1],lines[2],lines[3] = bi[7],bi[8],bi[9]
h = min(bis[7].high, bis[8].high, bis[9].high)
l = max(bis[7].low, bis[8].low, bis[9].low)
print(f"H3 - bi[7],bi[8],bi[9]: ZG={h:.1f} ZD={l:.1f}")
print(f"Expected: ZG=67299.4 ZD=65826.1")
# bi[7] h=67299.4 l=65826.1, bi[8] h=67299.4 l=65595.7, bi[9] h=68283.7 l=65595.7
# ZG = min(67299.4, 67299.4, 68283.7) = 67299.4 ✓
# ZD = max(65826.1, 65595.7, 65595.7) = 65826.1 ✓

# Also check ZS[2]: expected ZG=68524.9 ZD=67712.3 lines=[23→26]
print("\n=== ZS[2] verification ===")
# ZS[2] starts at bi[23]
# H3: bi[24],bi[25],bi[26]
h = min(bis[24].high, bis[25].high, bis[26].high)
l = max(bis[24].low, bis[25].low, bis[26].low)
print(f"H3 - bi[24],bi[25],bi[26]: ZG={h:.1f} ZD={l:.1f}")
print(f"Expected: ZG=68524.9 ZD=67712.3")
