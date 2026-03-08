"""Check bi[24] 3buy details and bi[7] vs bi[13] 3sell details"""
import sys; sys.path.insert(0, 'src')
from chanlun.cl_open import CL
from chanlun.cl_pyarmor import CL as CLP
import pandas as pd

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')
config = {}
cdp = CLP('BTC/USDT', '60m', config=config); cdp.process_klines(df)
bi_zssp = cdp.get_bi_zss()
bisp = cdp.get_bis()

# Print ZS details
for i, zs in enumerate(bi_zssp):
    print(f"PyZS[{i}]: ZG={zs.zg:.1f} ZD={zs.zd:.1f} lines=[{zs.lines[0].index}..{zs.lines[-1].index}] line_num={zs.line_num}")

# bi[7] vs ZS[0]:
print(f"\nbi[7] up: high={bisp[7].high:.1f}")
print(f"ZS[0].zd={bi_zssp[0].zd:.1f}")
print(f"bi[7].high < ZS[0].zd: {bisp[7].high < bi_zssp[0].zd}")

# bi[13] vs ZS[0]:
print(f"\nbi[13] up: high={bisp[13].high:.1f}")
print(f"bi[13].high < ZS[0].zd: {bisp[13].high < bi_zssp[0].zd}")
print(f"bi[13].index={bisp[13].index}, ZS[0].last_line.index={bi_zssp[0].lines[-1].index}")

# bi[24] vs ZSs:
print(f"\nbi[24] down: low={bisp[24].low:.1f} high={bisp[24].high:.1f}")
for i, zs in enumerate(bi_zssp):
    last_idx = zs.lines[-1].index
    if bisp[24].index > last_idx and bisp[24].type == "down":
        print(f"  vs ZS[{i}]: ZG={zs.zg:.1f} low>zg: {bisp[24].low > zs.zg}")
        print(f"  bisp[24].index-last_idx = {bisp[24].index - last_idx}")

# Check if pyarmor gives 3buy to bi[24]:
print(f"\nbi[24] mmds: {bisp[24].line_mmds('|')}")

# Check bi[24] bcs:
print(f"bi[24] bcs: {bisp[24].line_bcs('|')}")

# Show all bcs+mmds for bi 23-26:
for j in range(22, 27):
    bp = bisp[j]
    print(f"bi[{j:2d}] {bp.type:4s}: high={bp.high:.1f} low={bp.low:.1f} mmds={bp.line_mmds('|')} bcs={bp.line_bcs('|')}")

# Now check: for bi[7] and bi[13], both are inside ZS[1] (lines 6..23)
# Why does bi[7] get 3sell but bi[13] doesn't?
print(f"\nbi[7].index={bisp[7].index}, in ZS[1] lines [6..23]: {6 <= bisp[7].index <= 23}")
print(f"bi[13].index={bisp[13].index}, in ZS[1] lines [6..23]: {6 <= bisp[13].index <= 23}")

# ZS[1] start_fx and end_fx:
# Maybe: if line is in some_zs.lines, skip 3sell for it
zs1 = bi_zssp[1]
print(f"\nZS[1].lines indices: {[l.index for l in zs1.lines]}")
# bi[7] IS in ZS[1].lines
print(f"bi[7] in ZS[1].lines: {bisp[7] in zs1.lines}")
print(f"bi[13] in ZS[1].lines: {bisp[13] in zs1.lines}")
