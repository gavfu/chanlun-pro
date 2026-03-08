"""Deeper analysis of pyarmor ZS construction.

Key questions:
1. When does ZS start? (confirmed: lines[0] is the first line entering)
2. ZG/ZD formula: confirmed lines[1],lines[2],lines[3] (H3)
3. GG/DD during extension
4. How does ZS terminate?
5. Level determination
"""
import sys; sys.path.insert(0, 'src')
import pandas as pd
from chanlun.cl_pyarmor import CL as CL_P
from chanlun.cl_interface import ZS

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')

# Intercept ZS creation
original_init = ZS.__init__
zs_creations = []

def tracked_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    import traceback
    tb = ''.join([l.strip() for l in traceback.format_stack() if 'frozen' in l])
    zs_creations.append({
        'zg': self.zg, 'zd': self.zd, 'gg': self.gg, 'dd': self.dd,
        'tb': tb
    })

ZS.__init__ = tracked_init

cd_p = CL_P('BTC/USDT', '60m')
cd_p.process_klines(df)

print(f"ZS creations: {len(zs_creations)}")
ZS.__init__ = original_init

print("\n=== Final BI ZS ===")
for i, zs in enumerate(cd_p.bi_zss):
    lines = [l.index for l in zs.lines]
    print(f"  ZS[{i}] level={zs.level} ZG={zs.zg:.1f} ZD={zs.zd:.1f} GG={zs.gg:.1f} DD={zs.dd:.1f}")
    print(f"          lines={lines}")

print("\n=== ZS GG/DD vs line highs/lows ===")
bis = cd_p.bis
for i, zs in enumerate(cd_p.bi_zss):
    print(f"\n  ZS[{i}]:")
    for l in zs.lines:
        print(f"    bi[{l.index}] {l.type} h={l.high:.1f} l={l.low:.1f}")
    all_highs = [l.high for l in zs.lines]
    all_lows = [l.low for l in zs.lines]
    print(f"  Computed GG={max(all_highs):.1f} (actual={zs.gg:.1f})")
    print(f"  Computed DD={min(all_lows):.1f} (actual={zs.dd:.1f})")
    
    # Try using lines[1:] for GG/DD
    all_highs_skip1 = [l.high for l in zs.lines[1:]]
    all_lows_skip1 = [l.low for l in zs.lines[1:]]
    print(f"  GG (skip 1st)={max(all_highs_skip1):.1f}, DD (skip 1st)={min(all_lows_skip1):.1f}")
    
    # level check
    level_lines = [l.index for l in zs.lines]
    print(f"  level={zs.level}")
