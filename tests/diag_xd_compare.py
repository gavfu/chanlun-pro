"""Detailed XD comparison and stroke/segment analysis."""
import sys; sys.path.insert(0, 'src')
import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')
cd_o = CL_O('BTC/USDT', '60m')
cd_o.process_klines(df)
cd_p = CL_P('BTC/USDT', '60m')
cd_p.process_klines(df)

print("=== ALL STROKES (BIs) ===")
for i, bi in enumerate(cd_p.bis):
    print(f"  bi[{i}] {bi.type} {bi.start.k.index}→{bi.end.k.index} "
          f"high={bi.high:.1f} low={bi.low:.1f}")

print("\n=== PYARMOR XDs ===")
for i, xd in enumerate(cd_p.xds):
    print(f"  xd[{i}] {xd.type} start_line={xd.start_line.index}(bi {xd.start_line.type} {xd.start_line.start.k.index}→{xd.start_line.end.k.index}) "
          f"end_line={xd.end_line.index}(bi {xd.end_line.type} {xd.end_line.start.k.index}→{xd.end_line.end.k.index}) "
          f"high={xd.high:.1f} low={xd.low:.1f} done={xd.done}")

print("\n=== OUR XDs ===")
for i, xd in enumerate(cd_o.xds):
    print(f"  xd[{i}] {xd.type} start_line={xd.start_line.index}(bi {xd.start_line.type} {xd.start_line.start.k.index}→{xd.start_line.end.k.index}) "
          f"end_line={xd.end_line.index}(bi {xd.end_line.type} {xd.end_line.start.k.index}→{xd.end_line.end.k.index}) "
          f"high={xd.high:.1f} low={xd.low:.1f} done={xd.done}")

# Also check how many BIs each pyarmor XD spans
print("\n=== PYARMOR XD BI Composition ===")
for i, xd in enumerate(cd_p.xds):
    start_idx = xd.start_line.index
    end_idx = xd.end_line.index
    bi_count = end_idx - start_idx + 1
    print(f"  xd[{i}] spans bi[{start_idx}]..bi[{end_idx}] ({bi_count} strokes)")
    for j in range(start_idx, end_idx + 1):
        bi = cd_p.bis[j]
        print(f"    bi[{j}] {bi.type} {bi.start.k.index}→{bi.end.k.index} h={bi.high:.1f} l={bi.low:.1f}")

# XD-related config values
print("\n=== XD Config ===")
for attr in ['xd_qj', 'xd_allow_bi_pohuai', 'xd_allow_split_no_highlow', 'xd_allow_split_zs_kz']:
    if hasattr(cd_o, attr):
        print(f"  {attr} = {getattr(cd_o, attr)}")
