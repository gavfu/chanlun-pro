"""Test trailing fractal fix and verify BIs still match."""
import sys; sys.path.insert(0, 'src')
import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')
cd_o = CL_O('BTC/USDT', '60m')
cd_o.process_klines(df)
cd_p = CL_P('BTC/USDT', '60m')
cd_p.process_klines(df)

print(f'FXs: ours={len(cd_o.fxs)}, pyarmor={len(cd_p.fxs)}')
print(f'BIs: ours={len(cd_o.bis)}, pyarmor={len(cd_p.bis)}')

# Check ALL FXs match
fx_mismatches = []
for i in range(max(len(cd_o.fxs), len(cd_p.fxs))):
    o = cd_o.fxs[i] if i < len(cd_o.fxs) else None
    p = cd_p.fxs[i] if i < len(cd_p.fxs) else None
    if o and p:
        if o.type != p.type or o.k.index != p.k.index or abs(o.val - p.val) > 0.1 or o.done != p.done:
            fx_mismatches.append(i)
    else:
        fx_mismatches.append(i)

if fx_mismatches:
    print(f'FX mismatches at indices: {fx_mismatches}')
    for i in fx_mismatches:
        o = cd_o.fxs[i] if i < len(cd_o.fxs) else None
        p = cd_p.fxs[i] if i < len(cd_p.fxs) else None
        o_str = f'{o.type}@{o.k.index} val={o.val:.1f} done={o.done}' if o else 'MISSING'
        p_str = f'{p.type}@{p.k.index} val={p.val:.1f} done={p.done}' if p else 'MISSING'
        print(f'  [{i}] ours={o_str} | pyarmor={p_str}')
else:
    print('ALL FXs match!')

# Check BIs still match
bi_ok = True
for i in range(min(len(cd_o.bis), len(cd_p.bis))):
    o = cd_o.bis[i]
    p = cd_p.bis[i]
    if o.type != p.type or o.start.k.index != p.start.k.index or o.end.k.index != p.end.k.index:
        bi_ok = False
        print(f'BI[{i}] MISMATCH: ours={o.type} {o.start.k.index}->{o.end.k.index}, pyarmor={p.type} {p.start.k.index}->{p.end.k.index}')
if len(cd_o.bis) != len(cd_p.bis):
    bi_ok = False
    print(f'BI count mismatch: {len(cd_o.bis)} vs {len(cd_p.bis)}')
if bi_ok:
    print('ALL BIs match!')
