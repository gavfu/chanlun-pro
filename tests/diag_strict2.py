# -*- coding: utf-8 -*-
"""
More systematic reverse-engineering:
Compare the k_gap < 13 strokes between strict=on and strict=off
to understand EXACTLY which candidates get blocked by strict check.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from chanlun.cl_pyarmor import CL

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "test_data", "BTC_USDT_60m_500.parquet"))

# Test with cgd_yes (default) + strict on/off
cd_on = CL("BTC/USDT", "60m")
cd_on.process_klines(df)
cd_off = CL("BTC/USDT", "60m", config={"allow_bi_fx_strict": 0})
cd_off.process_klines(df)

bis_on = cd_on.get_bis()
bis_off = cd_off.get_bis()

print(f"Strict ON: {len(bis_on)} strokes")
print(f"Strict OFF: {len(bis_off)} strokes")

# Find strokes that exist in OFF but not in ON
# (these are strokes that strict check BLOCKS)
on_set = set()
for bi in bis_on:
    on_set.add((bi.start.k.index, bi.end.k.index, bi.type))

off_set = set()
for bi in bis_off:
    off_set.add((bi.start.k.index, bi.end.k.index, bi.type))

blocked = off_set - on_set
allowed = on_set - off_set

print(f"\nStrokes in OFF but not in ON (BLOCKED by strict):")
for s, e, t in sorted(blocked):
    # Find the FX details
    for bi in bis_off:
        if bi.start.k.index == s and bi.end.k.index == e:
            sfx = bi.start
            efx = bi.end
            k_gap = efx.k.k_index - sfx.k.k_index
            sh = sfx.high('fx_qj_k', 'fx_qy_three')
            sl = sfx.low('fx_qj_k', 'fx_qy_three')
            eh = efx.high('fx_qj_k', 'fx_qy_three')
            el = efx.low('fx_qj_k', 'fx_qy_three')
            print(f"  {t:5s} {s:3d}->{e:3d} k={k_gap:3d} sh={sh:.1f} sl={sl:.1f} eh={eh:.1f} el={el:.1f}")
            break

print(f"\nStrokes in ON but not in OFF (ALLOWED by strict only exist because other strokes got blocked):")
for s, e, t in sorted(allowed):
    for bi in bis_on:
        if bi.start.k.index == s and bi.end.k.index == e:
            sfx = bi.start
            efx = bi.end  
            k_gap = efx.k.k_index - sfx.k.k_index
            sh = sfx.high('fx_qj_k', 'fx_qy_three')
            sl = sfx.low('fx_qj_k', 'fx_qy_three')
            eh = efx.high('fx_qj_k', 'fx_qy_three')
            el = efx.low('fx_qj_k', 'fx_qy_three')
            print(f"  {t:5s} {s:3d}->{e:3d} k={k_gap:3d} sh={sh:.1f} sl={sl:.1f} eh={eh:.1f} el={el:.1f}")
            break

# Print both full stroke lists side by side
print(f"\n=== STRICT ON ({len(bis_on)} strokes) ===")
for i, bi in enumerate(bis_on):
    k_gap = bi.end.k.k_index - bi.start.k.k_index
    in_off = (bi.start.k.index, bi.end.k.index, bi.type) in off_set
    marker = "" if in_off else " [ONLY-ON]"
    print(f"  bi[{i:2d}] {bi.type:5s} {bi.start.k.index:3d}->{bi.end.k.index:3d} k={k_gap:3d}{marker}")

print(f"\n=== STRICT OFF ({len(bis_off)} strokes) ===")
for i, bi in enumerate(bis_off):
    k_gap = bi.end.k.k_index - bi.start.k.k_index
    in_on = (bi.start.k.index, bi.end.k.index, bi.type) in on_set
    marker = "" if in_on else " [ONLY-OFF]"
    print(f"  bi[{i:2d}] {bi.type:5s} {bi.start.k.index:3d}->{bi.end.k.index:3d} k={k_gap:3d}{marker}")
