# -*- coding: utf-8 -*-
"""
Trace pyarmor's _build_bis decisions for ETH/USDT 60m.
Captures: start_fx assignments, end_fx assignments, BI confirmations.
Uses call-level + selective line-level sys.settrace.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from chanlun.cl_pyarmor import CL as CL_Pyarmor
from chanlun.cl_interface import FX, BI

DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")

def load(symbol, freq, limit):
    cache_key = f"{symbol.replace('/', '_')}_{freq}_{limit}"
    return pd.read_parquet(os.path.join(DATA_DIR, f"{cache_key}.parquet"))


# ---- First, run without tracing to get baseline ----
df = load("ETH/USDT", "60m", 1000)

cd = CL_Pyarmor("ETH/USDT", "60m", config={})
cd.process_klines(df)
bis = cd.get_bis()
fxs = cd.get_fxs()

print(f"Total BIs: {len(bis)}, Total FXs: {len(fxs)}")
print(f"\nFirst 20 BIs:")
for b in bis[:20]:
    sp = " [SPLIT]" if b.is_split else ""
    print(f"  #{b.index:>2} {b.type:>4} [{b.start.k.index:>3}→{b.end.k.index:<3}] "
          f"start_val={b.start.val:.2f} end_val={b.end.val:.2f}{sp}")

print(f"\nFXs around index 30-60:")
for fx in fxs:
    if 30 <= fx.k.index <= 60:
        print(f"  FX({fx.k.index},{fx.type}) val={fx.val:.2f} k_index={fx.k.k_index}")

# Also get cl_open FXs for comparison
from chanlun.cl_open import CL as CL_Open
cd_o = CL_Open("ETH/USDT", "60m", config={})
cd_o.process_klines(df)
o_fxs = cd_o.get_fxs()
o_bis_raw = []

# Get pre-split BIs
import types
orig = cd_o._bi_special_bi_split.__func__
def capture(self, bis):
    o_bis_raw.extend(bis)
    return orig(self, bis)
cd_o2 = CL_Open("ETH/USDT", "60m", config={})
cd_o2._bi_special_bi_split = types.MethodType(capture, cd_o2)
cd_o2.process_klines(df)

print(f"\nOpen pre-split first 20 BIs:")
for b in o_bis_raw[:20]:
    print(f"  #{b.index:>2} {b.type:>4} [{b.start.k.index:>3}→{b.end.k.index:<3}]")

# Check if FXs match
print(f"\nOpen FXs count={len(o_fxs)}, Pyarmor FXs count={len(fxs)}")
# Compare FX sequences
fx_match = True
for i in range(min(len(o_fxs), len(fxs))):
    if o_fxs[i].k.index != fxs[i].k.index or o_fxs[i].type != fxs[i].type:
        print(f"  FX#{i}: open=({o_fxs[i].type},{o_fxs[i].k.index}) pyarmor=({fxs[i].type},{fxs[i].k.index})")
        fx_match = False
        if i > 10:
            print("  ... (more diffs)")
            break
if fx_match:
    print("  All FXs match ✅")
