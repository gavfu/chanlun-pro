# -*- coding: utf-8 -*-
"""
Check strict values with different qj/qy parameters
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "test_data", "BTC_USDT_60m_500.parquet"))

from chanlun.cl_open import CL
cd = CL("BTC/USDT", "60m")
cd.process_klines(df)

fx_by_idx = {fx.k.index: fx for fx in cd.fxs}

start = fx_by_idx[347]
end = fx_by_idx[355]

configs = [
    ("fx_qj_k",  "fx_qy_three"),
    ("fx_qj_k",  "fx_qy_one"),
    ("fx_qj_ck", "fx_qy_three"),
    ("fx_qj_ck", "fx_qy_one"),
]

print("=== di@347 → ding@355 strict check with different configs ===")
for qj, qy in configs:
    sh = start.high(qj, qy)
    eh = end.high(qj, qy)
    sl = start.low(qj, qy)
    el = end.low(qj, qy)
    print(f"\n{qj}, {qy}:")
    print(f"  start.high={sh:.1f}, end.high={eh:.1f} → sh>eh = {sh > eh}")
    print(f"  end.low={el:.1f}, start.low={sl:.1f} → el<sl = {el < sl}")
    print(f"  → Check1 blocks: {sh > eh}, Check2 blocks: {el < sl}")
    
# Also check with val (FX.val = CLKline high/low)
print(f"\n=== Using FX.val directly ===")
print(f"  start.val = {start.val:.1f} (di: low of CLKline)")
print(f"  end.val = {end.val:.1f} (ding: high of CLKline)")
print(f"  start CLK h={start.k.h:.1f}, l={start.k.l:.1f}")
print(f"  end CLK h={end.k.h:.1f}, l={end.k.l:.1f}")

# Check with just CLKline h/l
print(f"\n=== Using CLKline h/l directly ===")
print(f"  start.k.h={start.k.h:.1f} > end.k.h={end.k.h:.1f} = {start.k.h > end.k.h}")
print(f"  end.k.l={end.k.l:.1f} < start.k.l={start.k.l:.1f} = {end.k.l < start.k.l}")
