# -*- coding: utf-8 -*-
"""
Compare FX values between cl_open and cl_pyarmor for FX indices 315-380.
Check if fractals themselves differ.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "test_data", "BTC_USDT_60m_500.parquet"))

from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

cd_o = CL_O("BTC/USDT", "60m")
cd_o.process_klines(df)

cd_p = CL_P("BTC/USDT", "60m")
cd_p.process_klines(df)

qj = "fx_qj_k"
qy = "fx_qy_three"

print("=== FX comparison around 315-380 ===")
print(f"{'idx':>4} {'type':>4} {'val_o':>10} {'val_p':>10} {'high_o':>10} {'high_p':>10} {'low_o':>10} {'low_p':>10} {'match':>5}")

# Build index lookup for pyarmor
fx_p_by_idx = {fx.k.index: fx for fx in cd_p.fxs}
fx_o_by_idx = {fx.k.index: fx for fx in cd_o.fxs}

# All fx indices in range
all_idxs = sorted(set(list(fx_o_by_idx.keys()) + list(fx_p_by_idx.keys())))
for idx in all_idxs:
    if idx < 310 or idx > 380:
        continue
    fo = fx_o_by_idx.get(idx)
    fp = fx_p_by_idx.get(idx)
    if fo and fp:
        val_match = abs(fo.val - fp.val) < 0.1
        high_match = abs(fo.high(qj, qy) - fp.high(qj, qy)) < 0.1
        low_match = abs(fo.low(qj, qy) - fp.low(qj, qy)) < 0.1
        all_match = val_match and high_match and low_match and fo.type == fp.type
        print(f"{idx:4d} {fo.type:>4} {fo.val:10.1f} {fp.val:10.1f} {fo.high(qj,qy):10.1f} {fp.high(qj,qy):10.1f} {fo.low(qj,qy):10.1f} {fp.low(qj,qy):10.1f} {'OK' if all_match else 'DIFF'}")
    elif fo:
        print(f"{idx:4d} {fo.type:>4} {fo.val:10.1f} {'---':>10} {fo.high(qj,qy):10.1f} {'---':>10} {fo.low(qj,qy):10.1f} {'---':>10} ONLY_OPEN")
    elif fp:
        print(f"{idx:4d} {fp.type:>4} {'---':>10} {fp.val:10.1f} {'---':>10} {fp.high(qj,qy):10.1f} {'---':>10} {fp.low(qj,qy):10.1f} ONLY_PYARMOR")

print(f"\n=== di@347 details ===")
fo = fx_o_by_idx.get(347)
fp = fx_p_by_idx.get(347)
if fo:
    print(f"  Open:    type={fo.type}, val={fo.val}, high={fo.high(qj,qy)}, low={fo.low(qj,qy)}")
    print(f"           k.index={fo.k.index}, k.k_index={fo.k.k_index}")
    print(f"           CLKlines: ", end="")
    for clk in [fo.elements[0], fo.k, fo.elements[2]]:
        print(f"[idx={clk.index}, h={clk.h:.1f}, l={clk.l:.1f}, klines={[k.k_index for k in clk.klines]}]", end=" ")
    print()
if fp:
    print(f"  Pyarmor: type={fp.type}, val={fp.val}, high={fp.high(qj,qy)}, low={fp.low(qj,qy)}")
    print(f"           k.index={fp.k.index}, k.k_index={fp.k.k_index}")
    print(f"           CLKlines: ", end="")
    for clk in [fp.elements[0], fp.k, fp.elements[2]]:
        print(f"[idx={clk.index}, h={clk.h:.1f}, l={clk.l:.1f}, klines={[k.k_index for k in clk.klines]}]", end=" ")
    print()

# Check key strict values
print(f"\n=== Strict check values for UP di@347 -> ding@X ===")
for ding_idx in [348, 351, 353, 355]:
    fo_start = fx_o_by_idx.get(347)
    fo_end = fx_o_by_idx.get(ding_idx)
    fp_start = fx_p_by_idx.get(347)
    fp_end = fx_p_by_idx.get(ding_idx)
    
    if fo_start and fo_end:
        sh = fo_start.high(qj, qy)
        eh = fo_end.high(qj, qy)
        print(f"  Open:    di@347->ding@{ding_idx}: start.high={sh:.1f} > end.high={eh:.1f} = {sh > eh} (blocked={sh > eh})")
    if fp_start and fp_end:
        sh = fp_start.high(qj, qy)
        eh = fp_end.high(qj, qy)
        print(f"  Pyarmor: di@347->ding@{ding_idx}: start.high={sh:.1f} > end.high={eh:.1f} = {sh > eh} (blocked={sh > eh})")
    print()
