# -*- coding: utf-8 -*-
"""
Dump all key gaps for the confirmation that should work.
Focus on: 
- bi[24] down 315->347, confirmation by ding@351 (first valid gap)
- bi[25] up 347->355, confirmation by di@362 or di@358
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "test_data", "BTC_USDT_60m_500.parquet"))

from chanlun.cl_open import CL
cd = CL("BTC/USDT", "60m")
cd.process_klines(df)

fx_by_idx = {fx.k.index: fx for fx in cd.fxs}

print("=== bi[24] down 315→347 confirmation candidates ===")
start = fx_by_idx[315]
end = fx_by_idx[347]
print(f"start: ding@315 k_idx={start.k.k_index}")
print(f"end:   di@347   k_idx={end.k.k_index}")
print(f"main stroke k_gap = {end.k.k_index - start.k.k_index}")
print()

# Confirmation: looking for ding after di@347
for idx in [348, 351, 353, 355]:
    fx = fx_by_idx.get(idx)
    if fx and fx.type == "ding":
        cl_gap = fx.k.index - end.k.index
        k_gap = fx.k.k_index - end.k.k_index
        k_gap_from_start = fx.k.k_index - start.k.k_index
        print(f"  ding@{idx}: cl_gap={cl_gap}, k_gap(from end)={k_gap}, k_gap(from start)={k_gap_from_start}")
        print(f"    For confirmation UP di@347 -> ding@{idx}:")
        print(f"    strict check: start.high={end.high('fx_qj_k','fx_qy_three'):.1f} > end.high={fx.high('fx_qj_k','fx_qy_three'):.1f} = {end.high('fx_qj_k','fx_qy_three') > fx.high('fx_qj_k','fx_qy_three')}")
        print(f"    k_gap_from_end={k_gap} < 13 = {k_gap < 13} (strict applies)")
        print(f"    k_gap_from_start={k_gap_from_start} < 13 = {k_gap_from_start < 13} (strict applies if using start)")
        print()

print("\n=== bi[25] up 347→355 setting end_fx ===")
start2 = fx_by_idx[347]
for idx in [348, 351, 353, 355]:
    fx = fx_by_idx.get(idx)
    if fx and fx.type == "ding":
        cl_gap = fx.k.index - start2.k.index
        k_gap = fx.k.k_index - start2.k.k_index
        print(f"  ding@{idx}: cl_gap={cl_gap}, k_gap={k_gap}")
        if cl_gap >= 4 and k_gap >= 4:
            print(f"    gap OK, strict check: start.high={start2.high('fx_qj_k','fx_qy_three'):.1f} > end.high={fx.high('fx_qj_k','fx_qy_three'):.1f} = {start2.high('fx_qj_k','fx_qy_three') > fx.high('fx_qj_k','fx_qy_three')}")
        else:
            print(f"    gap FAILS")
        print()

print("\n=== bi[25] up 347→355 confirmation candidates ===")
end2 = fx_by_idx[355]
for idx in [358, 362, 365, 367, 370]:
    fx = fx_by_idx.get(idx)
    if fx and fx.type == "di":
        cl_gap = fx.k.index - end2.k.index
        k_gap = fx.k.k_index - end2.k.k_index
        k_gap_from_start = fx.k.k_index - start2.k.k_index
        print(f"  di@{idx}: cl_gap={cl_gap}, k_gap(from end)={k_gap}, k_gap(from start)={k_gap_from_start}")
        sh = end2.high('fx_qj_k','fx_qy_three')
        eh = fx.high('fx_qj_k','fx_qy_three')
        sl = end2.low('fx_qj_k','fx_qy_three')
        el = fx.low('fx_qj_k','fx_qy_three')
        print(f"    DOWN confirm: start.low={sl:.1f} < end.low={el:.1f} = {sl < el}")
        print(f"    DOWN confirm: end.high={eh:.1f} > start.high={sh:.1f} = {eh > sh}")
        # CGD check: between end2 and fx, any di more extreme?
        cgd_block = False
        for mid_idx in range(end2.k.index + 1, fx.k.index):
            mid_fx = fx_by_idx.get(mid_idx)
            if mid_fx and mid_fx.type == "di" and mid_fx.val < fx.val:
                print(f"    CGD: di@{mid_idx} val={mid_fx.val:.1f} < di@{idx} val={fx.val:.1f} → BLOCKS")
                cgd_block = True
                break
        if not cgd_block:
            print(f"    CGD: OK (no more extreme di between {end2.k.index} and {idx})")
        print()
