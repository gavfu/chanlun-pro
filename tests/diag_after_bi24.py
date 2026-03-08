# -*- coding: utf-8 -*-
"""
Trace what happens after bi[24] is confirmed in our algorithm.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "test_data", "BTC_USDT_60m_500.parquet"))

from chanlun.cl_open import CL
cd = CL("BTC/USDT", "60m")
cd.process_klines(df)

fx_by_idx = {fx.k.index: fx for fx in cd.fxs}
fxs = cd.fxs

# Find the index of di@347 in fxs
fx347_pos = None
for pos, fx in enumerate(fxs):
    if fx.k.index == 347:
        fx347_pos = pos
        break

print(f"di@347 is at fxs position {fx347_pos}")
print(f"Total fxs: {len(fxs)}")
print(f"\nRemaining fxs after di@347:")

for pos in range(fx347_pos, len(fxs)):
    fx = fxs[pos]
    h = fx.high("fx_qj_k", "fx_qy_three")
    l = fx.low("fx_qj_k", "fx_qy_three")
    print(f"  fxs[{pos:3d}] {fx.type:4s}@{fx.k.index:3d} val={fx.val:10.1f} high={h:10.1f} low={l:10.1f} k_idx={fx.k.k_index}")

# Simulate from di@347 as start_fx
print(f"\n=== Simulating UP stroke from di@347 ===")
start_fx = fxs[fx347_pos]
print(f"start_fx: di@{start_fx.k.index} val={start_fx.val:.1f} high={start_fx.high('fx_qj_k','fx_qy_three'):.1f}")

end_fx = None
for pos in range(fx347_pos + 1, len(fxs)):
    cur_fx = fxs[pos]
    if end_fx is None:
        if cur_fx.type == start_fx.type:
            # Same type - check if more extreme
            if start_fx.type == "di" and cur_fx.val < start_fx.val:
                print(f"  [{pos}] di@{cur_fx.k.index}: val={cur_fx.val:.1f} < start.val={start_fx.val:.1f} → would update start (but can't after bi created)")
        else:
            # Opposite type - try to set end_fx
            cl_gap = cur_fx.k.index - start_fx.k.index
            k_gap = cur_fx.k.k_index - start_fx.k.k_index
            valid_gap = cl_gap >= 4 and k_gap >= 4
            if valid_gap:
                sh = start_fx.high('fx_qj_k','fx_qy_three')
                eh = cur_fx.high('fx_qj_k','fx_qy_three')
                strict_block = sh > eh
                print(f"  [{pos}] ding@{cur_fx.k.index}: cl_gap={cl_gap}, k_gap={k_gap}, strict: {sh:.1f}>{eh:.1f}={strict_block}", end="")
                if strict_block and k_gap < 13:
                    print(" → BLOCKED by strict")
                else:
                    print(" → PASS (strict_skip or pass)")
                    if end_fx is None:
                        print(f"    → SET end_fx = ding@{cur_fx.k.index}")
                        end_fx = cur_fx
                        end_idx = pos
            else:
                print(f"  [{pos}] ding@{cur_fx.k.index}: cl_gap={cl_gap}, k_gap={k_gap} → gap FAILS")
    else:
        if cur_fx.type == end_fx.type:
            # Extension
            if cur_fx.val >= end_fx.val:
                print(f"  [{pos}] ding@{cur_fx.k.index}: val={cur_fx.val:.1f} >= end.val={end_fx.val:.1f} → could extend")
        else:
            # Confirmation
            cl_gap = cur_fx.k.index - end_fx.k.index
            k_gap = cur_fx.k.k_index - end_fx.k.k_index
            print(f"  [{pos}] di@{cur_fx.k.index}: confirm cl_gap={cl_gap}, k_gap={k_gap}", end="")
            if cl_gap < 4 or k_gap < 4:
                print(" → gap FAILS")
            else:
                # Check strict for DOWN confirmation
                sl = end_fx.low('fx_qj_k','fx_qy_three')
                el = cur_fx.low('fx_qj_k','fx_qy_three')
                eh2 = cur_fx.high('fx_qj_k','fx_qy_three')
                sh2 = end_fx.high('fx_qj_k','fx_qy_three')
                strict1 = sl < el  # start.low < end.low
                strict2 = eh2 > sh2  # end.high > start.high
                if k_gap < 13:
                    print(f" strict1: {sl:.1f}<{el:.1f}={strict1}, strict2: {eh2:.1f}>{sh2:.1f}={strict2}", end="")
                    if strict1 or strict2:
                        print(" → BLOCKED")
                    else:
                        # CGD check
                        cgd_block = False
                        for j in range(end_idx + 1, pos):
                            mid_fx = fxs[j]
                            if mid_fx.type == cur_fx.type:
                                if cur_fx.type == "di" and mid_fx.val < cur_fx.val:
                                    print(f" CGD blocks (di@{mid_fx.k.index} val={mid_fx.val:.1f} < {cur_fx.val:.1f})")
                                    cgd_block = True
                                    break
                        if not cgd_block:
                            print(" → CONFIRMED!")
                else:
                    print(f" (strict_skip since k_gap={k_gap}>=13)", end="")
                    # CGD check
                    cgd_block = False
                    for j in range(end_idx + 1, pos):
                        mid_fx = fxs[j]
                        if mid_fx.type == cur_fx.type:
                            if cur_fx.type == "di" and mid_fx.val < cur_fx.val:
                                print(f" CGD blocks (di@{mid_fx.k.index} val={mid_fx.val:.1f} < {cur_fx.val:.1f})")
                                cgd_block = True
                                break
                    if not cgd_block:
                        print(" → CONFIRMED!")

if end_fx is not None:
    print(f"\n→ Last end_fx: ding@{end_fx.k.index}")
else:
    print(f"\n→ No valid end_fx found! Algorithm produces no stroke starting from di@347")
