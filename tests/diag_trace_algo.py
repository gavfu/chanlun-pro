"""Trace algorithm step by step for FX19 to FX26 in 500k dataset"""
import sys, pathlib, copy
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
import pandas as pd
from chanlun.cl_open import CL as CLOpen

df = pd.read_parquet(pathlib.Path(__file__).parent / "test_data" / "BTC_USDT_60m_500.parquet")
c = CLOpen("BTC/USDT", "60m", {})
c.process_klines(df)
fxs = c.fxs

print("FX19..FX26 properties:")
for i in range(19, 27):
    fx = fxs[i]
    print(f"  FX{i}: {fx.type} val={fx.val} cl={fx.k.index} k={fx.k.k_index}")

print()
print("Step-by-step trace after bi[4] confirmed (start=FX19(di)):")
print("  State: start=FX19, end=None")

start_fx_idx = 19
end_fx_idx = None

for i in range(20, 30):
    if i >= len(fxs):
        break
    cur = fxs[i]
    start = fxs[start_fx_idx]
    
    if end_fx_idx is None:
        if cur.type == start.type:
            # same type: check if more extreme
            if (cur.type == "di" and cur.val < start.val) or (cur.type == "ding" and cur.val > start.val):
                print(f"  FX{i}: same type as start, more extreme → update start to FX{i}")
                start_fx_idx = i
            else:
                print(f"  FX{i}: same type as start, NOT more extreme → skip")
        else:
            valid = c._bi_fx_valid(start, cur)
            if valid:
                end_fx_idx = i
                print(f"  FX{i}: opposite type, valid! → set end=FX{i}")
            else:
                print(f"  FX{i}: opposite type, NOT valid → skip")
    else:
        end = fxs[end_fx_idx]
        if cur.type == end.type:
            # same type as end
            if (cur.type == "di" and cur.val <= end.val) or (cur.type == "ding" and cur.val >= end.val):
                valid = c._bi_fx_valid(start, cur)
                if valid:
                    print(f"  FX{i}: same type as end, more extreme, valid → extend end to FX{i}")
                    end_fx_idx = i
                else:
                    print(f"  FX{i}: same type as end, more extreme, NOT valid → skip extension")
            else:
                print(f"  FX{i}: same type as end, NOT more extreme → no extension")
        else:
            # opposite type (same as start) → confirm check
            confirm = c._bi_fx_valid(end, cur)
            # bi_fx_cgd check
            if confirm:
                k_gap_confirm = cur.k.k_index - end.k.k_index
                if k_gap_confirm < c.fx_check_k_nums:
                    for j in range(end_fx_idx + 1, i):
                        mid = fxs[j]
                        if mid.type == cur.type:
                            if cur.type == "di" and mid.val < cur.val:
                                confirm = False
                                print(f"    cgd check: FX{j}.val({mid.val}) < FX{i}.val({cur.val}) → reject confirm")
                                break
                            elif cur.type == "ding" and mid.val > cur.val:
                                confirm = False
                                print(f"    cgd check: FX{j}.val({mid.val}) > FX{i}.val({cur.val}) → reject confirm")
                                break
            if confirm:
                print(f"  FX{i}: confirm FX{end_fx_idx}! → BI[start=FX{start_fx_idx},end=FX{end_fx_idx}] CONFIRMED")
                start_fx_idx = end_fx_idx
                end_fx_idx = None
            else:
                print(f"  FX{i}: opposite type, NOT confirm (valid={c._bi_fx_valid(end, cur)}) → skip")

print(f"\nFinal state: start=FX{start_fx_idx}, end={'FX'+str(end_fx_idx) if end_fx_idx else 'None'}")
