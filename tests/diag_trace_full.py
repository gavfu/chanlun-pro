"""Monkey-patch _build_bis to trace algorithm for FX19-FX26"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
import pandas as pd
from chanlun.cl_open import CL as CLOpen
from chanlun.cl_interface import Config, BI
from typing import List

df = pd.read_parquet(pathlib.Path(__file__).parent / "test_data" / "BTC_USDT_60m_500.parquet")
c = CLOpen("BTC/USDT", "60m", {})
# Process klines to get FXs first
c.process_klines(df)
fxs = c.fxs

# Manually trace the algorithm for FX10..FX28
print("Tracing _build_bis from FX0 to FX28:")
print(f"bi_type={c.bi_type}")
print()

# Partial FX list from FX0 to FX30 so that bi[5] region is visible
trace_fxs = fxs[:32] if len(fxs) >= 32 else fxs

bis = []
start_fx = trace_fxs[0]
start_idx = 0
end_fx = None
end_idx = -1

i = 1
while i < len(trace_fxs):
    cur_fx = trace_fxs[i]
    
    if end_fx is None:
        if cur_fx.type == start_fx.type:
            if (start_fx.type == "ding" and cur_fx.val > start_fx.val) or \
               (start_fx.type == "di" and cur_fx.val < start_fx.val):
                print(f"  i={i} FX{cur_fx.index}: same type, more extreme → update start FX{start_fx.index}→FX{cur_fx.index}")
                start_fx = cur_fx
                start_idx = i
            # else skip
        else:
            valid = c._bi_fx_valid(start_fx, cur_fx)
            cl_gap = cur_fx.k.index - start_fx.k.index
            k_gap = cur_fx.k.k_index - start_fx.k.k_index
            if valid:
                print(f"  i={i} FX{cur_fx.index}: opposite, valid (cl={cl_gap},k={k_gap}) → set end=FX{cur_fx.index}")
                end_fx = cur_fx
                end_idx = i
            else:
                if i >= 19:
                    print(f"  i={i} FX{cur_fx.index}: opposite, NOT valid (cl={cl_gap},k={k_gap}) → skip")
        i += 1
    else:
        if cur_fx.type == end_fx.type:
            if (end_fx.type == "di" and cur_fx.val <= end_fx.val) or \
               (end_fx.type == "ding" and cur_fx.val >= end_fx.val):
                valid = c._bi_fx_valid(start_fx, cur_fx)
                cl_gap = cur_fx.k.index - start_fx.k.index
                k_gap = cur_fx.k.k_index - start_fx.k.k_index
                if valid:
                    print(f"  i={i} FX{cur_fx.index}: same as end, more extreme, valid (cl={cl_gap},k={k_gap}) → extend end FX{end_fx.index}→FX{cur_fx.index}")
                    end_fx = cur_fx
                    end_idx = i
                else:
                    print(f"  i={i} FX{cur_fx.index}: same as end, more extreme, NOT valid (cl={cl_gap},k={k_gap}) → skip extension")
            else:
                pass  # not more extreme, ignore
            i += 1
        else:
            # confirm check
            confirm = c._bi_fx_valid(end_fx, cur_fx)
            cl_gap = cur_fx.k.index - end_fx.k.index
            k_gap = cur_fx.k.k_index - end_fx.k.k_index
            print(f"  i={i} FX{cur_fx.index}: confirm? _bi_fx_valid(end=FX{end_fx.index},cur=FX{cur_fx.index}) cl={cl_gap} k={k_gap} = {confirm}")
            
            # cgd check
            if confirm and c.bi_fx_cgd == Config.BI_FX_CHD_YES.value:
                k_gap_confirm = cur_fx.k.k_index - end_fx.k.k_index
                if k_gap_confirm < c.fx_check_k_nums:
                    for j in range(end_idx + 1, i):
                        mid_fx = trace_fxs[j]
                        if mid_fx.type == cur_fx.type:
                            if cur_fx.type == "di" and mid_fx.val < cur_fx.val:
                                confirm = False
                                print(f"    → cgd BLOCKED by FX{mid_fx.index}.val({mid_fx.val}) < cur.val({cur_fx.val})")
                                break
                            elif cur_fx.type == "ding" and mid_fx.val > cur_fx.val:
                                confirm = False
                                print(f"    → cgd BLOCKED by FX{mid_fx.index}.val({mid_fx.val}) > cur.val({cur_fx.val})")
                                break
            
            if confirm:
                print(f"  → CONFIRM: BI[FX{start_fx.index}→FX{end_fx.index}] added! New start=FX{end_fx.index}, reset end=None, restart i={end_idx+1}")
                bi_type = "down" if start_fx.type == "ding" else "up"
                bi = BI(start=start_fx, end=end_fx, _type=bi_type, index=len(bis), default_zs_type=c.default_bi_zs_type)
                bis.append(bi)
                start_fx = end_fx
                start_idx = end_idx
                end_fx = None
                end_idx = -1
                i = start_idx + 1
                print(f"  State: start=FX{start_fx.index}, end=None, i={i}")
            else:
                i += 1

print(f"\nFinal BIs traced:")
for b in bis:
    print(f"  bi[{b.index}]: FX{b.start.index}→FX{b.end.index}")
print(f"  Final state: start=FX{start_fx.index}, end={'FX'+str(end_fx.index) if end_fx else 'None'}")
