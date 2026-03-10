# -*- coding: utf-8 -*-
"""
Diagnose why cl_open chooses BI#3 = up[47→50] instead of up[39→45]
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_interface import FX, Config

DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")

def load(symbol, freq, limit):
    cache_key = f"{symbol.replace('/', '_')}_{freq}_{limit}"
    return pd.read_parquet(os.path.join(DATA_DIR, f"{cache_key}.parquet"))


df = load("ETH/USDT", "60m", 1000)

cd = CL_Open("ETH/USDT", "60m", config={})
cd.process_klines(df)
fxs = cd.get_fxs()

# Find the relevant FXs
print("FXs from index 35 to 55:")
target_fxs = []
for idx, fx in enumerate(fxs):
    if 35 <= fx.k.index <= 55:
        print(f"  fxs[{idx}]: FX({fx.k.index},{fx.type}) val={fx.val:.2f} "
              f"k_index={fx.k.k_index} "
              f"high={fx.high(cd.fx_qj, cd.fx_qy):.2f} low={fx.low(cd.fx_qj, cd.fx_qy):.2f}")
        target_fxs.append((idx, fx))

# Find FX(39,di) and FX(45,ding)
fx39 = next(fx for fx in fxs if fx.k.index == 39)
fx45 = next(fx for fx in fxs if fx.k.index == 45)
fx41 = next(fx for fx in fxs if fx.k.index == 41)
fx43 = next(fx for fx in fxs if fx.k.index == 43)
fx47 = next(fx for fx in fxs if fx.k.index == 47)
fx50 = next(fx for fx in fxs if fx.k.index == 50)

print(f"\nbi_type = {cd.bi_type}")
print(f"fx_check_k_nums = {cd.fx_check_k_nums}")
print(f"allow_bi_fx_strict = {cd.allow_bi_fx_strict}")
print(f"bi_fx_cgd = {cd.bi_fx_cgd}")

# Test _bi_fx_valid for key pairs
pairs = [
    ("FX(39,di)", "FX(41,ding)", fx39, fx41),
    ("FX(39,di)", "FX(45,ding)", fx39, fx45),
    ("FX(39,di)", "FX(50,ding)", fx39, fx50),
    ("FX(43,di)", "FX(45,ding)", fx43, fx45),
    ("FX(47,di)", "FX(50,ding)", fx47, fx50),
]

print(f"\n_bi_fx_valid tests:")
for label_s, label_e, s, e in pairs:
    valid = cd._bi_fx_valid(s, e)
    cl_gap = e.k.index - s.k.index
    k_gap = e.k.k_index - s.k.k_index
    qj, qy = cd.fx_qj, cd.fx_qy
    s_h, s_l = s.high(qj, qy), s.low(qj, qy)
    e_h, e_l = e.high(qj, qy), e.low(qj, qy)
    print(f"  {label_s} → {label_e}: valid={valid}  cl_gap={cl_gap}  k_gap={k_gap}")
    print(f"    start: h={s_h:.2f} l={s_l:.2f}  end: h={e_h:.2f} l={e_l:.2f}")
    if s.type == "di" and e.type == "ding":
        # up bi: start.high should <= end.high, end.low should >= start.low
        print(f"    up bi strict checks: start.high({s_h:.2f}) > end.high({e_h:.2f})? {s_h > e_h}")
        print(f"    up bi strict checks: end.low({e_l:.2f}) < start.low({s_l:.2f})? {e_l < s_l}")

# Simulate what _build_bis does after BI#2 (down 35→39)
# start_fx = FX(39,di), scanning from fxs[idx_of_39+1]
fx39_idx = next(i for i, fx in enumerate(fxs) if fx.k.index == 39)
print(f"\n\nSimulating _build_bis after BI#2 (down 35→39):")
print(f"  start_fx = FX(39,di) at fxs[{fx39_idx}]")
print(f"  has_confirmed_bi = True, len(bis) = 3")

# Walk through the same logic
start_fx = fx39
start_idx = fx39_idx
end_fx = None
end_idx = -1

for i in range(fx39_idx + 1, min(fx39_idx + 15, len(fxs))):
    cur_fx = fxs[i]
    print(f"\n  i={i}: FX({cur_fx.k.index},{cur_fx.type}) val={cur_fx.val:.2f}")

    if end_fx is None:
        if cur_fx.type == start_fx.type:
            # Same type - check if "better"
            if start_fx.type == "di" and cur_fx.val < start_fx.val:
                print(f"    → REPLACE start_fx: FX({start_fx.k.index}) val={start_fx.val:.2f} → FX({cur_fx.k.index}) val={cur_fx.val:.2f} (lower di)")
                start_fx = cur_fx
                start_idx = i
            else:
                print(f"    → SKIP same type, not better (start val={start_fx.val:.2f})")
        else:
            # Opposite type - check can_set_endpoint
            # For old BI after confirmed: k_gap >= 4
            k_gap = cur_fx.k.k_index - start_fx.k.k_index
            full_valid = cd._bi_fx_valid(start_fx, cur_fx)
            relaxed = k_gap >= 4
            print(f"    → Opposite type. k_gap={k_gap}, _bi_fx_valid={full_valid}, k_gap>=4={relaxed}")
            # _can_set_endpoint for old BI after confirm: k_gap >= 4
            if relaxed:
                # Check adjacent FX restriction
                if i == start_idx + 1:
                    restrict = cd._bi_fx_valid(start_fx, cur_fx)
                    print(f"    → Adjacent restriction (i==start_idx+1): _bi_fx_valid={restrict}")
                    if restrict:
                        end_fx = cur_fx
                        end_idx = i
                        print(f"    → SET end_fx = FX({cur_fx.k.index},{cur_fx.type})")
                else:
                    end_fx = cur_fx
                    end_idx = i
                    print(f"    → SET end_fx = FX({cur_fx.k.index},{cur_fx.type})")
            else:
                print(f"    → Cannot set endpoint (k_gap too small)")
    else:
        if cur_fx.type == end_fx.type:
            if end_fx.type == "ding" and cur_fx.val >= end_fx.val:
                ext_valid = cd._bi_fx_valid(start_fx, cur_fx)
                print(f"    → Same as end, higher/equal ding. _bi_fx_valid(start,cur)={ext_valid}")
                if ext_valid:
                    end_fx = cur_fx
                    end_idx = i
                    print(f"    → EXTEND end_fx = FX({cur_fx.k.index},{cur_fx.type})")
        else:
            confirm = cd._bi_fx_valid(end_fx, cur_fx)
            print(f"    → Opposite to end. Confirm _bi_fx_valid(end,cur)={confirm}")
            if confirm:
                print(f"    → CONFIRM BI: {start_fx.type}→{end_fx.type} [{start_fx.k.index}→{end_fx.k.index}]")
                break
