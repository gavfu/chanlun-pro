"""Trace exact _build_bis execution for BTC60 around bi[15].
Instead of modifying cl_open.py, replicate the algorithm with debug prints."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P
from chanlun.cl_interface import FX, BI

CL_CONFIG = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
}

df = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cd_o = CL_O("test", "test", config=CL_CONFIG)
cd_o.process_klines(df)

fxs = cd_o.get_fxs()
qj = cd_o.fx_qj; qy = cd_o.fx_qy

# Find di@299 position in fxs
di299_pos = None
for pos, fx in enumerate(fxs):
    if fx.k.k_index == 299 and fx.type == "di":
        di299_pos = pos
        break

print(f"di@299 is at fxs[{di299_pos}]")
print(f"  klines: {[(ck.index, ck.k_index) for ck in fxs[di299_pos].klines]}")

# Now trace _build_bis starting from di@299
start_fx = fxs[di299_pos]
end_fx = None
end_idx = -1

i = di299_pos + 1
bi_count = 0
while i < len(fxs) and bi_count < 5:
    cur_fx = fxs[i]
    
    if end_fx is None:
        if cur_fx.type == start_fx.type:
            # Same type — in confirmed mode, don't update start
            print(f"  [{i}] {cur_fx.type}@{cur_fx.k.k_index} same type → skip")
        else:
            # Check _bi_fx_valid
            cl = cur_fx.k.index - start_fx.k.index
            k = cur_fx.k.k_index - start_fx.k.k_index
            valid = cd_o._bi_fx_valid(start_fx, cur_fx)
            print(f"  [{i}] {cur_fx.type}@{cur_fx.k.k_index} cl={cl} k={k} valid={valid}")
            if valid:
                end_fx = cur_fx
                end_idx = i
                print(f"       → SET end_fx = {cur_fx.type}@{cur_fx.k.k_index}")
        i += 1
    else:
        if cur_fx.type == end_fx.type:
            # Same type as end_fx → extension
            more_extreme = (end_fx.type == "di" and cur_fx.val <= end_fx.val) or \
                           (end_fx.type == "ding" and cur_fx.val >= end_fx.val)
            valid = cd_o._bi_fx_valid(start_fx, cur_fx) if more_extreme else False
            print(f"  [{i}] {cur_fx.type}@{cur_fx.k.k_index} val={cur_fx.val:.2f} "
                  f"extend={'YES' if more_extreme and valid else 'NO'} "
                  f"(extreme={more_extreme}, valid={valid})")
            if more_extreme and valid:
                end_fx = cur_fx
                end_idx = i
                print(f"       → EXTEND end_fx = {cur_fx.type}@{cur_fx.k.k_index}")
            i += 1
        else:
            # Confirmation check
            confirm = cd_o._bi_fx_valid(end_fx, cur_fx)
            cl_c = cur_fx.k.index - end_fx.k.index
            k_c = cur_fx.k.k_index - end_fx.k.k_index
            print(f"  [{i}] {cur_fx.type}@{cur_fx.k.k_index} confirm cl={cl_c} k={k_c} "
                  f"valid={confirm}")
            if confirm:
                bi_type = "down" if start_fx.type == "ding" else "up"
                print(f"  *** CONFIRMED: {bi_type} {start_fx.k.k_index}→{end_fx.k.k_index}")
                bi_count += 1
                start_fx = end_fx
                end_fx = None
                end_idx = -1
                i = i - (i - end_idx - 1)  # NO! Actually i = start_idx + 1
                # Let me fix this: we need start_idx
                # After confirmation, i should be end_idx + 1
                # Actually looking at code: start_fx = end_fx, i = start_idx + 1
                # But start_idx was set to end_idx
                # So i = end_idx + 1? No: i = start_idx + 1 where start_idx = end_idx
                # Oh wait, I need to track end_idx properly
                # Let me restart with proper tracking
                break
            else:
                i += 1

# Actually let me just use the real code with monkey-patching for debug
print(f"\n\n=== Using real code with debug ===")

# Store original method
orig_bi_fx_valid = cd_o._bi_fx_valid.__func__

call_count = [0]

def debug_bi_fx_valid(self, start_fx, end_fx):
    result = orig_bi_fx_valid(self, start_fx, end_fx)
    sk = start_fx.k.k_index
    ek = end_fx.k.k_index
    # Only log around the region of interest (k=290-340)
    if 290 <= sk <= 340 or 290 <= ek <= 340:
        cl = end_fx.k.index - start_fx.k.index
        k = ek - sk
        call_count[0] += 1
        print(f"  _bi_fx_valid({start_fx.type}@{sk}→{end_fx.type}@{ek}) "
              f"cl={cl} k={k} → {result}")
    return result

# Monkey-patch
import types
cd_o2 = CL_O("test", "test", config=CL_CONFIG)
cd_o2._bi_fx_valid = types.MethodType(debug_bi_fx_valid, cd_o2)
cd_o2.process_klines(df)

bis_o2 = cd_o2.get_bis()
# Show bi[14-17]
print(f"\nResulting BIs [14-17]:")
for j in range(14, min(18, len(bis_o2))):
    b = bis_o2[j]
    print(f"  [{j}] {b.type} {b.start.k.k_index}→{b.end.k.k_index}")
