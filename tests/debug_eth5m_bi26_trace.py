"""Detailed trace of _build_bis for ETH5m around the first divergence point (bi[26]).
Goal: understand exactly why pyarmor creates bi k=333→337 but open creates bi k=333→349."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

CL_CONFIG = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
}

df = pd.read_parquet("tests/test_data/ETH_USDT_5m_1000.parquet")
cd_o = CL_O("test", "test", config=CL_CONFIG)
cd_o.process_klines(df)
cd_p = CL_P("test", "test", config=CL_CONFIG)
cd_p.process_klines(df)

fxs_o = cd_o.get_fxs()
bis_o = cd_o.get_bis()
bis_p = cd_p.get_bis()

# Pyarmor bi[26] = down k=333→337, bi[27] = up k=337→362
# Open    bi[26] = down k=333→349, bi[27] = up k=349→362
# 
# bi[25] end = ding@333 in both (start of the down BI)
# Open thinks end is di@349, pyarmor thinks end is di@337
#
# _build_bis logic for bi[26]:
#   start_fx = ding@333 (top), looking for di end_fx
#   Scans FXes after ding@333...

# Show FXes from k_index 333 to 365
print("=== FXes from k=333 to k=365 ===")
fx_list = []
for idx, fx in enumerate(fxs_o):
    if 333 <= fx.k.k_index <= 365:
        fx_list.append((idx, fx))
        print(f"  fxs[{idx}] {fx.type:>4} k_index={fx.k.k_index} val={fx.val:.2f} k.index={fx.k.index}")

# Find start_fx = ding@333
start_fx = None
start_idx = None
for idx, fx in enumerate(fxs_o):
    if fx.type == "ding" and fx.k.k_index == 333:
        start_fx = fx
        start_idx = idx
        break

qj = cd_o.fx_qj
qy = cd_o.fx_qy

print(f"\nstart_fx: ding@{start_fx.k.k_index} val={start_fx.val:.2f}")
print(f"  high={start_fx.high(qj,qy):.2f} low={start_fx.low(qj,qy):.2f}")

# Simulate _build_bis scan from start_fx
print("\n=== Simulating _build_bis scan ===")
end_fx = None
end_idx = -1

for i in range(start_idx + 1, len(fxs_o)):
    cur_fx = fxs_o[i]
    if cur_fx.k.k_index > 365:
        break
    
    cl_gap_from_start = cur_fx.k.index - start_fx.k.index
    k_gap_from_start = cur_fx.k.k_index - start_fx.k.k_index
    
    if end_fx is None:
        # Looking for first valid end_fx (di type)
        if cur_fx.type != start_fx.type:  # di
            valid = cd_o._bi_fx_valid(start_fx, cur_fx)
            print(f"  Try end_fx: di@{cur_fx.k.k_index} val={cur_fx.val:.2f} cl_gap={cl_gap_from_start} k_gap={k_gap_from_start}")
            print(f"    high={cur_fx.high(qj,qy):.2f} low={cur_fx.low(qj,qy):.2f}")
            print(f"    _bi_fx_valid(ding@333, di@{cur_fx.k.k_index}) = {valid}")
            if valid:
                end_fx = cur_fx
                end_idx = i
                print(f"    → SET end_fx = di@{cur_fx.k.k_index}")
        else:
            print(f"  Skip same-type: ding@{cur_fx.k.k_index} val={cur_fx.val:.2f}")
    else:
        # Already have end_fx
        if cur_fx.type == end_fx.type:
            # Same type as end_fx (di)
            if cur_fx.val <= end_fx.val:
                valid = cd_o._bi_fx_valid(start_fx, cur_fx)
                print(f"  Better di: di@{cur_fx.k.k_index} val={cur_fx.val:.2f} (vs end_fx val={end_fx.val:.2f})")
                print(f"    _bi_fx_valid(ding@333, di@{cur_fx.k.k_index}) = {valid}")
                if valid:
                    print(f"    → REPLACE end_fx = di@{cur_fx.k.k_index}")
                    end_fx = cur_fx
                    end_idx = i
                else:
                    print(f"    → Not valid, keep end_fx = di@{end_fx.k.k_index}")
            else:
                print(f"  Worse di: di@{cur_fx.k.k_index} val={cur_fx.val:.2f} > end_fx val={end_fx.val:.2f} → skip")
        else:
            # Opposite type (ding) - check confirmation
            confirm = cd_o._bi_fx_valid(end_fx, cur_fx)
            cl_gap_confirm = cur_fx.k.index - end_fx.k.index
            k_gap_confirm = cur_fx.k.k_index - end_fx.k.k_index
            print(f"  Confirm check: ding@{cur_fx.k.k_index} val={cur_fx.val:.2f}")
            print(f"    _bi_fx_valid(di@{end_fx.k.k_index}, ding@{cur_fx.k.k_index}) = {confirm}")
            print(f"    cl_gap={cl_gap_confirm} k_gap={k_gap_confirm}")
            if confirm:
                print(f"    high={cur_fx.high(qj,qy):.2f} low={cur_fx.low(qj,qy):.2f}")
                print(f"    end_fx.high={end_fx.high(qj,qy):.2f} end_fx.low={end_fx.low(qj,qy):.2f}")
                # Check strict details
                if k_gap_confirm < cd_o.fx_check_k_nums:
                    # Up BI from end_fx(di) to cur_fx(ding)
                    print(f"    STRICT: end_fx.high({end_fx.high(qj,qy):.2f}) > cur.high({cur_fx.high(qj,qy):.2f})? = {end_fx.high(qj,qy) > cur_fx.high(qj,qy)}")
                    print(f"    STRICT: cur.low({cur_fx.low(qj,qy):.2f}) < end_fx.low({end_fx.low(qj,qy):.2f})? = {cur_fx.low(qj,qy) < end_fx.low(qj,qy)}")
                print(f"    → CONFIRMED! BI: ding@333 → di@{end_fx.k.k_index}")
                break
            else:
                print(f"    → NOT confirmed, continue scanning")
                # Detail why it failed
                if cl_gap_confirm < 4:
                    print(f"    Failed: cl_gap ({cl_gap_confirm}) < 4")
                elif k_gap_confirm < cd_o.fx_check_k_nums:
                    print(f"    In strict check range (k_gap={k_gap_confirm} < {cd_o.fx_check_k_nums})")
                    print(f"      end_fx.high={end_fx.high(qj,qy):.2f} cur.high={cur_fx.high(qj,qy):.2f}")
                    print(f"      end_fx.low={end_fx.low(qj,qy):.2f} cur.low={cur_fx.low(qj,qy):.2f}")
                    if end_fx.high(qj,qy) > cur_fx.high(qj,qy):
                        print(f"      → FAILS: end_fx(di).high > cur(ding).high")
                    if cur_fx.low(qj,qy) < end_fx.low(qj,qy):
                        print(f"      → FAILS: cur(ding).low < end_fx(di).low")

print(f"\n=== RESULT ===")
print(f"Open bi[26]: ding@333 → di@{bis_o[26].end.k.k_index}")
print(f"Pyarmor bi[26]: ding@333 → di@{bis_p[26].end.k.k_index}")
