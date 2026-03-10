"""Trace _build_bis for BTC5m around bi[61] divergence.
Open: bi[61] up k=875→879, pyarmor: bi[61] up k=875→889"""
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

df = pd.read_parquet("tests/test_data/BTC_USDT_5m_1000.parquet")
cd_o = CL_O("test", "test", config=CL_CONFIG)
cd_o.process_klines(df)
cd_p = CL_P("test", "test", config=CL_CONFIG)
cd_p.process_klines(df)

bis_o = cd_o.get_bis()
bis_p = cd_p.get_bis()
fxs = cd_o.get_fxs()

qj = cd_o.fx_qj
qy = cd_o.fx_qy

print("=== Open BIs [59:67] ===")
for bi in bis_o[59:67]:
    cl_gap = bi.end.k.index - bi.start.k.index
    k_gap = bi.end.k.k_index - bi.start.k.k_index
    print(f"  bi[{bi.index}] {bi.type} k={bi.start.k.k_index}→{bi.end.k.k_index} cl_gap={cl_gap} k_gap={k_gap}")

print("\n=== Pyarmor BIs [59:67] ===")
for bi in bis_p[59:67]:
    cl_gap = bi.end.k.index - bi.start.k.index
    k_gap = bi.end.k.k_index - bi.start.k.k_index
    print(f"  bi[{bi.index}] {bi.type} k={bi.start.k.k_index}→{bi.end.k.k_index} cl_gap={cl_gap} k_gap={k_gap}")

# Open bi[60] ends at di@875. bi[61] = up k=875→879 (small BI)
# Pyarmor bi[60] ends at di@875. bi[61] = up k=875→889 (larger BI)
# Both start at di@875, but open finds ding@879 as end_fx while pyarmor extends to ding@889

print("\n=== FXes from k=875 to k=895 ===")
start_fx = None
for idx, fx in enumerate(fxs):
    if fx.type == "di" and fx.k.k_index == 875:
        start_fx = fx
    if 875 <= fx.k.k_index <= 895:
        print(f"  fxs[{idx}] {fx.type:>4} k_index={fx.k.k_index} val={fx.val:.2f} k.index={fx.k.index}")

print(f"\nstart_fx: di@{start_fx.k.k_index} val={start_fx.val:.2f}")

# Simulate _build_bis scan from start_fx = di@875
print("\n=== Simulating scan from di@875 ===")
end_fx = None
start_idx = None
for idx, fx in enumerate(fxs):
    if fx is start_fx:
        start_idx = idx
        break

for i in range(start_idx + 1, len(fxs)):
    cur_fx = fxs[i]
    if cur_fx.k.k_index > 895:
        break
    
    if end_fx is None:
        if cur_fx.type != start_fx.type:  # ding
            valid = cd_o._bi_fx_valid(start_fx, cur_fx)
            cl_gap = cur_fx.k.index - start_fx.k.index
            k_gap = cur_fx.k.k_index - start_fx.k.k_index
            print(f"  Try end_fx: ding@{cur_fx.k.k_index} val={cur_fx.val:.2f} cl_gap={cl_gap} k_gap={k_gap} valid={valid}")
            if valid:
                end_fx = cur_fx
                end_fx_idx = i
                print(f"    → SET end_fx")
    else:
        if cur_fx.type == end_fx.type:  # ding
            if cur_fx.val >= end_fx.val:
                valid = cd_o._bi_fx_valid(start_fx, cur_fx)
                print(f"  Higher ding: ding@{cur_fx.k.k_index} val={cur_fx.val:.2f} (vs {end_fx.val:.2f}) valid={valid}")
                if valid:
                    print(f"    → REPLACE end_fx")
                    end_fx = cur_fx
                    end_fx_idx = i
            else:
                print(f"  Lower ding: ding@{cur_fx.k.k_index} val={cur_fx.val:.2f} → skip")
        else:  # di — check confirmation
            confirm = cd_o._bi_fx_valid(end_fx, cur_fx)
            cl_gap = cur_fx.k.index - end_fx.k.index
            k_gap = cur_fx.k.k_index - end_fx.k.k_index
            print(f"  Confirm: {end_fx.type}@{end_fx.k.k_index}→{cur_fx.type}@{cur_fx.k.k_index} "
                  f"cl_gap={cl_gap} k_gap={k_gap} confirm={confirm}")
            if confirm:
                print(f"    → CONFIRMED! BI: di@875 → ding@{end_fx.k.k_index}")
                break
            else:
                # Show why rejection
                if cl_gap < 4:
                    print(f"    Failed: cl_gap ({cl_gap}) < 4")
                elif k_gap < cd_o.fx_check_k_nums:
                    print(f"    In strict range. end.high={end_fx.high(qj,qy):.2f} "
                          f"cur.high={cur_fx.high(qj,qy):.2f} "
                          f"end.low={end_fx.low(qj,qy):.2f} "
                          f"cur.low={cur_fx.low(qj,qy):.2f}")
                    # For down BI (ding→di):
                    c1 = end_fx.low(qj,qy) < cur_fx.low(qj,qy)
                    c2 = cur_fx.high(qj,qy) > end_fx.high(qj,qy)
                    if c1: print(f"    → FAILS C1: end_fx.low < cur.low")
                    if c2: print(f"    → FAILS C2: cur.high > end_fx.high")
