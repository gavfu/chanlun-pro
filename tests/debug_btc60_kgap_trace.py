"""Trace BTC60 to understand why k_gap creates extra BIs.
Also verify: what if we use k_gap >= 4 + NO strict at all (but still standard cl_gap)?
Or: k_gap >= 4 + right-half strict + check extension?
"""
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
cd_p = CL_P("test", "test", config=CL_CONFIG)
cd_p.process_klines(df)

fxs = cd_o.get_fxs()
bis_o = cd_o.get_bis()
bis_p = cd_p.get_bis()

print("=== BTC60: First 10 BIs compared ===")
for j in range(10):
    bo = bis_o[j] if j < len(bis_o) else None
    bp = bis_p[j] if j < len(bis_p) else None
    bo_s = f"{bo.type} {bo.start.k.k_index}→{bo.end.k.k_index} cl={bo.end.k.index-bo.start.k.index}" if bo else "---"
    bp_s = f"{bp.type} {bp.start.k.k_index}→{bp.end.k.k_index} cl={bp.end.k.index-bp.start.k.index}" if bp else "---"
    match = "✓" if (bo and bp and bo.start.k.k_index == bp.start.k.k_index and bo.end.k.k_index == bp.end.k.k_index) else "✗"
    print(f"  [{j}] {match} open={bo_s}  pya={bp_s}")

# Now trace how k_gap variant creates the extra BI at 69→95
# The k_gap variant would scan from ding@69 as start_fx.
# Let's trace the FXes after ding@69:
print(f"\n=== FXes from k=69 onward ===")
qj = cd_o.fx_qj; qy = cd_o.fx_qy
ding69 = None
for fx in fxs:
    if 69 <= fx.k.k_index <= 110:
        cl_from_69 = None
        k_from_69 = None
        if ding69 and fx.type == "di":
            cl_from_69 = fx.k.index - ding69.k.index
            k_from_69 = fx.k.k_index - ding69.k.k_index
        if fx.k.k_index == 69 and fx.type == "ding":
            ding69 = fx
        suffix = ""
        if cl_from_69 is not None:
            suffix = f" gap_from_69: cl={cl_from_69}, k={k_from_69}"
        print(f"  {fx.type:4s} k={fx.k.k_index} (idx={fx.k.index}) val={fx.val:.2f}{suffix}")

# The key question: why does the k_gap algorithm terminate the down BI at di@95 
# instead of extending to di@101?
# Let's check: after finding di@78 as candidate, would it extend to di@87, di@95, di@101?
# First, do di@87 and di@95 have lower vals?
print(f"\n=== Di FXes val comparison ===")
di_vals = {}
for fx in fxs:
    if fx.type == "di" and 70 <= fx.k.k_index <= 110:
        di_vals[fx.k.k_index] = fx.val
        print(f"  di@{fx.k.k_index}: val={fx.val:.2f}")

# Check: di@78=87650.10, di@87=87711.00, di@95=83338.30, di@101=81000.00
# So di extensions: 78→87 NO (87650<87711, not more extreme for di)
# 78→95 YES (87650>83338)
# 95→101 YES (83338>81000)
# 
# After extending to di@101, confirmation would need ding after 101.
# di@101→ding@105: cl_gap=?, k_gap=4
print(f"\n=== Checking di@101→ding@105 ===")
di101 = None
ding105 = None
for fx in fxs:
    if fx.k.k_index == 101 and fx.type == "di": di101 = fx
    if fx.k.k_index == 105 and fx.type == "ding": ding105 = fx

if di101 and ding105:
    cl = ding105.k.index - di101.k.index
    k = ding105.k.k_index - di101.k.k_index
    h_s = di101.high(qj, qy); l_s = di101.low(qj, qy)
    h_e = ding105.high(qj, qy); l_e = ding105.low(qj, qy)
    c1 = h_s > h_e
    c2 = l_e < l_s
    print(f"  cl_gap={cl}, k_gap={k}")
    print(f"  start.h={h_s:.2f}, l={l_s:.2f}")
    print(f"  end.h={h_e:.2f}, l={l_e:.2f}")
    print(f"  C1={c1}, C2={c2}, strict={'FAIL' if c1 or c2 else 'PASS'}")

# So the key question is: WHY does the k_gap variant stop extension at di@95 and not extend to di@101?
# The answer must be about the confirmation happening first!
# Let's trace step by step:
print(f"\n=== Step-by-step k_gap trace from ding@69 ===")
start_fx = ding69
end_fx = None
idx_map = {fx.k.k_index: (i, fx) for i, fx in enumerate(fxs)}

# Find start index in fxs
start_i = None
for i, fx in enumerate(fxs):
    if fx.k.k_index == 69 and fx.type == "ding":
        start_i = i
        break

if start_i:
    i = start_i + 1
    while i < len(fxs) and i < start_i + 30:
        cur_fx = fxs[i]
        if end_fx is None:
            if cur_fx.type == start_fx.type:
                print(f"  [{i}] same type {cur_fx.type} k={cur_fx.k.k_index} "
                      f"val={cur_fx.val:.2f} → skip/update start")
            else:
                k_gap = cur_fx.k.k_index - start_fx.k.k_index
                cl_gap = cur_fx.k.index - start_fx.k.index
                # check valid with k_gap >= 4
                valid = k_gap >= 4
                if valid and k_gap < 13:
                    h_s = start_fx.high(qj, qy); l_s = start_fx.low(qj, qy)
                    h_e = cur_fx.high(qj, qy); l_e = cur_fx.low(qj, qy)
                    c1 = l_s < l_e  # ding→di: start.low < end.low
                    c2 = h_e > h_s  # end.high > start.high
                    if c1 or c2:
                        valid = False
                        print(f"  [{i}] candidate di@{cur_fx.k.k_index} "
                              f"k_gap={k_gap} cl_gap={cl_gap} STRICT FAIL C1={c1} C2={c2}")
                    else:
                        print(f"  [{i}] candidate di@{cur_fx.k.k_index} "
                              f"k_gap={k_gap} cl_gap={cl_gap} → SET end_fx")
                        end_fx = cur_fx
                elif valid:
                    print(f"  [{i}] candidate di@{cur_fx.k.k_index} "
                          f"k_gap={k_gap} cl_gap={cl_gap} → k_gap>=13, skip strict → SET end_fx")
                    end_fx = cur_fx
                else:
                    print(f"  [{i}] candidate di@{cur_fx.k.k_index} "
                          f"k_gap={k_gap} cl_gap={cl_gap} → GAP FAIL")
            i += 1
        else:
            if cur_fx.type == end_fx.type:
                # Extension check — with k_gap valid check
                k_gap_ext = cur_fx.k.k_index - start_fx.k.k_index
                valid_ext = k_gap_ext >= 4
                if valid_ext and k_gap_ext < 13:
                    h_s = start_fx.high(qj, qy); l_s = start_fx.low(qj, qy)
                    h_e = cur_fx.high(qj, qy); l_e = cur_fx.low(qj, qy)
                    c1 = l_s < l_e
                    c2 = h_e > h_s
                    if c1 or c2: valid_ext = False
                
                if valid_ext and cur_fx.val <= end_fx.val:  # lower di
                    print(f"  [{i}] extend di@{cur_fx.k.k_index} val={cur_fx.val:.2f} "
                          f"(was {end_fx.k.k_index} val={end_fx.val:.2f}) → EXTEND")
                    end_fx = cur_fx
                else:
                    reason = "val not lower" if valid_ext else "fx_valid fail"
                    print(f"  [{i}] extend di@{cur_fx.k.k_index} val={cur_fx.val:.2f} "
                          f"→ NO ({reason})")
                i += 1
            else:
                # Confirmation check
                k_gap_conf = cur_fx.k.k_index - end_fx.k.k_index
                cl_gap_conf = cur_fx.k.index - end_fx.k.index
                valid_conf = k_gap_conf >= 4
                if valid_conf and k_gap_conf < 13:
                    h_s = end_fx.high(qj, qy); l_s = end_fx.low(qj, qy)
                    h_e = cur_fx.high(qj, qy); l_e = cur_fx.low(qj, qy)
                    # di→ding (up bi confirmation)
                    c1 = h_s > h_e
                    c2 = l_e < l_s
                    if c1 or c2:
                        valid_conf = False
                        print(f"  [{i}] confirm ding@{cur_fx.k.k_index} "
                              f"k_gap={k_gap_conf} cl_gap={cl_gap_conf} STRICT FAIL C1={c1} C2={c2}")
                    else:
                        print(f"  [{i}] confirm ding@{cur_fx.k.k_index} "
                              f"k_gap={k_gap_conf} cl_gap={cl_gap_conf} → CONFIRMED!")
                        print(f"  → BI: {start_fx.type} {start_fx.k.k_index}→{end_fx.k.k_index}")
                        break
                elif valid_conf:
                    print(f"  [{i}] confirm ding@{cur_fx.k.k_index} "
                          f"k_gap={k_gap_conf} → k_gap>=13, skip strict → CONFIRMED!")
                    print(f"  → BI: {start_fx.type} {start_fx.k.k_index}→{end_fx.k.k_index}")
                    break
                else:
                    print(f"  [{i}] confirm ding@{cur_fx.k.k_index} "
                          f"k_gap={k_gap_conf} cl_gap={cl_gap_conf} → GAP FAIL")
                i += 1
