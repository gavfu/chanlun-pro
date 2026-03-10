"""Correctly trace BTC60 from ding@69 using proper FX list positions."""
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

df = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cd_o = CL_O("test", "test", config=CL_CONFIG)
cd_o.process_klines(df)

fxs = cd_o.get_fxs()
qj = cd_o.fx_qj; qy = cd_o.fx_qy

# Build a map from k_index to fxs list position
print("=== FXes around k=69-120 (by k_index) ===")
fx_pos = {}
for pos, fx in enumerate(fxs):
    fx_pos[fx.k.k_index] = pos
    if 60 <= fx.k.k_index <= 120:
        print(f"  fxs[{pos:3d}]: {fx.type:4s} k={fx.k.k_index:4d} (ck_idx={fx.k.index:3d}) "
              f"val={fx.val:.2f}")

# Now trace k_gap variant from ding@69
print(f"\n=== k_gap trace from ding@69 ===")
ding69_pos = fx_pos[69]
start_fx = fxs[ding69_pos]
end_fx = None
print(f"  Start: fxs[{ding69_pos}] = ding@{start_fx.k.k_index}")

i = ding69_pos + 1
while i < len(fxs) and i < ding69_pos + 30:
    cur = fxs[i]
    if end_fx is None:
        if cur.type == start_fx.type:
            print(f"  fxs[{i}]: ding@{cur.k.k_index} same type → skip")
        else:
            k_gap = cur.k.k_index - start_fx.k.k_index
            # strict from ding→di (down BI primary)
            h_s = start_fx.high(qj, qy); l_s = start_fx.low(qj, qy)
            h_e = cur.high(qj, qy); l_e = cur.low(qj, qy)
            c1 = l_s < l_e  # start.low < end.low (bad for down BI)
            c2 = h_e > h_s  # end.high > start.high (bad for down BI)
            strict_fail = (c1 or c2) if k_gap < 13 else False
            gap_ok = k_gap >= 4
            print(f"  fxs[{i}]: di@{cur.k.k_index} k_gap={k_gap} "
                  f"gap={'OK' if gap_ok else 'FAIL'} "
                  f"strict={'FAIL' if strict_fail else 'PASS'} "
                  f"(C1={c1} C2={c2})"
                  f"{' → SET end_fx' if gap_ok and not strict_fail else ''}")
            if gap_ok and not strict_fail:
                end_fx = cur
        i += 1
    else:
        if cur.type == end_fx.type:  # same type → extension
            k_gap_ext = cur.k.k_index - start_fx.k.k_index
            # check valid for extension
            h_s = start_fx.high(qj, qy); l_s = start_fx.low(qj, qy)
            h_e = cur.high(qj, qy); l_e = cur.low(qj, qy)
            c1 = l_s < l_e; c2 = h_e > h_s
            strict_fail = (c1 or c2) if k_gap_ext < 13 else False
            valid = k_gap_ext >= 4 and not strict_fail
            lower = cur.val <= end_fx.val
            if valid and lower:
                print(f"  fxs[{i}]: di@{cur.k.k_index} extend val={cur.val:.2f} "
                      f"(was di@{end_fx.k.k_index} val={end_fx.val:.2f}) → EXTEND")
                end_fx = cur
            else:
                reason = []
                if not valid: reason.append("invalid")
                if not lower: reason.append("not lower")
                print(f"  fxs[{i}]: di@{cur.k.k_index} extend → NO ({', '.join(reason)})")
            i += 1
        else:
            # confirmation
            k_gap_c = cur.k.k_index - end_fx.k.k_index
            cl_gap_c = cur.k.index - end_fx.k.index
            # di→ding confirmation: UP BI
            h_s = end_fx.high(qj, qy); l_s = end_fx.low(qj, qy)
            h_e = cur.high(qj, qy); l_e = cur.low(qj, qy)
            c1 = h_s > h_e; c2 = l_e < l_s
            strict_fail = (c1 or c2) if k_gap_c < 13 else False
            gap_ok = k_gap_c >= 4
            if gap_ok and not strict_fail:
                print(f"  fxs[{i}]: ding@{cur.k.k_index} confirm k_gap={k_gap_c} "
                      f"cl_gap={cl_gap_c} → CONFIRMED!")
                print(f"  → BI: down {start_fx.k.k_index}→{end_fx.k.k_index}")
                break
            else:
                reason = "GAP FAIL" if not gap_ok else f"STRICT FAIL(C1={c1},C2={c2})"
                print(f"  fxs[{i}]: ding@{cur.k.k_index} confirm k_gap={k_gap_c} "
                      f"cl_gap={cl_gap_c} → {reason}")
            i += 1

# Also do baseline (cl_gap) trace:            
print(f"\n=== cl_gap (baseline) trace from ding@69 ===")
end_fx = None
i = ding69_pos + 1
while i < len(fxs) and i < ding69_pos + 30:
    cur = fxs[i]
    if end_fx is None:
        if cur.type == start_fx.type:
            print(f"  fxs[{i}]: ding@{cur.k.k_index} same type → skip")
        else:
            cl_gap = cur.k.index - start_fx.k.index
            k_gap = cur.k.k_index - start_fx.k.k_index
            h_s = start_fx.high(qj, qy); l_s = start_fx.low(qj, qy)
            h_e = cur.high(qj, qy); l_e = cur.low(qj, qy)
            c1 = l_s < l_e; c2 = h_e > h_s
            strict_fail = (c1 or c2) if k_gap < 13 else False
            gap_ok = cl_gap >= 4
            print(f"  fxs[{i}]: di@{cur.k.k_index} cl_gap={cl_gap} k_gap={k_gap} "
                  f"gap={'OK' if gap_ok else 'FAIL'} "
                  f"strict={'FAIL' if strict_fail else 'PASS'}"
                  f"{' → SET end_fx' if gap_ok and not strict_fail else ''}")
            if gap_ok and not strict_fail:
                end_fx = cur
        i += 1
    else:
        if cur.type == end_fx.type:
            cl_gap_ext = cur.k.index - start_fx.k.index
            k_gap_ext = cur.k.k_index - start_fx.k.k_index
            h_s = start_fx.high(qj, qy); l_s = start_fx.low(qj, qy)
            h_e = cur.high(qj, qy); l_e = cur.low(qj, qy)
            c1 = l_s < l_e; c2 = h_e > h_s
            strict_fail = (c1 or c2) if k_gap_ext < 13 else False
            valid = cl_gap_ext >= 4 and not strict_fail
            lower = cur.val <= end_fx.val
            if valid and lower:
                print(f"  fxs[{i}]: di@{cur.k.k_index} extend val={cur.val:.2f} "
                      f"(was di@{end_fx.k.k_index} val={end_fx.val:.2f}) → EXTEND")
                end_fx = cur
            else:
                reason = []
                if not valid: reason.append(f"invalid(cl={cl_gap_ext})")
                if not lower: reason.append("not lower")
                print(f"  fxs[{i}]: di@{cur.k.k_index} extend → NO ({', '.join(reason)})")
            i += 1
        else:
            cl_gap_c = cur.k.index - end_fx.k.index
            k_gap_c = cur.k.k_index - end_fx.k.k_index
            h_s = end_fx.high(qj, qy); l_s = end_fx.low(qj, qy)
            h_e = cur.high(qj, qy); l_e = cur.low(qj, qy)
            c1 = h_s > h_e; c2 = l_e < l_s
            strict_fail = (c1 or c2) if k_gap_c < 13 else False
            gap_ok = cl_gap_c >= 4
            if gap_ok and not strict_fail:
                print(f"  fxs[{i}]: ding@{cur.k.k_index} confirm cl_gap={cl_gap_c} "
                      f"k_gap={k_gap_c} → CONFIRMED!")
                print(f"  → BI: down {start_fx.k.k_index}→{end_fx.k.k_index}")
                break
            else:
                reason = "CL_GAP FAIL" if not gap_ok else f"STRICT FAIL(C1={c1},C2={c2})"
                print(f"  fxs[{i}]: ding@{cur.k.k_index} confirm cl_gap={cl_gap_c} "
                      f"k_gap={k_gap_c} → {reason}")
            i += 1
