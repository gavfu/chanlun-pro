"""Deep investigation of BTC60 bi[15]: why does pyarmor accept cl_gap=1?

Hypothesis: Maybe pyarmor uses a different FX interval check that makes strict PASS
for di@299→ding@304, despite cl_gap=1.

Or maybe pyarmor doesn't use cl_gap at all, but uses k_gap with a lower threshold.
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

# Find the FXes involved in BTC60's first divergence
# Pyarmor bi[15] = up 299→304 (cl_gap=1, k_gap=5)
# Open bi[15] = up 299→308
print("=== BTC60 pyarmor bis around divergence ===")
for j in range(13, 20):
    bp = bis_p[j] if j < len(bis_p) else None
    bo = bis_o[j] if j < len(bis_o) else None
    if bp:
        print(f"  pya[{j}]: {bp.type:4s} {bp.start.k.k_index:4d}→{bp.end.k.k_index:4d} "
              f"cl={bp.end.k.index - bp.start.k.index}")
    if bo:
        print(f"  open[{j}]: {bo.type:4s} {bo.start.k.k_index:4d}→{bo.end.k.k_index:4d} "
              f"cl={bo.end.k.index - bo.start.k.index}")
    print()

# Find di@299 and ding@304 FXes
di_299 = None
ding_304 = None
ding_308 = None
for fx in fxs:
    if fx.k.k_index == 299 and fx.type == "di":
        di_299 = fx
    if fx.k.k_index == 304 and fx.type == "ding":
        ding_304 = fx
    if fx.k.k_index == 308 and fx.type == "ding":
        ding_308 = fx

if di_299:
    print(f"=== di@299 ===")
    print(f"  val={di_299.val}")
    print(f"  k.index={di_299.k.index}, k.k_index={di_299.k.k_index}")
    for i, ck in enumerate(di_299.klines):
        print(f"  klines[{i}]: ck_index={ck.index}, k_index={ck.k_index}, "
              f"h={ck.h}, l={ck.l}")
        for j, rk in enumerate(ck.klines):
            print(f"    raw[{j}]: index={rk.index}, h={rk.h}, l={rk.l}")
    qj = cd_o.fx_qj; qy = cd_o.fx_qy
    print(f"  high(qj_k, three) = {di_299.high(qj, qy)}")
    print(f"  low(qj_k, three)  = {di_299.low(qj, qy)}")

if ding_304:
    print(f"\n=== ding@304 ===")
    print(f"  val={ding_304.val}")
    print(f"  k.index={ding_304.k.index}, k.k_index={ding_304.k.k_index}")
    for i, ck in enumerate(ding_304.klines):
        print(f"  klines[{i}]: ck_index={ck.index}, k_index={ck.k_index}, "
              f"h={ck.h}, l={ck.l}")
        for j, rk in enumerate(ck.klines):
            print(f"    raw[{j}]: index={rk.index}, h={rk.h}, l={rk.l}")
    print(f"  high(qj_k, three) = {ding_304.high(qj, qy)}")
    print(f"  low(qj_k, three)  = {ding_304.low(qj, qy)}")

# Strict check details for di@299→ding@304
if di_299 and ding_304:
    print(f"\n=== Strict check: di@299 → ding@304 ===")
    print(f"  di@299 → ding@304  (UP bi)")
    print(f"  start=di, end=ding")
    print(f"  C1: start_fx.high(qj,qy) > end_fx.high(qj,qy)?")
    print(f"      {di_299.high(qj, qy)} > {ding_304.high(qj, qy)} = {di_299.high(qj, qy) > ding_304.high(qj, qy)}")
    print(f"  C2: end_fx.low(qj,qy) < start_fx.low(qj,qy)?")
    print(f"      {ding_304.low(qj, qy)} < {di_299.low(qj, qy)} = {ding_304.low(qj, qy) < di_299.low(qj, qy)}")
    # Check with only center and right klines (klines[1:])
    import chanlun.cl_interface as cli
    print(f"\n  -- With klines[1:] for start_fx --")
    # compute high of di@299 using klines[1:] only
    di299_h_right = max([rk.h for ck in di_299.klines[1:] for rk in ck.klines])
    di299_l_right = min([rk.l for ck in di_299.klines[1:] for rk in ck.klines])
    print(f"  di@299 high(klines[1:]) = {di299_h_right}")
    print(f"  di@299 low(klines[1:])  = {di299_l_right}")
    print(f"  C1_right: {di299_h_right} > {ding_304.high(qj, qy)} = {di299_h_right > ding_304.high(qj, qy)}")
    print(f"  C2_right: {ding_304.low(qj, qy)} < {di299_l_right} = {ding_304.low(qj, qy) < di299_l_right}")

# What about using fx_qj_ck (merged K-line values)?
if di_299 and ding_304:
    print(f"\n=== With fx_qj_ck (merged K-line max/min) ===")
    from chanlun.cl_interface import FX_QJ_CK, FX_QY_THREE
    print(f"  di@299 high(ck, three) = {di_299.high(FX_QJ_CK, FX_QY_THREE)}")
    print(f"  di@299 low(ck, three)  = {di_299.low(FX_QJ_CK, FX_QY_THREE)}")
    print(f"  ding@304 high(ck, three) = {ding_304.high(FX_QJ_CK, FX_QY_THREE)}")
    print(f"  ding@304 low(ck, three)  = {ding_304.low(FX_QJ_CK, FX_QY_THREE)}")
    print(f"  C1_ck: di299.h > ding304.h = "
          f"{di_299.high(FX_QJ_CK, FX_QY_THREE) > ding_304.high(FX_QJ_CK, FX_QY_THREE)}")
    print(f"  C2_ck: ding304.l < di299.l = "
          f"{ding_304.low(FX_QJ_CK, FX_QY_THREE) < di_299.low(FX_QJ_CK, FX_QY_THREE)}")

# Also: what if strict is checked with center-kline only (klines[1])?
if di_299 and ding_304:
    print(f"\n=== With center kline only ===")
    di_center = di_299.klines[1]
    ding_center = ding_304.klines[1]
    print(f"  di@299 center: h={di_center.h}, l={di_center.l}")
    print(f"  ding@304 center: h={ding_center.h}, l={ding_center.l}")
    print(f"  C1_center: {di_center.h} > {ding_center.h} = {di_center.h > ding_center.h}")
    print(f"  C2_center: {ding_center.l} < {di_center.l} = {ding_center.l < di_center.l}")
    # CK-level max high / min low
    di_max_h = max(ck.h for ck in di_299.klines)
    ding_max_h = max(ck.h for ck in ding_304.klines)
    di_min_l = min(ck.l for ck in di_299.klines)
    ding_min_l = min(ck.l for ck in ding_304.klines)
    print(f"\n  CK-level (using merged K-line h/l, NOT raw):")
    print(f"  di@299 max_ck_h={di_max_h}, ding@304 max_ck_h={ding_max_h}")
    print(f"  C1_ck_level: {di_max_h} > {ding_max_h} = {di_max_h > ding_max_h}")

# Check what BIs pyarmor creates after ding@304 — is there a confirmation?
print(f"\n=== Pyarmor BIs from 304 onward ===")
for j in range(len(bis_p)):
    bp = bis_p[j]
    if bp.start.k.k_index >= 290 and bp.start.k.k_index <= 320:
        print(f"  pya[{j}]: {bp.type:4s} {bp.start.k.k_index:4d}→{bp.end.k.k_index:4d} "
              f"cl={bp.end.k.index - bp.start.k.index}, k={bp.end.k.k_index - bp.start.k.k_index}")

# Check whether pyarmor's confirmation from ding@304 passes open's strict check
print(f"\n=== Confirmation check from ding@304 ===")
pya_confirm_end = None
for j in range(len(bis_p)):
    bp = bis_p[j]
    if bp.start.k.k_index == 304:
        pya_confirm_end = bp.end.k.k_index
        break

if pya_confirm_end:
    print(f"  pyarmor bi from 304 ends at {pya_confirm_end}")
    confirm_fx = None
    for fx in fxs:
        if fx.k.k_index == pya_confirm_end:
            confirm_fx = fx
            break
    if confirm_fx and ding_304:
        print(f"  confirm_fx type={confirm_fx.type}, val={confirm_fx.val}")
        cl_gap = confirm_fx.k.index - ding_304.k.index
        k_gap = confirm_fx.k.k_index - ding_304.k.k_index
        print(f"  cl_gap={cl_gap}, k_gap={k_gap}")
        
        # Standard strict
        h_start = ding_304.high(qj, qy)
        l_start = ding_304.low(qj, qy)
        h_end = confirm_fx.high(qj, qy)
        l_end = confirm_fx.low(qj, qy)
        print(f"  ding@304.high={h_start}, low={l_start}")
        print(f"  confirm.high={h_end}, low={l_end}")
        # Down BI from ding@304: start_fx=ding, end_fx=di
        # C1: start_fx.low < end_fx.low
        # C2: end_fx.high > start_fx.high
        print(f"  DOWN BI strict check:")
        print(f"  C1: start.low < end.low? {l_start} < {l_end} = {l_start < l_end}")
        print(f"  C2: end.high > start.high? {h_end} > {h_start} = {h_end > h_start}")
