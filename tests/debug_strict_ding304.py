"""Check ding@304→di@309 with right-half strict for BOTH start AND end FXes."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O

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

ding304 = None
di309 = None
for fx in fxs:
    if fx.k.k_index == 304 and fx.type == "ding": ding304 = fx
    if fx.k.k_index == 309 and fx.type == "di": di309 = fx

if di309:
    print(f"=== di@309 ===")
    print(f"  val={di309.val}")
    for i, ck in enumerate(di309.klines):
        print(f"  klines[{i}]: ck_idx={ck.index}, k_idx={ck.k_index}, h={ck.h}, l={ck.l}")
        for j, rk in enumerate(ck.klines):
            print(f"    raw[{j}]: idx={rk.index}, h={rk.h}, l={rk.l}")
    print(f"  high(full) = {di309.high(qj, qy)}")
    print(f"  low(full) = {di309.low(qj, qy)}")

if ding304 and di309:
    ding304_h_rh = max([rk.h for ck in ding304.klines[1:] for rk in ck.klines])
    ding304_l_rh = min([rk.l for ck in ding304.klines[1:] for rk in ck.klines])
    
    print(f"\n=== ding@304 → di@309 strict check ===")
    print(f"  Standard strict (all klines):")
    h_s = ding304.high(qj, qy); l_s = ding304.low(qj, qy)
    h_e = di309.high(qj, qy); l_e = di309.low(qj, qy)
    c1 = l_s < l_e; c2 = h_e > h_s
    print(f"    start.h={h_s:.2f}, l={l_s:.2f}, end.h={h_e:.2f}, l={l_e:.2f}")
    print(f"    C1(start.l<end.l): {l_s:.2f}<{l_e:.2f} = {c1}")
    print(f"    C2(end.h>start.h): {h_e:.2f}>{h_s:.2f} = {c2}")
    print(f"    → {'FAIL' if c1 or c2 else 'PASS'}")
    
    print(f"\n  Right-half start strict (klines[1:] for start):")
    c1r = ding304_l_rh < di309.low(qj, qy)
    c2r = di309.high(qj, qy) > ding304_h_rh
    print(f"    start.h_rh={ding304_h_rh:.2f}, l_rh={ding304_l_rh:.2f}")
    print(f"    C1={c1r}, C2={c2r} → {'FAIL' if c1r or c2r else 'PASS'}")
    
    # CK-level strict:
    print(f"\n  CK-level strict (merged klines h/l):")
    ding304_h_ck = max(ck.h for ck in ding304.klines)
    ding304_l_ck = min(ck.l for ck in ding304.klines)
    di309_h_ck = max(ck.h for ck in di309.klines)
    di309_l_ck = min(ck.l for ck in di309.klines)
    c1ck = ding304_l_ck < di309_l_ck
    c2ck = di309_h_ck > ding304_h_ck
    print(f"    start.h_ck={ding304_h_ck:.2f}, l_ck={ding304_l_ck:.2f}")
    print(f"    end.h_ck={di309_h_ck:.2f}, l_ck={di309_l_ck:.2f}")
    print(f"    C1={c1ck}, C2={c2ck} → {'FAIL' if c1ck or c2ck else 'PASS'}")
    
    # CK right-half:
    print(f"\n  CK right-half strict:")
    ding304_h_ck_rh = max(ck.h for ck in ding304.klines[1:])
    ding304_l_ck_rh = min(ck.l for ck in ding304.klines[1:])
    c1ckr = ding304_l_ck_rh < di309_l_ck
    c2ckr = di309_h_ck > ding304_h_ck_rh
    print(f"    start.h={ding304_h_ck_rh:.2f}, l={ding304_l_ck_rh:.2f}")
    print(f"    end.h={di309_h_ck:.2f}, l={di309_l_ck:.2f}")
    print(f"    C1={c1ckr}, C2={c2ckr} → {'FAIL' if c1ckr or c2ckr else 'PASS'}")

# KEY: di@299→ding@304 with non-shared klines
print(f"\n\n=== di@299→ding@304 (PRIMARY) ===")
di299 = None
for fx in fxs:
    if fx.k.k_index == 299 and fx.type == "di": di299 = fx; break

if di299 and ding304:
    shared_ck_ids = set(ck.index for ck in di299.klines) & set(ck.index for ck in ding304.klines)
    print(f"  Shared CK indices: {shared_ck_ids}")
    
    di299_ns = [ck for ck in di299.klines if ck.index not in shared_ck_ids]
    ding304_ns = [ck for ck in ding304.klines if ck.index not in shared_ck_ids]
    
    print(f"  di@299 non-shared klines: {[(ck.index, ck.k_index) for ck in di299_ns]}")
    print(f"  ding@304 non-shared klines: {[(ck.index, ck.k_index) for ck in ding304_ns]}")
    
    if di299_ns and ding304_ns:
        di299_h_ns = max([rk.h for ck in di299_ns for rk in ck.klines])
        di299_l_ns = min([rk.l for ck in di299_ns for rk in ck.klines])
        ding304_h_ns = max([rk.h for ck in ding304_ns for rk in ck.klines])
        ding304_l_ns = min([rk.l for ck in ding304_ns for rk in ck.klines])
        c1ns = di299_h_ns > ding304_h_ns
        c2ns = ding304_l_ns < di299_l_ns
        print(f"    C1(start.h>end.h): {di299_h_ns:.2f}>{ding304_h_ns:.2f} = {c1ns}")
        print(f"    C2(end.l<start.l): {ding304_l_ns:.2f}<{di299_l_ns:.2f} = {c2ns}")
        print(f"    → {'FAIL' if c1ns or c2ns else 'PASS'}")

# What about di@299→ding@304 with right-half for start?
    print(f"\n  Right-half start strict:")
    di299_h_rh = max([rk.h for ck in di299.klines[1:] for rk in ck.klines])
    di299_l_rh = min([rk.l for ck in di299.klines[1:] for rk in ck.klines])
    c1rh = di299_h_rh > ding304.high(qj, qy)
    c2rh = ding304.low(qj, qy) < di299_l_rh
    print(f"    di299_h_rh={di299_h_rh:.2f}, ding304_h={ding304.high(qj,qy):.2f}")
    print(f"    di299_l_rh={di299_l_rh:.2f}, ding304_l={ding304.low(qj,qy):.2f}")
    print(f"    C1={c1rh}, C2={c2rh} → {'FAIL' if c1rh or c2rh else 'PASS'}")
