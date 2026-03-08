# -*- coding: utf-8 -*-
"""Analyze divergence points between open and pyarmor for 500k dataset"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pathlib
import pandas as pd
from chanlun.cl_open import CL as CLOpen
from chanlun.cl_interface import Config

df = pd.read_parquet(pathlib.Path(__file__).parent / "test_data" / "BTC_USDT_60m_500.parquet")
c = CLOpen("BTC/USDT", "60m", {})
c.process_klines(df)

fxs = c.fxs

qj_ck = Config.FX_QJ_CK.value
qy_mid = Config.FX_QY_MIDDLE.value

def show_fx_pair(fxs, a, b, label=""):
    fx_a, fx_b = fxs[a], fxs[b]
    cl_gap = fx_b.k.index - fx_a.k.index
    k_gap = fx_b.k.k_index - fx_a.k.k_index
    print(f"\n  FX{a}({fx_a.type})->FX{b}({fx_b.type}) {label}")
    print(f"    cl_gap={cl_gap}, k_gap={k_gap}, val: {fx_a.val}->{fx_b.val}")
    
    sh_ck = fx_a.high(qj_ck, qy_mid)
    sl_ck = fx_a.low(qj_ck, qy_mid)
    eh_ck = fx_b.high(qj_ck, qy_mid)
    el_ck = fx_b.low(qj_ck, qy_mid)
    
    gap_ok = k_gap >= 4 and not (cl_gap < 4 and k_gap < 5)
    
    if fx_a.type == "di" and fx_b.type == "ding":
        fail1 = sh_ck > eh_ck
        fail2 = el_ck < sl_ck
        print(f"    CK+MID strict(up): start_h({sh_ck}) > end_h({eh_ck})? {fail1}")
        print(f"    CK+MID strict(up): end_l({el_ck}) < start_l({sl_ck})? {fail2}")
        strict_ok = not fail1 and not fail2
    elif fx_a.type == "ding" and fx_b.type == "di":
        fail1 = sl_ck < el_ck
        fail2 = eh_ck > sh_ck
        print(f"    CK+MID strict(dn): start_l({sl_ck}) < end_l({el_ck})? {fail1}")
        print(f"    CK+MID strict(dn): end_h({eh_ck}) > start_h({sh_ck})? {fail2}")
        strict_ok = not fail1 and not fail2
    else:
        strict_ok = True
    print(f"    => bi_valid? gap_ok={gap_ok}, strict_ok={strict_ok} => {gap_ok and strict_ok}")

print("=== Divergence #1: bi[5] ===")
show_fx_pair(fxs, 19, 22, "(open bi[5] end, EXTRA)")
show_fx_pair(fxs, 19, 26, "(pyarmor bi[5] end, MISSING from open)")

print("\n=== Divergence #2: bi[12] ===")
show_fx_pair(fxs, 54, 57, "(open bi[14] end, EXTRA)")
show_fx_pair(fxs, 54, 69, "(pyarmor bi[12] end, MISSING from open)")

print("\n=== Divergence #3: bi[21] ===")
show_fx_pair(fxs, 129, 130, "(open bi[27] end, EXTRA)")
show_fx_pair(fxs, 129, 136, "(pyarmor bi[21] end, MISSING from open)")

from chanlun.cl_open import CL

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "test_data", "BTC_USDT_60m_500.parquet"))
cd = CL("BTC/USDT", "60m")
cd.process_klines(df)

fxs = cd.get_fxs()
fx_map = {fx.k.index: fx for fx in fxs}

# All ONLY-OFF first strokes (first divergences where strict blocks)
# From the analysis: up 40->44, up 96->101, up 262->267
cases = [
    (40, 44, "up"),
    (96, 101, "up"),
    (262, 267, "up"),
    # Also check valid cases that pyarmor accepts
    (25, 29, "up"),
    (65, 74, "up"),
    (347, 355, "up"),
]

for s_ck, e_ck, btype in cases:
    if s_ck not in fx_map or e_ck not in fx_map:
        print(f"MISSING fx for {s_ck} or {e_ck}")
        continue
    s = fx_map[s_ck]
    e = fx_map[e_ck]
    k_gap = e.k.k_index - s.k.k_index
    
    print(f"\n{'='*60}")
    print(f"{btype} {s_ck}->{e_ck} k_gap={k_gap}")
    
    # Show CLKlines for each fractal
    for fx_idx, fx, label in [(s_ck, s, "start"), (e_ck, e, "end")]:
        print(f"  {label} FX ck={fx_idx} {fx.type} val={fx.val:.1f}")
        for j, ck in enumerate(fx.klines):
            if ck is None:
                continue
            klines_str = " | ".join([f"k[{k.index}] h={k.h:.1f} l={k.l:.1f}" for k in ck.klines])
            print(f"    ck[{ck.index}] h={ck.h:.1f} l={ck.l:.1f} <- {klines_str}")
    
    # All qj/qy combos
    for qj in ['fx_qj_k', 'fx_qj_ck']:
        for qy in ['fx_qy_middle', 'fx_qy_three']:
            sh = s.high(qj, qy)
            sl = s.low(qj, qy)
            eh = e.high(qj, qy)
            el = e.low(qj, qy)
            # For up stroke: di->ding
            check_h = sh > eh
            check_h_eq = sh >= eh
            check_l = sl > el
            check_l_eq = sl >= el
            print(f"  {qj:10s} {qy:14s}: sh={sh:8.1f} sl={sl:8.1f} eh={eh:8.1f} el={el:8.1f}")
            print(f"    sh>eh={check_h} sh>=eh={check_h_eq} sl>el={check_l} sl>=el={check_l_eq}")
