# -*- coding: utf-8 -*-
"""Compare _bi_fx_valid between cl_open and pyarmor for specific ETH5m FX pairs"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor

DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
df = pd.read_parquet(os.path.join(DATA_DIR, 'ETH_USDT_5m_1000.parquet'))

cd_open = CL_Open("ETH/USDT", "5m", config={})
cd_open.process_klines(df)

cd_py = CL_Pyarmor("ETH/USDT", "5m", config={})
cd_py.process_klines(df)

# Both have same FXs
open_fx = {fx.k.index: fx for fx in cd_open.fxs}
py_fx = {fx.k.index: fx for fx in cd_py.fxs}

qj = cd_open.fx_qj
qy = cd_open.fx_qy

# Key pairs to test — where cl_open confirms but pyarmor doesn't
# Diff #34: end_fx=357, confirming=361 (should not confirm per pyarmor)
# Diff #43: end_fx=454, confirming=458 (should not confirm per pyarmor)
# Diff #55: end_fx=594, confirming=598 (should not confirm per pyarmor)

# Also test the start→end_fx pairs to see if they match
pairs = [
    # (name, start_idx, end_idx)
    # Diff #34 area
    ("346→352 (set end_fx)", 346, 352),
    ("346→357 (extend end_fx)", 346, 357),
    ("357→359 (try confirm)", 357, 359),
    ("357→361 (cl_open confirms)", 357, 361),
    ("346→363 (pyarmor extends to)", 346, 363),
    ("363→369 (pyarmor confirms from 363)", 363, 369),
    # Diff #43 area
    ("448→452 (try set end)", 448, 452),
    ("448→454 (cl_open sets end)", 448, 454),
    ("454→456 (try confirm)", 454, 456),
    ("454→458 (cl_open confirms)", 454, 458),
    ("448→461 (pyarmor end_fx)", 448, 461),
    ("461→465 (pyarmor confirms)", 461, 465),
    # Diff #55 area
    ("581→586 (try set end)", 581, 586),
    ("581→589 (extend)", 581, 589),
    ("581→594 (cl_open end)", 581, 594),
    ("594→595 (try confirm)", 594, 595),
    ("594→598 (cl_open confirms)", 594, 598),
    ("581→601 (pyarmor end)", 581, 601),
    ("601→605 (pyarmor confirms)", 601, 605),
]

print(f"{'Pair':<40} {'cl_open':>8} {'pyarmor':>8} {'Match':>6}")
print("-" * 65)

for name, si, ei in pairs:
    o_s = open_fx.get(si)
    o_e = open_fx.get(ei)
    p_s = py_fx.get(si)
    p_e = py_fx.get(ei)

    if not (o_s and o_e and p_s and p_e):
        print(f"{name:<40} {'N/A':>8} {'N/A':>8}")
        continue

    o_result = cd_open._bi_fx_valid(o_s, o_e)
    p_result = cd_py._bi_check_bi_fx_ok(p_s, p_e)

    match = "✅" if o_result == p_result else "❌"
    print(f"{name:<40} {str(o_result):>8} {str(p_result):>8} {match:>6}")

    if o_result != p_result:
        cl_gap = o_e.k.index - o_s.k.index
        k_gap = o_e.k.k_index - o_s.k.k_index
        print(f"  cl_gap={cl_gap}, k_gap={k_gap}, fx_check_k_nums={cd_open.fx_check_k_nums}")
        print(f"  start: FX({si},{o_s.type}) h={o_s.high(qj,qy):.2f} l={o_s.low(qj,qy):.2f}")
        print(f"  end:   FX({ei},{o_e.type}) h={o_e.high(qj,qy):.2f} l={o_e.low(qj,qy):.2f}")
