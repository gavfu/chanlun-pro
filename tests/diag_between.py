# -*- coding: utf-8 -*-
"""
Check if strict validation looks at K-lines BETWEEN start and end fractals
rather than just fractal element high/low values.

Theory: For up stroke di->ding, check if ANY K-line between them has a high
above the ding's val (or equivalently, if there's a ding-like pattern in between).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from chanlun.cl_open import CL

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "test_data", "BTC_USDT_60m_500.parquet"))
cd = CL("BTC/USDT", "60m")
cd.process_klines(df)

fxs = cd.get_fxs()
fx_map = {fx.k.index: fx for fx in fxs}
cl_klines = cd.get_cl_klines()

# For each case, check if there's a K-line between start and end
# with high > end.val (for up) or low < end.val (for down)
cases = [
    # (start_ck, end_ck, type, should_block)
    (40, 44, "up", True),
    (96, 101, "up", True),
    (262, 267, "up", True),
    (25, 29, "up", False),
    (65, 74, "up", False),
    (347, 355, "up", False),
    # Also check some down cases from strict OFF list
]

print("=== Check for K-lines between start/end with extreme values ===\n")
for s_ck, e_ck, btype, should_block in cases:
    s = fx_map[s_ck]
    e = fx_map[e_ck]
    
    # Get all CLKlines between start and end (exclusive)
    max_h = 0
    min_l = float('inf')
    max_h_between = 0  # excluding start/end fractal klines
    min_l_between = float('inf')
    
    # Inclusive of start and end
    for ck in cl_klines[s_ck:e_ck+1]:
        max_h = max(max_h, ck.h)
        min_l = min(min_l, ck.l)
    
    # Exclusive of start and end fratal's 3 CLKlines (just the "between" klines)
    start_end = s_ck + 2  # skip start fractal's last element
    end_start = e_ck - 1  # skip end fractal's first element
    for idx in range(start_end, end_start + 1):
        if 0 <= idx < len(cl_klines):
            ck = cl_klines[idx]
            max_h_between = max(max_h_between, ck.h)
            min_l_between = min(min_l_between, ck.l)
    
    # Original K-line highs between
    klines = cd.get_src_klines()
    max_k_h_between = 0
    min_k_l_between = float('inf')
    s_k_idx = s.k.k_index + 1
    e_k_idx = e.k.k_index
    for k in klines[s_k_idx:e_k_idx]:
        max_k_h_between = max(max_k_h_between, k.h)
        min_k_l_between = min(min_k_l_between, k.l)
    
    # Check various conditions
    blocked = "SHOULD_BLOCK" if should_block else "SHOULD_PASS"
    
    if btype == "up":
        # Does the start's FX range "contain" any part of the end's range?
        # i.e., is there overlap suggesting the "up" isn't clean?
        
        # Check: max high of all klines between > end val?
        check_a = max_h > e.val
        check_b = max_h_between > e.val
        check_c = max_k_h_between > e.val
        
        # Check: start's first CLKline h > end.val?
        s_first_h = s.klines[0].h if s.klines[0] is not None else 0
        check_d = s_first_h > e.val
        
        # Check: any CLKline before the end that has h > end.k.h?
        check_e = max_h > e.k.h
        check_f = max_k_h_between > e.k.h
        
        # Check start fractal's left element (ck[s-1]) h > end val
        left_ck_h = s.klines[0].h if len(s.klines) > 0 and s.klines[0] is not None else 0
        check_g = left_ck_h > e.val
        
        print(f"{btype} {s_ck:3d}->{e_ck:3d} [{blocked}]")
        print(f"  s.val={s.val:.1f} e.val={e.val:.1f} s_first_ck_h={s_first_h:.1f}")
        print(f"  max_h_inclusive={max_h:.1f} max_h_between={max_h_between:.1f} max_k_h={max_k_h_between:.1f}")
        print(f"  check_a(max_h_inc > e.val)={check_a}")
        print(f"  check_b(max_h_btw > e.val)={check_b}")
        print(f"  check_c(max_kh_btw > e.val)={check_c}")
        print(f"  check_d(s_first_h > e.val)={check_d}")
        print(f"  check_e(max_h > e.k.h)={check_e}")
        print(f"  check_f(max_kh > e.k.h)={check_f}")
        print(f"  check_g(left_ck > e.val)={check_g}")
    print()
