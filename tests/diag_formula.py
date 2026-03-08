# -*- coding: utf-8 -*-
"""Reverse-engineer the exact strict check formula by analyzing ALL pyarmor strokes"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "test_data", "BTC_USDT_60m_500.parquet"))

cd_p = CL_Pyarmor("BTC/USDT", "60m")
cd_p.process_klines(df)
cd_o = CL_Open("BTC/USDT", "60m")
cd_o.process_klines(df)

o_fxs = cd_o.get_fxs()
p_bis = cd_p.get_bis()
qj = cd_o.fx_qj
qy = cd_o.fx_qy

# For each pyarmor stroke with k_gap < fx_check_k_nums, show ALL possible check values
print("=== VALID strokes (pyarmor, k_gap < 13): candidate formulae ===")
print(f"{'bi':>4s} {'type':>5s} {'s->e':>10s} {'k_gap':>5s} | {'s.h3':>8s} {'s.l3':>8s} {'e.h3':>8s} {'e.l3':>8s} | {'s.kh':>8s} {'s.kl':>8s} {'e.kh':>8s} {'e.kl':>8s}")

valid_rows = []
for i, bi in enumerate(p_bis):
    s = bi.start
    e = bi.end
    k_gap = e.k.k_index - s.k.k_index
    if k_gap >= 13:
        continue
    sh = s.high(qj, qy)
    sl = s.low(qj, qy)
    eh = e.high(qj, qy)
    el = e.low(qj, qy)
    
    print(f"bi[{i:2d}] {bi.type:5s} {s.k.index:3d}->{e.k.index:3d} k={k_gap:3d} | "
          f"sh={sh:8.1f} sl={sl:8.1f} eh={eh:8.1f} el={el:8.1f} | "
          f"skh={s.k.h:8.1f} skl={s.k.l:8.1f} ekh={e.k.h:8.1f} ekl={e.k.l:8.1f}")
    valid_rows.append({
        'idx': i, 'type': bi.type, 'k_gap': k_gap,
        'sh': sh, 'sl': sl, 'eh': eh, 'el': el,
        'skh': s.k.h, 'skl': s.k.l, 'ekh': e.k.h, 'ekl': e.k.l,
        'sval': s.val, 'eval': e.val,
    })

# Now collect ALL REJECTED candidates (should fail strict but pass gap check)
# These are fractals that become end_fx candidates in the algorithm but pyarmor doesn't use
print(f"\n=== REJECTED candidates (should fail, k_gap < 13) ===")

# The REJECTED cases from the previous analysis
# Let me compute them by finding opposite-type fxs between consecutive pyarmor stroke start/ends
reject_rows = []
for i, bi in enumerate(p_bis):
    end_fx = bi.end
    if i + 1 >= len(p_bis):
        break
    next_start = p_bis[i + 1].start
    # next_start should be same as end_fx
    assert next_start.k.index == end_fx.k.index, f"bi[{i}] end={end_fx.k.index} != bi[{i+1}] start={next_start.k.index}"
    
    next_end = p_bis[i + 1].end
    
    for fx in o_fxs:
        if fx.k.index <= end_fx.k.index:
            continue
        if fx.k.index >= next_end.k.index:
            break
        if fx.type == end_fx.type:
            continue
        
        cl_gap = fx.k.index - end_fx.k.index
        k_gap = fx.k.k_index - end_fx.k.k_index
        if cl_gap < 4 or k_gap < 4:
            continue
        if k_gap >= 13:
            continue
        
        s = end_fx
        e = fx
        sh = s.high(qj, qy)
        sl = s.low(qj, qy)
        eh = e.high(qj, qy)
        el = e.low(qj, qy)
        
        stroke_type = "up" if s.type == "di" else "down"
        print(f"REJ[{i:2d}] {stroke_type:5s} {s.k.index:3d}->{e.k.index:3d} k={k_gap:3d} | "
              f"sh={sh:8.1f} sl={sl:8.1f} eh={eh:8.1f} el={el:8.1f} | "
              f"skh={s.k.h:8.1f} skl={s.k.l:8.1f} ekh={e.k.h:8.1f} ekl={e.k.l:8.1f}")
        reject_rows.append({
            'idx': i, 'type': stroke_type, 'k_gap': k_gap,
            'sh': sh, 'sl': sl, 'eh': eh, 'el': el,
            'skh': s.k.h, 'skl': s.k.l, 'ekh': e.k.h, 'ekl': e.k.l,
            'sval': s.val, 'eval': e.val,
        })

# Now try EVERY possible strict check formula and see which one classifies correctly
print(f"\n=== Testing formulae ===")
print(f"Valid: {len(valid_rows)} strokes, Rejected: {len(reject_rows)} candidates")
print()

# For each formula, count how many valid it would PASS and how many rejected it would BLOCK
def test_formula(name, check_fn):
    valid_pass = sum(1 for r in valid_rows if check_fn(r))
    valid_fail = len(valid_rows) - valid_pass
    reject_block = sum(1 for r in reject_rows if not check_fn(r))
    reject_pass = len(reject_rows) - reject_block
    perfect = valid_fail == 0 and reject_pass == 0
    print(f"  {name:40s} valid_pass={valid_pass}/{len(valid_rows)} reject_block={reject_block}/{len(reject_rows)} {'*** PERFECT ***' if perfect else ''}")
    if valid_fail > 0:
        print(f"    FAILS valid: {[r['idx'] for r in valid_rows if not check_fn(r)]}")
    if reject_pass > 0:
        print(f"    MISSES reject: {[r['idx'] for r in reject_rows if check_fn(r)]}")

# Formula 1: current (h>h for up, l<l for down) — but this is for BLOCKING, so we check PASS
def f1(r):
    if r['type'] == 'up':
        return not (r['sh'] > r['eh'])  # pass if NOT (start high > end high)
    else:
        return not (r['sl'] < r['el'])  # pass if NOT (start low < end low)

# Formula 2: also check the other direction
def f2(r):
    if r['type'] == 'up':
        return not (r['sh'] > r['eh']) and not (r['sl'] > r['el'])
    else:
        return not (r['sl'] < r['el']) and not (r['sh'] < r['eh'])

# Formula 3: using center K-line values
def f3(r):
    if r['type'] == 'up':
        return not (r['skh'] > r['ekh'])
    else:
        return not (r['skl'] < r['ekl'])

# Formula 4: using center K-line both directions
def f4(r):
    if r['type'] == 'up':
        return not (r['skh'] > r['ekh']) and not (r['skl'] > r['ekl'])
    else:
        return not (r['skl'] < r['ekl']) and not (r['skh'] < r['ekh'])

# Formula 5: h3>h3 AND l3>l3 for up; l3<l3 AND h3<h3 for down
def f5(r):
    if r['type'] == 'up':
        return not (r['sh'] > r['eh'] and r['sl'] > r['el'])
    else:
        return not (r['sl'] < r['el'] and r['sh'] < r['eh'])

# Formula 6: cross-check: s.l3 > e.h3 blocking (no overlap allowed)
def f6(r):
    if r['type'] == 'up':
        return r['sh'] < r['eh']  # must be: start high < end high
    else:
        return r['sl'] > r['el']  # must be: start low > end low

# Formula 7: using val instead of h3/l3
def f7(r):
    if r['type'] == 'up':
        return r['sval'] < r['eval']  # di val < ding val
    else:
        return r['sval'] > r['eval']  # ding val > di val

# Formula 8: cross k.h and h3
def f8(r):
    if r['type'] == 'up':
        return not (r['skh'] > r['eh'])
    else:
        return not (r['skl'] < r['el'])

# Formula 9: h3 for one direction, k.h for other
def f9(r):
    if r['type'] == 'up':
        return not (r['sh'] > r['ekh'])
    else:
        return not (r['sl'] < r['ekl'])

# Formula 10: k.h vs k.h and k.l vs k.l
def f10(r):
    if r['type'] == 'up':
        return not (r['skh'] > r['ekh'] or r['skl'] > r['ekl'])
    else:
        return not (r['skl'] < r['ekl'] or r['skh'] < r['ekh'])

# Formula 11: h3 for main check, k.h for cross
def f11(r):
    if r['type'] == 'up':
        return not (r['sh'] > r['eh']) and not (r['skh'] > r['ekl'])
    else:
        return not (r['sl'] < r['el']) and not (r['skl'] < r['ekh'])

# Formula 12: inverse check on val
def f12(r):
    if r['type'] == 'up':
        return not (r['sh'] > r['eh']) and r['sval'] < r['eval']
    else:
        return not (r['sl'] < r['el']) and r['sval'] > r['eval']

test_formula("F1: h3>h3 (current)", f1)
test_formula("F2: h3>h3 OR l3>l3 (both dirs)", f2)
test_formula("F3: k.h>k.h (center only)", f3)
test_formula("F4: k.h>k.h AND k.l>k.l", f4)
test_formula("F5: h3>h3 AND l3>l3", f5)
test_formula("F6: require h3<h3 (strict separation)", f6)
test_formula("F7: val comparison", f7)
test_formula("F8: k.h vs h3 (cross)", f8)
test_formula("F9: h3 vs k.h (cross)", f9)
test_formula("F10: k.h OR k.l", f10)
test_formula("F11: h3 + k.h vs k.l cross", f11)
test_formula("F12: h3 + val comparison", f12)
