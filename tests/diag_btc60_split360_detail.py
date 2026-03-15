"""
Diagnose the BTC60 bi[16] split (down 360→403):
- cl_open splits to: 360→378, 378→389, 389→403
- pyarmor splits to: 360→370, 370→381, 381→403
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O

CL_CONFIG = {
    'bi_type': 'bi_type_old', 'fx_qj': 'fx_qj_k', 'fx_qy': 'fx_qy_three',
    'bi_fx_cgd': 'bi_fx_cgd_yes', 'fx_check_k_nums': 13,
    'bi_split_k_cross_nums': '20,1', 'xd_bzh': 'xd_bzh_no',
}

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_1000.parquet')
co = CL_O("t", "t", config=CL_CONFIG)
co.process_klines(df)

pre = co._build_bis(co.get_fxs())
target = None
for bi in pre:
    if bi.start.k.k_index == 360 and bi.end.k.k_index == 403:
        target = bi
        break

print(f"Target bi: {target.type} k={target.start.k.k_index}→{target.end.k.k_index}")
print(f"  start CLK={target.start.k.index}, end CLK={target.end.k.index}")

qj, qy = co.fx_qj, co.fx_qy
start_idx = target.start.k.index
end_idx = target.end.k.index
internal_fxs = [fx for fx in co.fxs if start_idx < fx.k.index < end_idx]

print(f"\n内部分型 ({len(internal_fxs)} 个):")
for fx in internal_fxs:
    print(f"  {fx.type:>4} k={fx.k.k_index:>4} val={fx.val:.2f} CLK={fx.k.index}")

# Triplet detection
threshold = co.bi_split_k_cross_nums
tolerance = co.bi_split_k_cross_tolerance
end_ki = target.end.k.k_index

print(f"\n=== 三元组交叉检测 (threshold={threshold}) ===")
triggered = []
for ti in range(len(internal_fxs) - 2):
    fx1 = internal_fxs[ti]
    fx2 = internal_fxs[ti + 1]
    fx3 = internal_fxs[ti + 2]
    h1, l1 = fx1.high(qj, qy), fx1.low(qj, qy)
    h2, l2 = fx2.high(qj, qy), fx2.low(qj, qy)
    h3, l3 = fx3.high(qj, qy), fx3.low(qj, qy)
    hit = miss = 0
    for ki in range(fx1.k.k_index, end_ki):
        k = co.src_klines[ki]
        if (k.h >= l1 and k.l <= h1 and k.h >= l2 and k.l <= h2 and k.h >= l3 and k.l <= h3):
            hit += 1
            miss = 0
        else:
            miss += 1
        if miss > tolerance: break
    label = "**TRIGGERED**" if hit >= threshold else ""
    print(f"  t[{ti}] ({fx1.type} k={fx1.k.k_index}, {fx2.type} k={fx2.k.k_index}, {fx3.type} k={fx3.k.k_index}) hits={hit} {label}")
    if hit >= threshold:
        triggered.append(ti)

if triggered:
    ti = triggered[0]
    triplet = (internal_fxs[ti], internal_fxs[ti + 1], internal_fxs[ti + 2])
    print(f"\n使用三元组 t[{ti}]:")
    for i, fx in enumerate(triplet):
        print(f"  fx{i}: {fx.type} k={fx.k.k_index} val={fx.val:.2f}")

    # Split1 selection (down bi → split1_type = "di")
    split1_type = "di"
    candidates_s1 = [fx for fx in triplet if fx.type == split1_type]
    print(f"\nsplit1 candidates (di from triplet, sorted by EXTREME val):")
    if split1_type == "di":
        candidates_s1.sort(key=lambda f: f.val)  # lowest first for di
    else:
        candidates_s1.sort(key=lambda f: f.val, reverse=True)
    for c in candidates_s1:
        gap_ok = co._split_gap_ok(target.start, c)
        cl_gap = c.k.index - target.start.k.index
        k_gap = c.k.k_index - target.start.k.k_index
        print(f"  {c.type} k={c.k.k_index} val={c.val:.2f} gap_ok={gap_ok} cl_gap={cl_gap} k_gap={k_gap}")

    actual_s1 = co._find_split1_from_triplet(target, triplet, split1_type)
    print(f"  → _find_split1_from_triplet result: {actual_s1.type} k={actual_s1.k.k_index}")

    # Split2 selection (down bi → split2_type = "ding")
    actual_s2 = co._find_split2(actual_s1, target.end, "ding", internal_fxs)
    if actual_s2:
        print(f"  → _find_split2 result: {actual_s2.type} k={actual_s2.k.k_index}")
    else:
        print(f"  → _find_split2 result: None")

    print(f"\n=== cl_open 拆分结果 ===")
    print(f"  bi1: {target.start.k.k_index}→{actual_s1.k.k_index}")
    if actual_s2:
        print(f"  bi2: {actual_s1.k.k_index}→{actual_s2.k.k_index}")
        print(f"  bi3: {actual_s2.k.k_index}→{target.end.k.k_index}")

    print(f"\n=== pyarmor 拆分结果 ===")
    print(f"  bi1: 360→370")
    print(f"  bi2: 370→381")
    print(f"  bi3: 381→403")

    # Check ALL internal di FXs
    print(f"\n=== ALL internal di FXs ===")
    for fx in internal_fxs:
        if fx.type == "di":
            gap_ok = co._split_gap_ok(target.start, fx)
            cl_gap = fx.k.index - target.start.k.index
            k_gap = fx.k.k_index - target.start.k.k_index
            in_triplet = fx in triplet
            print(f"  di k={fx.k.k_index} val={fx.val:.2f} gap_ok={gap_ok} cl_gap={cl_gap} k_gap={k_gap} in_triplet={in_triplet}")
