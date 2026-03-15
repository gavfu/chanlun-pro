"""
诊断笔拆分差异：详细跟踪 _bi_special_bi_split 对 bi[8] (187→215) 的处理过程。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P
from chanlun.cl_interface import Config

CL_CONFIG = {
    'bi_type': 'bi_type_old', 'fx_qj': 'fx_qj_k', 'fx_qy': 'fx_qy_three',
    'bi_fx_cgd': 'bi_fx_cgd_yes', 'fx_check_k_nums': 13,
    'bi_split_k_cross_nums': '20,1', 'xd_bzh': 'xd_bzh_no',
}

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_1000.parquet')

co = CL_O("test", "test", config=CL_CONFIG)
co.process_klines(df)

# 获取 pre-split bis
fxs = co.get_fxs()
pre_split_bis = co._build_bis(fxs)

# 找到 pre-split bi[8] (187→215)
target_bi = None
for bi in pre_split_bis:
    if bi.start.k.k_index == 187 and bi.end.k.k_index == 215:
        target_bi = bi
        break

if target_bi is None:
    print("ERROR: target bi not found")
    sys.exit(1)

print(f"目标笔: {target_bi.type} k={target_bi.start.k.k_index}→{target_bi.end.k.k_index}")
print(f"  start CLK index: {target_bi.start.k.index}, end CLK index: {target_bi.end.k.index}")

# 获取内部分型
start_idx = target_bi.start.k.index
end_idx = target_bi.end.k.index
internal_fxs = [fx for fx in co.fxs if start_idx < fx.k.index < end_idx]
print(f"\n内部分型 ({len(internal_fxs)} 个):")
qj, qy = co.fx_qj, co.fx_qy
for fx in internal_fxs:
    h, l = fx.high(qj, qy), fx.low(qj, qy)
    print(f"  {fx.type:>4} k_idx={fx.k.k_index:>4} val={fx.val:.2f} CLK_idx={fx.k.index} high({qj},{qy})={h:.2f} low({qj},{qy})={l:.2f}")

# 三元组检测
print(f"\n=== 三元组交叉计数 ===")
threshold = co.bi_split_k_cross_nums  # 20
tolerance = co.bi_split_k_cross_tolerance  # 1
end_ki = target_bi.end.k.k_index

triggered_ti = -1
for ti in range(len(internal_fxs) - 2):
    fx1 = internal_fxs[ti]
    fx2 = internal_fxs[ti + 1]
    fx3 = internal_fxs[ti + 2]

    h1, l1 = fx1.high(qj, qy), fx1.low(qj, qy)
    h2, l2 = fx2.high(qj, qy), fx2.low(qj, qy)
    h3, l3 = fx3.high(qj, qy), fx3.low(qj, qy)

    hit_count = 0
    miss_count = 0

    for ki in range(fx1.k.k_index, end_ki):
        k = co.src_klines[ki]
        if (k.h >= l1 and k.l <= h1
                and k.h >= l2 and k.l <= h2
                and k.h >= l3 and k.l <= h3):
            hit_count += 1
            miss_count = 0
        else:
            miss_count += 1
        if miss_count > tolerance:
            break

    label = "**TRIGGERED**" if hit_count >= threshold else ""
    print(f"  triplet[{ti}]: ({fx1.type} k={fx1.k.k_index}, {fx2.type} k={fx2.k.k_index}, {fx3.type} k={fx3.k.k_index}) hits={hit_count} {label}")
    
    if hit_count >= threshold and triggered_ti < 0:
        triggered_ti = ti

if triggered_ti >= 0:
    print(f"\n触发三元组: triplet[{triggered_ti}]")
    triplet = (internal_fxs[triggered_ti], internal_fxs[triggered_ti + 1], internal_fxs[triggered_ti + 2])
    for i, fx in enumerate(triplet):
        print(f"  fx{i+1}: {fx.type} k_idx={fx.k.k_index} val={fx.val:.2f} CLK_idx={fx.k.index}")

    # Split point selection (down bi: start(ding) → di_fx → ding_fx → end(di))
    print(f"\n=== Split point selection (down bi) ===")
    
    # _find_split1_from_triplet
    split1_type = "di"  # for down bi, split1 is di
    candidates_s1 = [fx for fx in triplet if fx.type == split1_type]
    candidates_s1.sort(key=lambda f: f.k.index)
    print(f"\nsplit1 candidates (di from triplet):")
    for fx in candidates_s1:
        gap_ok = co._split_gap_ok(target_bi.start, fx)
        cl_gap = fx.k.index - target_bi.start.k.index
        k_gap = fx.k.k_index - target_bi.start.k.k_index
        print(f"  {fx.type} k_idx={fx.k.k_index} CLK={fx.k.index} gap_ok={gap_ok} cl_gap={cl_gap} k_gap={k_gap}")
    
    # What does _find_split1_from_triplet actually return?
    di_fx = co._find_split1_from_triplet(target_bi, triplet, "di")
    print(f"\n_find_split1_from_triplet result: {di_fx.type} k_idx={di_fx.k.k_index}")
    
    # _find_split2
    split2_type = "ding"  # for down bi, split2 is ding
    candidates_s2 = [fx for fx in internal_fxs
                     if fx.type == split2_type and fx.k.index > di_fx.k.index]
    print(f"\nsplit2 candidates (ding after split1):")
    for fx in candidates_s2:
        gap_ok = co._split_gap_ok(di_fx, fx)
        cl_gap = fx.k.index - di_fx.k.index
        k_gap = fx.k.k_index - di_fx.k.k_index
        val_ok = fx.val > di_fx.val  # ding.val > di.val
        print(f"  {fx.type} k_idx={fx.k.k_index} CLK={fx.k.index} val={fx.val:.2f} gap_ok={gap_ok} val_ok={val_ok} cl_gap={cl_gap} k_gap={k_gap}")
    
    ding_fx = co._find_split2(di_fx, target_bi.end, "ding", internal_fxs)
    if ding_fx:
        print(f"\n_find_split2 result: {ding_fx.type} k_idx={ding_fx.k.k_index}")
    else:
        print(f"\n_find_split2 result: None")
    
    # Final split result
    print(f"\n=== cl_open 拆分结果 ===")
    print(f"  bi1: {target_bi.start.type}(k={target_bi.start.k.k_index}) → {di_fx.type}(k={di_fx.k.k_index})")
    if ding_fx:
        print(f"  bi2: {di_fx.type}(k={di_fx.k.k_index}) → {ding_fx.type}(k={ding_fx.k.k_index})")
        print(f"  bi3: {ding_fx.type}(k={ding_fx.k.k_index}) → {target_bi.end.type}(k={target_bi.end.k.k_index})")

# --- 对比 pyarmor 的拆分 ---
cp = CL_P("test", "test", config=CL_CONFIG)
cp.process_klines(df)
print(f"\n=== pyarmor 拆分结果 ===")
bis_p = cp.get_bis()
for i, bi in enumerate(bis_p):
    if 187 <= bi.start.k.k_index <= 215 or 187 <= bi.end.k.k_index <= 215:
        print(f"  bi[{i}] {bi.type:>4} k={bi.start.k.k_index:>4}→{bi.end.k.k_index:>4} is_split='{bi.is_split}'")
