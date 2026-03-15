"""
诊断：检查 _build_bis 输出（拆分前）与 _bi_special_bi_split 输出（拆分后）的差异。
还检查 cl_open 的 split 逻辑是否设置 is_split 字段。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

CL_CONFIG = {
    'bi_type': 'bi_type_old', 'fx_qj': 'fx_qj_k', 'fx_qy': 'fx_qy_three',
    'bi_fx_cgd': 'bi_fx_cgd_yes', 'fx_check_k_nums': 13,
    'bi_split_k_cross_nums': '20,1', 'xd_bzh': 'xd_bzh_no',
}

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_1000.parquet')

# --- 方法1：手动调用 _build_bis 和 _bi_special_bi_split ---
co = CL_O("test", "test", config=CL_CONFIG)
co.process_klines(df)

# 获取 fxs 并手动调用 _build_bis
fxs = co.get_fxs()
pre_split_bis = co._build_bis(fxs)
post_split_bis = co._bi_special_bi_split(pre_split_bis)

print("=== _build_bis 输出（拆分前）===")
for i, bi in enumerate(pre_split_bis[:15]):
    print(f"  bi[{i}] {bi.type:>4} k={bi.start.k.k_index:>4}→{bi.end.k.k_index:>4} is_split='{bi.is_split}'")

print(f"\n=== _bi_special_bi_split 输出（拆分后）===")
for i, bi in enumerate(post_split_bis[:15]):
    print(f"  bi[{i}] {bi.type:>4} k={bi.start.k.k_index:>4}→{bi.end.k.k_index:>4} is_split='{bi.is_split}'")

# 比较前后差异
print(f"\n拆分前笔数: {len(pre_split_bis)}, 拆分后笔数: {len(post_split_bis)}")
if len(pre_split_bis) != len(post_split_bis):
    print("拆分发生了变化!")
    # 找到第一个差异
    for i in range(min(len(pre_split_bis), len(post_split_bis))):
        pb = pre_split_bis[i] if i < len(pre_split_bis) else None
        ab = post_split_bis[i] if i < len(post_split_bis) else None
        if pb and ab:
            if pb.start.k.k_index != ab.start.k.k_index or pb.end.k.k_index != ab.end.k.k_index:
                print(f"  首个差异 bi[{i}]: pre={pb.start.k.k_index}→{pb.end.k.k_index} post={ab.start.k.k_index}→{ab.end.k.k_index}")
                break

# --- 方法2：对比 co.get_bis() (经过 split 后的最终结果) ---
print(f"\n=== co.get_bis() 最终结果 ===")
final_bis = co.get_bis()
for i, bi in enumerate(final_bis[:15]):
    print(f"  bi[{i}] {bi.type:>4} k={bi.start.k.k_index:>4}→{bi.end.k.k_index:>4} is_split='{bi.is_split}'")

# --- 方法3：pyarmor 对比 ---
cp = CL_P("test", "test", config=CL_CONFIG)
cp.process_klines(df)
bis_p = cp.get_bis()
print(f"\n=== pyarmor bis ===")
for i, bi in enumerate(bis_p[:15]):
    print(f"  bi[{i}] {bi.type:>4} k={bi.start.k.k_index:>4}→{bi.end.k.k_index:>4} is_split='{bi.is_split}'")

# --- 方法4：验证 _bi_fx_valid 对 bi[8] start→end 的判断 ---
print(f"\n=== _bi_fx_valid 验证 ===")
# 找到 final bi[8] 的 start 和 end FX
if len(final_bis) > 8:
    bi8 = final_bis[8]
    bi8_start = bi8.start
    bi8_end = bi8.end
    valid = co._bi_fx_valid(bi8_start, bi8_end)
    cl_gap = bi8_end.k.index - bi8_start.k.index
    k_gap = bi8_end.k.k_index - bi8_start.k.k_index
    print(f"  bi[8]: {bi8.type} k={bi8_start.k.k_index}→{bi8_end.k.k_index}")
    print(f"  _bi_fx_valid = {valid}")
    print(f"  cl_gap = {cl_gap}, k_gap = {k_gap}")
    print(f"  start.k.index = {bi8_start.k.index}, end.k.index = {bi8_end.k.index}")

# 找 pre_split bi 在差异区域
print(f"\n=== pre_split bis 在 k=187 附近 ===")
for i, bi in enumerate(pre_split_bis):
    if 170 <= bi.start.k.k_index <= 220 or 170 <= bi.end.k.k_index <= 220:
        valid = co._bi_fx_valid(bi.start, bi.end)
        cl_gap = bi.end.k.index - bi.start.k.index
        k_gap = bi.end.k.k_index - bi.start.k.k_index
        print(f"  bi[{i}] {bi.type:>4} k={bi.start.k.k_index:>4}→{bi.end.k.k_index:>4} valid={valid} cl_gap={cl_gap} k_gap={k_gap}")
