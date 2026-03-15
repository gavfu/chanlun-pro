"""
精确诊断笔差异：逐步追踪 cl_open 和 cl_pyarmor 的 _build_bis 行为
目标：找到 cl_open 接受但 cl_pyarmor 拒绝的分型对(或反过来)

策略：由于两者的分型列表完全一致(fxs ✅)，所以可以：
1. 拿到相同的 fxs 列表
2. 用 cl_open 的 _bi_fx_valid 逐对检测
3. 用 cl_pyarmor 的 _bi_fx_valid (或其笔结果) 对比
4. 找到首个分歧点
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

# 用 BTC60 小数据集 (首个差异在 bi[8])
df = pd.read_parquet('tests/test_data/BTC_USDT_60m_1000.parquet')

co = CL_O("test", "test", config=CL_CONFIG)
co.process_klines(df)
cp = CL_P("test", "test", config=CL_CONFIG)
cp.process_klines(df)

fxs_o = co.get_fxs()
fxs_p = cp.get_fxs()
bis_o = co.get_bis()
bis_p = cp.get_bis()

# 确认分型完全一致
print(f"分型数量: open={len(fxs_o)} pyarmor={len(fxs_p)}")
fx_match = all(
    fo.type == fp.type and fo.k.k_index == fp.k.k_index
    for fo, fp in zip(fxs_o, fxs_p)
)
print(f"分型全部一致: {fx_match}")

# 找首个笔差异
first_diff = None
for i in range(min(len(bis_o), len(bis_p))):
    bo, bp = bis_o[i], bis_p[i]
    if bo.start.k.k_index != bp.start.k.k_index or bo.end.k.k_index != bp.end.k.k_index:
        first_diff = i
        break
print(f"\n首个笔差异: bi[{first_diff}]")
print(f"  open:    {bis_o[first_diff].type} k={bis_o[first_diff].start.k.k_index}→{bis_o[first_diff].end.k.k_index} split={bis_o[first_diff].is_split}")
print(f"  pyarmor: {bis_p[first_diff].type} k={bis_p[first_diff].start.k.k_index}→{bis_p[first_diff].end.k.k_index} split={bis_p[first_diff].is_split}")

# 获取差异前一笔的 end (= 差异笔的 start)
# 两者此处应一致
prev_bi_o = bis_o[first_diff - 1] if first_diff > 0 else None
prev_bi_p = bis_p[first_diff - 1] if first_diff > 0 else None
if prev_bi_o and prev_bi_p:
    print(f"\n前一笔 bi[{first_diff-1}]:")
    print(f"  open:    {prev_bi_o.type} k={prev_bi_o.start.k.k_index}→{prev_bi_o.end.k.k_index}")
    print(f"  pyarmor: {prev_bi_p.type} k={prev_bi_p.start.k.k_index}→{prev_bi_p.end.k.k_index}")

# 关键分析：在 bi[first_diff] 的 start 之后，cl_open 找到了什么 end
diff_bo = bis_o[first_diff]
diff_bp = bis_p[first_diff]

# 找到 cl_open 笔 start 对应的分型索引
start_fx = diff_bo.start
start_fx_idx = None
for idx, fx in enumerate(fxs_o):
    if fx.k.k_index == start_fx.k.k_index and fx.type == start_fx.type:
        start_fx_idx = idx
        break

print(f"\n差异笔 start: fx[{start_fx_idx}] {start_fx.type} k_idx={start_fx.k.k_index}")

# 从 start_fx 之后扫描，找 cl_open 认可的第一个 end_fx
print(f"\n--- 从 start_fx 后扫描分型 ---")
for j in range(start_fx_idx + 1, min(start_fx_idx + 30, len(fxs_o))):
    fx = fxs_o[j]
    if fx.type == start_fx.type:
        print(f"  fx[{j}] {fx.type:>4} k_idx={fx.k.k_index:>4} val={fx.val:.2f} (同向,skip)")
        continue
    
    # 反向分型 - 测试 _bi_fx_valid
    valid = co._bi_fx_valid(start_fx, fx)
    cl_gap = fx.k.index - start_fx.k.index
    k_gap = fx.k.k_index - start_fx.k.k_index
    
    # 手动检查 strict 条件
    strict_info = ""
    if k_gap < co.fx_check_k_nums and co.allow_bi_fx_strict:
        qj, qy = co.fx_qj, co.fx_qy
        if start_fx.type == "ding" and fx.type == "di":
            s_low = start_fx.low(qj, qy)
            e_low = fx.low(qj, qy)
            e_high = fx.high(qj, qy)
            s_high = start_fx.high(qj, qy)
            if s_low < e_low:
                strict_info = f"STRICT_FAIL: start.low({s_low:.2f}) < end.low({e_low:.2f})"
            elif e_high > s_high:
                strict_info = f"STRICT_FAIL: end.high({e_high:.2f}) > start.high({s_high:.2f})"
            else:
                strict_info = f"strict_ok: s_low={s_low:.2f}>=e_low={e_low:.2f}, e_high={e_high:.2f}<=s_high={s_high:.2f}"
        elif start_fx.type == "di" and fx.type == "ding":
            s_high = start_fx.high(qj, qy)
            e_high = fx.high(qj, qy)
            e_low = fx.low(qj, qy)
            s_low = start_fx.low(qj, qy)
            if s_high > e_high:
                strict_info = f"STRICT_FAIL: start.high({s_high:.2f}) > end.high({e_high:.2f})"
            elif e_low < s_low:
                strict_info = f"STRICT_FAIL: end.low({e_low:.2f}) < start.low({s_low:.2f})"
            else:
                strict_info = f"strict_ok: s_high={s_high:.2f}<=e_high={e_high:.2f}, e_low={e_low:.2f}>=s_low={s_low:.2f}"

    is_open_end = (fx.k.k_index == diff_bo.end.k.k_index)
    is_pyarmor_end = (fx.k.k_index == diff_bp.end.k.k_index)
    markers = []
    if is_open_end: markers.append("← OPEN_END")
    if is_pyarmor_end: markers.append("← PYARMOR_END")
    marker = " ".join(markers)
    
    print(f"  fx[{j}] {fx.type:>4} k_idx={fx.k.k_index:>4} val={fx.val:.2f} cl_gap={cl_gap} k_gap={k_gap} valid={valid} {strict_info} {marker}")

# 检查 pyarmor 端的笔是否是 split 笔
print(f"\n--- pyarmor 差异笔附近 ---")
for i in range(max(0, first_diff - 1), min(first_diff + 4, len(bis_p))):
    bp = bis_p[i]
    print(f"  pyarmor bi[{i}] {bp.type:>4} k={bp.start.k.k_index:>4}→{bp.end.k.k_index:>4} split={bp.is_split}")

# 检查 open 端的笔
print(f"\n--- open 差异笔附近 ---")
for i in range(max(0, first_diff - 1), min(first_diff + 4, len(bis_o))):
    bo = bis_o[i]
    print(f"  open bi[{i}] {bo.type:>4} k={bo.start.k.k_index:>4}→{bo.end.k.k_index:>4} split={bo.is_split}")
