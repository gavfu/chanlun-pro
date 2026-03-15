"""诊断笔拆分差异 - 检查 cl_open split 是否工作"""
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

co = CL_O("test", "test", config=CL_CONFIG)
co.process_klines(df)
cp = CL_P("test", "test", config=CL_CONFIG)
cp.process_klines(df)

bis_o = co.get_bis()
bis_p = cp.get_bis()

# 1. 检查 open 是否有任何 split 笔
split_o = [b for b in bis_o if b.is_split]
split_p = [b for b in bis_p if b.is_split]
print(f"cl_open split bishu: {len(split_o)}")
for b in split_o:
    print(f"  bi[{b.index}] {b.type} k={b.start.k.k_index}→{b.end.k.k_index} split={b.is_split}")

print(f"\ncl_pyarmor split bishu: {len(split_p)}")
for b in split_p:
    print(f"  bi[{b.index}] {b.type} k={b.start.k.k_index}→{b.end.k.k_index} split={b.is_split}")

# 2. 关注 bi[8] 附近：open 的分型和笔
print("\n\n--- bi[8] 附近的分型分析 ---")
fxs = co.get_fxs()
# 找 k_index 在 180-210 之间的分型
nearby_fxs = [f for f in fxs if 180 <= f.k.k_index <= 210]
print(f"FXs in k_index 180-210:")
for fx in nearby_fxs:
    print(f"  fx[{fx.index}] {fx.type:>4} k_idx={fx.k.k_index} val={fx.val:.2f} done={fx.done}")

# 3. 检查在 bi[6-12] 范围内的笔详细信息
print(f"\n--- open bis[6-12] ---")
for i in range(6, min(13, len(bis_o))):
    b = bis_o[i]
    span = b.end.k.k_index - b.start.k.k_index
    print(f"  bi[{i}] {b.type:>4} k={b.start.k.k_index:>4}→{b.end.k.k_index:>4} span={span:>3} h={b.high:.2f} l={b.low:.2f} split=[{b.is_split}]")

print(f"\n--- pyarmor bis[6-12] ---")
for i in range(6, min(13, len(bis_p))):
    b = bis_p[i]
    span = b.end.k.k_index - b.start.k.k_index
    print(f"  bi[{i}] {b.type:>4} k={b.start.k.k_index:>4}→{b.end.k.k_index:>4} span={span:>3} h={b.high:.2f} l={b.low:.2f} split=[{b.is_split}]")

# 4. 检查 k=187~199 之间的缠论K线，看看 open 是如何找到 k=192 作为笔端
print(f"\n--- CLKlines k_index 185-202 ---")
cl_klines = co.get_cl_klines()
for ck in cl_klines:
    if 185 <= ck.k_index <= 202:
        print(f"  ck[{ck.index}] k_idx={ck.k_index} h={ck.h:.2f} l={ck.l:.2f} n={ck.n}")

# 5. 检查 k=187 附近的 FX (start of divergent bi)
print(f"\n--- FXs around start of divergent bi ---")
start_fx_idx = None
for fx in fxs:
    if fx.k.k_index == 187:
        start_fx_idx = fx.index
        break
if start_fx_idx:
    for fx in fxs[max(0,start_fx_idx-2):start_fx_idx+8]:
        print(f"  fx[{fx.index}] {fx.type:>4} k_idx={fx.k.k_index} val={fx.val:.2f}")
