# -*- coding: utf-8 -*-
"""追踪 _build_bis 算法的完整执行过程，找出 bi[14] 分歧原因"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor
from chanlun.cl_interface import Config, FX, BI

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "test_data", "BTC_USDT_60m_500.parquet"))

cd_p = CL_Pyarmor("BTC/USDT", "60m")
cd_p.process_klines(df)

cd_o = CL_Open("BTC/USDT", "60m")
cd_o.process_klines(df)

# Pyarmor strokes
p_bis = cd_p.get_bis()
print(f"pyarmor strokes: {len(p_bis)}")
for i, bi in enumerate(p_bis):
    print(f"  bi[{i:2d}] {bi.type:4s} ck:{bi.start.k.index:3d}->{bi.end.k.index:3d}")

print()

# Now manually trace the _build_bis algorithm with verbose logging
fxs = cd_o.get_fxs()
qj = cd_o.fx_qj
qy = cd_o.fx_qy

print(f"\n===== 手动追踪 _build_bis (从 bi[13] 的起点 ding ck=138 之后) =====")

# Find the start of bi[13] in our result: up 138->149
# Actually let's trace from bi[12] end = di 138 = bi[13] start
# Find index of di ck=138 in fxs
fx_map = {fx.k.index: fx for fx in fxs}

# Start tracing after bi[12] is confirmed correctly (ck=113→138)
# So start_fx = di ck=138, scanning for end_fx
trace_start_ck = 262

# Find fx index
start_fx_idx = None
for idx, fx in enumerate(fxs):
    if fx.k.index == trace_start_ck:
        start_fx_idx = idx
        break

print(f"Starting trace from fx[{start_fx_idx}]: {fxs[start_fx_idx].type} ck={fxs[start_fx_idx].k.index}")
print()

# Simulate algorithm from this point
start_fx = fxs[start_fx_idx]
start_idx = start_fx_idx
end_fx = None
end_idx = -1
bis = []

i = start_fx_idx + 1
step = 0
while i < len(fxs) and len(bis) < 10:  # trace just a few strokes
    cur_fx = fxs[i]
    step += 1
    
    if end_fx is None:
        if cur_fx.type == start_fx.type:
            if start_fx.type == "ding" and cur_fx.val > start_fx.val:
                print(f"  step{step}: ck={cur_fx.k.index} {cur_fx.type} val={cur_fx.val:.1f} → 同类更优 → 更新start为 {cur_fx.type} ck={cur_fx.k.index}")
                start_fx = cur_fx
                start_idx = i
            elif start_fx.type == "di" and cur_fx.val < start_fx.val:
                print(f"  step{step}: ck={cur_fx.k.index} {cur_fx.type} val={cur_fx.val:.1f} → 同类更优 → 更新start为 {cur_fx.type} ck={cur_fx.k.index}")
                start_fx = cur_fx
                start_idx = i
            else:
                print(f"  step{step}: ck={cur_fx.k.index} {cur_fx.type} val={cur_fx.val:.1f} → 同类不优 → 跳过 (start={start_fx.type} ck={start_fx.k.index})")
        else:
            valid = cd_o._bi_fx_valid(start_fx, cur_fx)
            cl_gap = cur_fx.k.index - start_fx.k.index
            k_gap = cur_fx.k.k_index - start_fx.k.k_index
            if valid:
                end_fx = cur_fx
                end_idx = i
                print(f"  step{step}: ck={cur_fx.k.index} {cur_fx.type} val={cur_fx.val:.1f} → 反类有效(cl={cl_gap},k={k_gap}) → 设为候选end_fx")
            else:
                print(f"  step{step}: ck={cur_fx.k.index} {cur_fx.type} val={cur_fx.val:.1f} → 反类无效(cl={cl_gap},k={k_gap}) → 跳过")
        i += 1
    else:
        if cur_fx.type == end_fx.type:
            # Same type as end_fx → try extend
            better = False
            if end_fx.type == "di" and cur_fx.val <= end_fx.val:
                better = True
            elif end_fx.type == "ding" and cur_fx.val >= end_fx.val:
                better = True
            
            if better:
                valid = cd_o._bi_fx_valid(start_fx, cur_fx)
                cl_gap = cur_fx.k.index - start_fx.k.index
                k_gap = cur_fx.k.k_index - start_fx.k.k_index
                if valid:
                    print(f"  step{step}: ck={cur_fx.k.index} {cur_fx.type} val={cur_fx.val:.1f} → 同类更优&有效(cl={cl_gap},k={k_gap}) → 延伸end_fx")
                    end_fx = cur_fx
                    end_idx = i
                else:
                    print(f"  step{step}: ck={cur_fx.k.index} {cur_fx.type} val={cur_fx.val:.1f} → 同类更优但无效(cl={cl_gap},k={k_gap}) → 跳过")
            else:
                print(f"  step{step}: ck={cur_fx.k.index} {cur_fx.type} val={cur_fx.val:.1f} → 同类不优 → 跳过 (end_fx={end_fx.type} ck={end_fx.k.index} val={end_fx.val:.1f})")
            i += 1
        else:
            # Opposite type → try confirm
            valid = cd_o._bi_fx_valid(end_fx, cur_fx)
            cl_gap_end_to_cur = cur_fx.k.index - end_fx.k.index
            k_gap_end_to_cur = cur_fx.k.k_index - end_fx.k.k_index
            
            # Also check strict details
            strict_info = ""
            if k_gap_end_to_cur < cd_o.fx_check_k_nums and cd_o.allow_bi_fx_strict:
                if end_fx.type == "ding" and cur_fx.type == "di":
                    s_low = end_fx.low(qj, qy)
                    e_low = cur_fx.low(qj, qy)
                    strict_info = f" strict: end.low3={s_low:.1f} vs cur.low3={e_low:.1f} block={s_low < e_low}"
                elif end_fx.type == "di" and cur_fx.type == "ding":
                    s_high = end_fx.high(qj, qy)
                    e_high = cur_fx.high(qj, qy)
                    strict_info = f" strict: end.high3={s_high:.1f} vs cur.high3={e_high:.1f} block={s_high > e_high}"
            elif k_gap_end_to_cur >= cd_o.fx_check_k_nums:
                strict_info = f" (k_gap≥{cd_o.fx_check_k_nums}, strict bypassed)"
            
            if valid:
                bi_type = "down" if start_fx.type == "ding" else "up"
                print(f"  step{step}: ck={cur_fx.k.index} {cur_fx.type} val={cur_fx.val:.1f} → ★确认笔★ {bi_type} {start_fx.k.index}->{end_fx.k.index} (confirm: cl={cl_gap_end_to_cur},k={k_gap_end_to_cur} valid=True{strict_info})")
                bi = BI(start=start_fx, end=end_fx, _type=bi_type, index=len(bis), default_zs_type=cd_o.default_bi_zs_type)
                bis.append(bi)
                start_fx = end_fx
                start_idx = end_idx
                end_fx = None
                end_idx = -1
                i = start_idx + 1
                print(f"         → 新start: {start_fx.type} ck={start_fx.k.index}, 重新从 fx_idx={start_idx+1} 开始")
            else:
                print(f"  step{step}: ck={cur_fx.k.index} {cur_fx.type} val={cur_fx.val:.1f} → 确认失败(cl={cl_gap_end_to_cur},k={k_gap_end_to_cur}){strict_info} → 跳过")
                i += 1

print(f"\n--- 追踪产生的笔 ---")
for i, bi in enumerate(bis):
    print(f"  bi[{i}] {bi.type:4s} {bi.start.k.index:3d}->{bi.end.k.index:3d}")
