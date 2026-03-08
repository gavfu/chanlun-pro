# -*- coding: utf-8 -*-
"""诊断：对比 open vs pyarmor 笔的详细信息"""
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

print(f"=== 缠论K线: open={len(cd_o.get_cl_klines())} pyarmor={len(cd_p.get_cl_klines())}")
print(f"=== 分型: open={len(cd_o.get_fxs())} pyarmor={len(cd_p.get_fxs())}")

# 对比分型
print("\n--- 分型对比 (前20) ---")
o_fxs = cd_o.get_fxs()
p_fxs = cd_p.get_fxs()
for i in range(min(20, len(o_fxs), len(p_fxs))):
    of = o_fxs[i]
    pf = p_fxs[i]
    match = "✅" if (of.type == pf.type and of.k.index == pf.k.index) else "❌"
    print(f"  [{i}] open: {of.type} ck={of.k.index} val={of.val:.1f} | pyarmor: {pf.type} ck={pf.k.index} val={pf.val:.1f} {match}")

# 对比笔
print(f"\n=== 笔: open={len(cd_o.get_bis())} pyarmor={len(cd_p.get_bis())}")
print("\n--- pyarmor 笔 (前15) ---")
for i, bi in enumerate(cd_p.get_bis()[:15]):
    gap = bi.end.k.index - bi.start.k.index
    k_gap = bi.end.k.k_index - bi.start.k.k_index
    print(f"  bi[{i}] {bi.type:4s} ck:{bi.start.k.index:3d}->{bi.end.k.index:3d} gap={gap:2d} k_gap={k_gap:3d} h={bi.high:.1f} l={bi.low:.1f}")

print("\n--- open 笔 (前15) ---")
for i, bi in enumerate(cd_o.get_bis()[:15]):
    gap = bi.end.k.index - bi.start.k.index
    k_gap = bi.end.k.k_index - bi.start.k.k_index
    print(f"  bi[{i}] {bi.type:4s} ck:{bi.start.k.index:3d}->{bi.end.k.index:3d} gap={gap:2d} k_gap={k_gap:3d} h={bi.high:.1f} l={bi.low:.1f}")

# 分析：pyarmor 笔的 CLKline gap 最小值是多少？
p_gaps = [bi.end.k.index - bi.start.k.index for bi in cd_p.get_bis()]
o_gaps = [bi.end.k.index - bi.start.k.index for bi in cd_o.get_bis()]
print(f"\n--- 笔的 CLKline gap 统计 ---")
print(f"  pyarmor: min={min(p_gaps)} max={max(p_gaps)} avg={sum(p_gaps)/len(p_gaps):.1f}")
print(f"  open:    min={min(o_gaps)} max={max(o_gaps)} avg={sum(o_gaps)/len(o_gaps):.1f}")

p_k_gaps = [bi.end.k.k_index - bi.start.k.k_index for bi in cd_p.get_bis()]
o_k_gaps = [bi.end.k.k_index - bi.start.k.k_index for bi in cd_o.get_bis()]
print(f"\n--- 笔的原始K线 gap 统计 ---")
print(f"  pyarmor: min={min(p_k_gaps)} max={max(p_k_gaps)} avg={sum(p_k_gaps)/len(p_k_gaps):.1f}")
print(f"  open:    min={min(o_k_gaps)} max={max(o_k_gaps)} avg={sum(o_k_gaps)/len(o_k_gaps):.1f}")

# 逐笔对比
print("\n--- 逐笔详细对比 ---")
o_bis = cd_o.get_bis()
p_bis = cd_p.get_bis()
for i in range(min(len(o_bis), len(p_bis))):
    ob = o_bis[i]
    pb = p_bis[i]
    s_match = "✅" if ob.start.k.index == pb.start.k.index else "❌"
    e_match = "✅" if ob.end.k.index == pb.end.k.index else "❌"
    t_match = "✅" if ob.type == pb.type else "❌"
    print(f"  bi[{i:2d}] type:{t_match} start:{s_match} end:{e_match} | open: {ob.type:4s} {ob.start.k.index:3d}->{ob.end.k.index:3d} h={ob.high:.1f} l={ob.low:.1f} | pyarmor: {pb.type:4s} {pb.start.k.index:3d}->{pb.end.k.index:3d} h={pb.high:.1f} l={pb.low:.1f}")
    if ob.start.k.index != pb.start.k.index or ob.end.k.index != pb.end.k.index:
        # 分析起始分型和结束分型的差异
        print(f"         open  start_fx: {ob.start.type} ck={ob.start.k.index} val={ob.start.val:.1f}")
        print(f"         pyarm start_fx: {pb.start.type} ck={pb.start.k.index} val={pb.start.val:.1f}")
        print(f"         open  end_fx:   {ob.end.type} ck={ob.end.k.index} val={ob.end.val:.1f}")
        print(f"         pyarm end_fx:   {pb.end.type} ck={pb.end.k.index} val={pb.end.val:.1f}")

# 检查 ck=40~60 的完整分型序列
print("\n--- 完整分型序列 ck=40~60 ---")
qj = cd_o.fx_qj
qy = cd_o.fx_qy
for fx in o_fxs:
    if 40 <= fx.k.index <= 60:
        print(f"  fx ck={fx.k.index:3d} {fx.type:4s} val={fx.val:.1f} k.h={fx.k.h:.1f} k.l={fx.k.l:.1f}")

# 从 ding ck=44 出发检查哪些 di 满足条件
print("\n--- 从 ding ck=44 出发检查 di ---")
fx_44 = [fx for fx in o_fxs if fx.k.index == 44]
if fx_44:
    sf = fx_44[0]
    for fx in o_fxs:
        if fx.k.index > 44 and fx.k.index <= 60 and fx.type == "di":
            cl_gap = fx.k.index - sf.k.index
            k_gap = fx.k.k_index - sf.k.k_index
            valid = cd_o._bi_fx_valid(sf, fx)
            print(f"    di ck={fx.k.index:3d} val={fx.val:.1f} cl_gap={cl_gap} k_gap={k_gap} valid={valid}")

# 同时检查 ck=138~160
print("\n--- 完整分型序列 ck=135~160 ---")
for fx in o_fxs:
    if 135 <= fx.k.index <= 160:
        print(f"  fx ck={fx.k.index:3d} {fx.type:4s} val={fx.val:.1f} k.h={fx.k.h:.1f} k.l={fx.k.l:.1f}")

# 检查 pyarmor 配置
print(f"\n--- pyarmor 配置 ---")
p_cfg = cd_p.get_config()
for key in sorted(p_cfg.keys()):
    if key.startswith(("bi_", "fx_", "allow_")):
        print(f"  {key}: {p_cfg[key]}")

# ===== bi[14] 详细追踪 =====
print("\n" + "="*60)
print("===== bi[21] 详细追踪：从 di ck=262 开始 =====")
print("="*60)

# 列出 ck=260~280 范围内所有分型
print("\n--- 分型序列 ck=260~285 ---")
qj = cd_o.fx_qj
qy = cd_o.fx_qy
for fx in o_fxs:
    if 260 <= fx.k.index <= 285:
        h3 = fx.high(qj, qy)
        l3 = fx.low(qj, qy)
        print(f"  fx ck={fx.k.index:3d} k_idx={fx.k.k_index:3d} {fx.type:4s} val={fx.val:.1f} k.h={fx.k.h:.1f} k.l={fx.k.l:.1f} | high3={h3:.1f} low3={l3:.1f}")

# 从 di ck=262 出发检查各分型
print("\n--- 从 di ck=262 模拟笔构建 ---")
fx_262 = [fx for fx in o_fxs if fx.k.index == 262][0]
print(f"start_fx: {fx_262.type} ck={fx_262.k.index} val={fx_262.val:.1f}")

following_fxs = [fx for fx in o_fxs if fx.k.index > 262 and fx.k.index <= 285]
for fx in following_fxs:
    valid = cd_o._bi_fx_valid(fx_262, fx)
    cl_gap = fx.k.index - fx_262.k.index
    k_gap = fx.k.k_index - fx_262.k.k_index
    h3 = fx.high(qj, qy)
    l3 = fx.low(qj, qy)
    print(f"  check: {fx.type:4s} ck={fx.k.index:3d} val={fx.val:.1f} cl_gap={cl_gap} k_gap={k_gap} high3={h3:.1f} low3={l3:.1f} valid={valid}")

# Check confirmation: ding ck=267 → di ck=?
print("\n--- 候选确认检查 from ding ck=267 ---")
fx_267 = [fx for fx in o_fxs if fx.k.index == 267]
if fx_267:
    fx_267 = fx_267[0]
    for fx in o_fxs:
        if fx.k.index > 267 and fx.k.index <= 280 and fx.type == "di":
            valid = cd_o._bi_fx_valid(fx_267, fx)
            cl_gap = fx.k.index - fx_267.k.index
            k_gap = fx.k.k_index - fx_267.k.k_index
            # Check CGD: is there a ding between 267 and fx that's higher (for "di" confirmation)?
            # Wait, cur_fx is di, so check if between end_fx(ding 267) and cur_fx(di) there's a di lower than cur_fx
            # Actually for confirmation, we check end_fx → cur_fx. end_fx=ding, cur_fx must be di
            # CGD check: between end_fx and cur_fx, is there a same-type (di) fx that's more extreme (lower)?
            cgd_block = False
            end_idx_here = [idx for idx, f in enumerate(o_fxs) if f.k.index == 267][0]
            cur_idx_here = [idx for idx, f in enumerate(o_fxs) if f.k.index == fx.k.index][0]
            for j in range(end_idx_here + 1, cur_idx_here):
                mid = o_fxs[j]
                if mid.type == fx.type:  # same type as confirmation fx
                    if fx.type == "di" and mid.val < fx.val:
                        cgd_block = True
                        print(f"    CGD block: mid di ck={mid.k.index} val={mid.val:.1f} < cur di ck={fx.k.index} val={fx.val:.1f}")
                        break
                    elif fx.type == "ding" and mid.val > fx.val:
                        cgd_block = True
                        print(f"    CGD block: mid ding ck={mid.k.index} val={mid.val:.1f} > cur ding ck={fx.k.index} val={fx.val:.1f}")
                        break
            print(f"  di ck={fx.k.index:3d} val={fx.val:.1f} cl={cl_gap} k={k_gap} valid={valid} cgd_block={cgd_block}")
