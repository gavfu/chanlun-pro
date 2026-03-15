"""
诊断 BTC5m pre-split bi[59]: open=up 875→879, pyarmor=up 875→889
追踪 _build_bis 的 FX 扫描过程，找出分歧点。
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

df = pd.read_parquet('tests/test_data/BTC_USDT_5m_1000.parquet')

co = CL_O("test", "test", config=CL_CONFIG)
co.process_klines(df)
cp = CL_P("test", "test", config=CL_CONFIG)
cp.process_klines(df)

fxs_o = co.get_fxs()
fxs_p = cp.get_fxs()

# Verify fxs match
fx_match = len(fxs_o) == len(fxs_p) and all(
    fo.type == fp.type and fo.k.k_index == fp.k.k_index
    for fo, fp in zip(fxs_o, fxs_p)
)
print(f"分型完全一致: {fx_match} (open={len(fxs_o)}, pyarmor={len(fxs_p)})")

# Find the start FX for bi[59] (di at k=875)
# Both agree on pre-split bi[58] being down ending at k=875
# So start_fx is di at k=875
start_fx_idx = None
for idx, fx in enumerate(fxs_o):
    if fx.k.k_index == 875 and fx.type == "di":
        start_fx_idx = idx
        break

start_fx = fxs_o[start_fx_idx]
print(f"\nstart_fx: fx[{start_fx_idx}] {start_fx.type} k_idx={start_fx.k.k_index} val={start_fx.val:.2f}")
print(f"  CLK index: {start_fx.k.index}")

# Scan reverse-direction FXs (ding) from start
print(f"\n--- 从 start_fx 后扫描分型 ---")
qj, qy = co.fx_qj, co.fx_qy
for j in range(start_fx_idx + 1, min(start_fx_idx + 20, len(fxs_o))):
    fx = fxs_o[j]
    
    # 计算gap
    cl_gap = fx.k.index - start_fx.k.index
    k_gap = fx.k.k_index - start_fx.k.k_index
    
    if fx.type == start_fx.type:
        print(f"  fx[{j}] {fx.type:>4} k_idx={fx.k.k_index:>4} val={fx.val:.2f} (同向)")
        continue
    
    # 反向 (ding) - bi_fx_valid check
    valid = co._bi_fx_valid(start_fx, fx)
    
    # 详细检查
    detail = ""
    if cl_gap < 4:
        detail = f"GAP_FAIL: cl_gap={cl_gap}<4"
    elif k_gap < co.fx_check_k_nums and co.allow_bi_fx_strict:
        if start_fx.type == "di" and fx.type == "ding":
            s_high = start_fx.high(qj, qy)
            e_high = fx.high(qj, qy)
            e_low = fx.low(qj, qy)
            s_low = start_fx.low(qj, qy)
            if s_high > e_high:
                detail = f"STRICT_FAIL: start.high({s_high:.2f}) > end.high({e_high:.2f})"
            elif e_low < s_low:
                detail = f"STRICT_FAIL: end.low({e_low:.2f}) < start.low({s_low:.2f})"
            else:
                detail = f"strict_ok: s_high={s_high:.2f}<=e_high={e_high:.2f}, e_low={e_low:.2f}>=s_low={s_low:.2f}"
    else:
        detail = "no_strict (k_gap >= 13)"
    
    marker = ""
    if fx.k.k_index == 879:
        marker = " ← OPEN_END"
    if fx.k.k_index == 889:
        marker += " ← PYARMOR_END"
    
    print(f"  fx[{j}] {fx.type:>4} k_idx={fx.k.k_index:>4} val={fx.val:.2f} cl_gap={cl_gap} k_gap={k_gap} valid={valid} {detail}{marker}")

# Also check: what does pyarmor do with bi around this area?
print(f"\n--- pyarmor bis around k=875 ---")
bis_p = cp.get_bis()
for bi in bis_p:
    if 860 <= bi.start.k.k_index <= 920 or 860 <= bi.end.k.k_index <= 920:
        print(f"  {bi.type:>4} k={bi.start.k.k_index:>4}→{bi.end.k.k_index:>4} is_split='{bi.is_split}'")

print(f"\n--- cl_open bis around k=875 ---")
bis_o = co.get_bis()
for bi in bis_o:
    if 860 <= bi.start.k.k_index <= 920 or 860 <= bi.end.k.k_index <= 920:
        print(f"  {bi.type:>4} k={bi.start.k.k_index:>4}→{bi.end.k.k_index:>4} is_split='{bi.is_split}'")

# Also check the _build_bis candidate behavior
# In _build_bis, at the start_fx=di(875), scanning for candidate end_fx (ding)
# Simulate the state machine from bi[58] end / bi[59] start
print(f"\n--- _build_bis state machine trace ---")
print("Using _build_bis logic to trace from start_fx=di(875)...")

# Check if there's a same-direction FX before the first valid reverse FX
# that could affect candidate selection
for j in range(start_fx_idx + 1, min(start_fx_idx + 20, len(fxs_o))):
    fx = fxs_o[j]
    cl_gap = fx.k.index - start_fx.k.index
    k_gap = fx.k.k_index - start_fx.k.k_index
    
    if fx.type != start_fx.type:
        # reverse: check _bi_fx_valid
        valid = co._bi_fx_valid(start_fx, fx)
        if valid:
            print(f"  FIRST VALID reverse: fx[{j}] {fx.type} k_idx={fx.k.k_index} → this becomes end_fx candidate")
            
            # Now simulate: after setting end_fx, what happens with the next FXs?
            end_fx = fx
            end_idx = j
            for k in range(j + 1, min(j + 15, len(fxs_o))):
                nfx = fxs_o[k]
                if nfx.type == end_fx.type:
                    # Same as end_fx - check extension
                    extend = False
                    if end_fx.type == "di" and nfx.val <= end_fx.val:
                        if co._bi_fx_valid(start_fx, nfx):
                            extend = True
                    elif end_fx.type == "ding" and nfx.val >= end_fx.val:
                        if co._bi_fx_valid(start_fx, nfx):
                            extend = True
                    if extend:
                        print(f"    fx[{k}] {nfx.type} k_idx={nfx.k.k_index} val={nfx.val:.2f} → EXTEND end_fx")
                        end_fx = nfx
                        end_idx = k
                    else:
                        print(f"    fx[{k}] {nfx.type} k_idx={nfx.k.k_index} val={nfx.val:.2f} → skip (not better or invalid)")
                else:
                    # Reverse of end_fx → check confirmation
                    confirm = co._bi_fx_valid(end_fx, nfx)
                    cl_gap2 = nfx.k.index - end_fx.k.index
                    k_gap2 = nfx.k.k_index - end_fx.k.k_index
                    if confirm:
                        print(f"    fx[{k}] {nfx.type} k_idx={nfx.k.k_index} val={nfx.val:.2f} → CONFIRM bi: {start_fx.k.k_index}→{end_fx.k.k_index}")
                        break
                    else:
                        print(f"    fx[{k}] {nfx.type} k_idx={nfx.k.k_index} val={nfx.val:.2f} → confirm_fail cl_gap={cl_gap2} k_gap={k_gap2}")
            break
