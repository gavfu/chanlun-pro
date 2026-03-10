"""Check if pyarmor's bi[14] also differs from baseline, causing different scan state at 299."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

CL_CONFIG = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
}

df = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cd_o = CL_O("test", "test", config=CL_CONFIG)
cd_o.process_klines(df)
cd_p = CL_P("test", "test", config=CL_CONFIG)
cd_p.process_klines(df)

bis_o = cd_o.get_bis()
bis_p = cd_p.get_bis()

print("=== ALL BIs comparison ===")
for j in range(max(len(bis_o), len(bis_p))):
    bo = bis_o[j] if j < len(bis_o) else None
    bp = bis_p[j] if j < len(bis_p) else None
    bo_s = f"{bo.type:4s} {bo.start.k.k_index:4d}→{bo.end.k.k_index:4d} cl={bo.end.k.index-bo.start.k.index:2d}" if bo else "---"
    bp_s = f"{bp.type:4s} {bp.start.k.k_index:4d}→{bp.end.k.k_index:4d} cl={bp.end.k.index-bp.start.k.index:2d}" if bp else "---"
    match = "✓" if bo and bp and bo.start.k.k_index == bp.start.k.k_index and bo.end.k.k_index == bp.end.k.k_index else "✗"
    if not (bo and bp and bo.start.k.k_index == bp.start.k.k_index and bo.end.k.k_index == bp.end.k.k_index):
        print(f"  [{j:2d}] {match} open={bo_s}  pya={bp_s}")
    
# Check: before bi[15], are all BIs identical?
print(f"\n=== Match summary ===")
first_diff = -1
for j in range(min(len(bis_o), len(bis_p))):
    bo = bis_o[j]; bp = bis_p[j]
    if bo.start.k.k_index != bp.start.k.k_index or bo.end.k.k_index != bp.end.k.k_index:
        if first_diff < 0:
            first_diff = j
            print(f"  First difference at bi[{j}]")
        
print(f"  BIs 0-{first_diff-1}: ALL MATCH" if first_diff > 0 else "  No differences")

# Now check: what's the scan state at the start of bi[15]?
# Both algorithms should have just finished bi[14] = down 291→299
# Then start_fx = di@299
# Then scan for ding candidates.
# 
# In open: di@299→ding@304 has cl_gap=1, FAILS gap check → skipped
# Then di@299→ding@308: cl_gap=4, passes gap. Check strict:
# Let me verify this in detail

fxs = cd_o.get_fxs()
qj = cd_o.fx_qj; qy = cd_o.fx_qy

# After bi[14] = down 291→299, start_fx = di@299
# Scan from di@299 onward
di299 = None
for fx in fxs:
    if fx.k.k_index == 299 and fx.type == "di": di299 = fx; break

print(f"\n=== After bi[14]: scan from di@299 ===")
for fx in fxs:
    if fx.type == "ding" and fx.k.k_index > 299 and fx.k.k_index <= 340:
        cl = fx.k.index - di299.k.index
        k = fx.k.k_index - di299.k.k_index
        h_s = di299.high(qj, qy); l_s = di299.low(qj, qy)
        h_e = fx.high(qj, qy); l_e = fx.low(qj, qy)
        c1 = h_s > h_e; c2 = l_e < l_s
        strict_fail = (c1 or c2) if k < 13 else False
        print(f"  ding@{fx.k.k_index}: cl={cl} k={k} "
              f"gap_cl={cl>=4} strict={'FAIL' if strict_fail else 'PASS'}"
              f" C1={c1} C2={c2}")

# The pyarmor scan from di@299:
# di@299→ding@304: cl=1 → FAILS cl_gap
# di@299→ding@308: cl=4 → passes gap. Strict: k=9 < 13
#   C1: di299.high=70274.5 > ding308.high=69887.7 → True → FAIL
# di@299→ding@312: cl=6 → passes gap. k=13 ≥ 13 → bypass strict → PASS!
#   But wait, ding@308 has higher val (69887.7) than ding@304 (70000)
#   Actually wait: for UP BI, we want the HIGHEST ding.

# So baseline should:
# 1. Find first valid candidate: ding@312 (cl=6, k=13, strict bypassed)
# Actually no, let me re-check ding@308:
print(f"\n=== Check ding@308 strict detail ===")
ding308 = None
for fx in fxs:
    if fx.k.k_index == 308 and fx.type == "ding": ding308 = fx; break

if di299 and ding308:
    cl = ding308.k.index - di299.k.index
    k = ding308.k.k_index - di299.k.k_index
    h_s = di299.high(qj, qy); l_s = di299.low(qj, qy)
    h_e = ding308.high(qj, qy); l_e = ding308.low(qj, qy)
    c1 = h_s > h_e; c2 = l_e < l_s
    print(f"  cl={cl}, k={k}")
    print(f"  di299.h={h_s:.2f}, l={l_s:.2f}")
    print(f"  ding308.h={h_e:.2f}, l={l_e:.2f}")
    print(f"  C1={c1} (start.h > end.h), C2={c2} (end.l < start.l)")
    print(f"  strict={'FAIL' if c1 or c2 else 'PASS'}")
    
    # With right-half for start:
    di299_h_rh = max([rk.h for ck in di299.klines[1:] for rk in ck.klines])
    di299_l_rh = min([rk.l for ck in di299.klines[1:] for rk in ck.klines])
    c1r = di299_h_rh > ding308.high(qj, qy)
    c2r = ding308.low(qj, qy) < di299_l_rh
    print(f"  Right-half start: h_rh={di299_h_rh:.2f}")
    print(f"  C1_rh={c1r}, C2_rh={c2r} → {'FAIL' if c1r or c2r else 'PASS'}")
