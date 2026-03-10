"""Check: for each pyarmor BI with cl_gap<4, trace the _build_bis scan to see
WHY our algorithm misses it. Is it the gap check OR is it the algorithm flow?"""
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

# The small cl_gap pyarmor BIs, but let's look at them in context
# For now focus on BTC60 and ETH5m since those are the ones with the first divergences

# BTC60: first divergence is at bi[15] which IS one of these small cl_gap BIs
# ETH5m: first divergence is at bi[26] which has cl_gap=4 k_gap=4 (NOT small cl_gap!)

# Let me re-verify: what's pyarmor's ETH5m bi[26]?
df = pd.read_parquet("tests/test_data/ETH_USDT_5m_1000.parquet")
cd_o = CL_O("test", "test", config=CL_CONFIG)
cd_o.process_klines(df)
cd_p = CL_P("test", "test", config=CL_CONFIG)
cd_p.process_klines(df)

print("=== ETH5m: Pyarmor's BIs around first divergence ===")
for bi in cd_p.get_bis()[24:30]:
    cl = bi.end.k.index - bi.start.k.index
    k = bi.end.k.k_index - bi.start.k.k_index
    print(f"  bi[{bi.index}] {bi.type} {bi.start.k.k_index}→{bi.end.k.k_index} cl={cl} k={k}")

print("\n=== ETH5m: Open's BIs ===")
for bi in cd_o.get_bis()[24:30]:
    cl = bi.end.k.index - bi.start.k.index
    k = bi.end.k.k_index - bi.start.k.k_index
    print(f"  bi[{bi.index}] {bi.type} {bi.start.k.k_index}→{bi.end.k.k_index} cl={cl} k={k}")

# Pyarmor ETH5m bi[26] = down 333→337, cl=4, k=4
# Open ETH5m bi[26] = down 333→349, cl=13, k=16
# Both have cl_gap >= 4 for the primary BI. The difference is in confirmation!

# In open: after setting end_fx = di@337, confirmation check happens at ding@339, ding@341, ding@344...
# All fail either gap or strict, so the algorithm continues extending.
# Eventually end_fx extends to di@349 (same val, replaces di@337)
# Then ding@352 or ding@357 confirms.

# In pyarmor: bi[26] = down 333→337, bi[27] = up 337→362
# So pyarmor's confirmation from di@337 is at ding@???. 
# bi[27] = up 337→362, so the confirmation FX is the one that confirmed bi[26].
# The confirmation checks: di@337 → ding@339 (cl=1,k=2), ding@341 (cl=3,k=4), ding@344 (cl=6,k=7)...
# In our analysis: ding@339 cl=1 strict=False, ding@341 cl=3 strict=False, ding@344 cl=6 strict=False
# ALL fail strict! Yet pyarmor confirms.

# Wait - let me look at this from the CONFIRMATION side.
# If pyarmor uses k_gap for confirmation gap check:
# di@337 → ding@341: k_gap=4, strict check for UP: 
#   di337.high > ding341.high? or ding341.low < di337.low?
fxs = cd_o.get_fxs()
qj = cd_o.fx_qj
qy = cd_o.fx_qy
fx_map = {}
for fx in fxs:
    fx_map[fx.k.k_index] = fx

print("\n=== Confirmation from di@337 (ETH5m) ===")
di337 = fx_map[337]
for ding_k in [339, 341, 344, 352, 357]:
    if ding_k not in fx_map:
        continue
    ding = fx_map[ding_k]
    cl_gap = ding.k.index - di337.k.index
    k_gap = ding.k.k_index - di337.k.k_index
    
    # Up BI strict: di337 → ding_X
    c1 = di337.high(qj, qy) > ding.high(qj, qy)
    c2 = ding.low(qj, qy) < di337.low(qj, qy)
    strict = not c1 and not c2
    
    print(f"  di@337 → ding@{ding_k}: cl={cl_gap} k={k_gap} strict={strict}")
    if not strict:
        if c1: print(f"    C1 FAIL: di337.high={di337.high(qj,qy):.2f} > ding.high={ding.high(qj,qy):.2f}")
        if c2: print(f"    C2 FAIL: ding.low={ding.low(qj,qy):.2f} < di337.low={di337.low(qj,qy):.2f}")

# Now check: what if the strict check uses "right half" for start, "all" for end?
print("\n=== Confirmation from di@337 with right-half strict ===")
for ding_k in [339, 341, 344, 352, 357]:
    if ding_k not in fx_map:
        continue
    ding = fx_map[ding_k]
    cl_gap = ding.k.index - di337.k.index
    k_gap = ding.k.k_index - di337.k.k_index
    
    # Use right-half of start (di337): klines[1:]
    klines_right = [ck for ck in di337.klines[1:] if ck is not None]
    sh = max(k.h for ck in klines_right for k in ck.klines)
    sl = min(k.l for ck in klines_right for k in ck.klines)
    
    # Use all of end (ding)
    eh = ding.high(qj, qy)
    el = ding.low(qj, qy)
    
    c1 = sh > eh  # start.high > end.high
    c2 = el < sl   # end.low < start.low
    strict = not c1 and not c2
    
    print(f"  di@337 → ding@{ding_k}: cl={cl_gap} k={k_gap} strict_right={strict}")
    if not strict:
        if c1: print(f"    C1 FAIL: start_right.high={sh:.2f} > ding.high={eh:.2f}")
        if c2: print(f"    C2 FAIL: ding.low={el:.2f} < start_right.low={sl:.2f}")

# Also check what di@337's klines look like
print("\n=== di@337 FX structure ===")
for i, ck in enumerate(di337.klines):
    if ck is not None:
        print(f"  klines[{i}]: ck_idx={ck.index} h={ck.h:.2f} l={ck.l:.2f}")
        for sk in ck.klines:
            print(f"    raw: idx={sk.index} h={sk.h:.2f} l={sk.l:.2f}")
