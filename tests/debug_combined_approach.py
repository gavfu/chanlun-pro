"""Reverse engineer pyarmor's _bi_fx_valid by testing if specific BI constructions
are accepted. Strategy: build test cases where we pass an FX sequence with exactly 2 FXes
and see if pyarmor constructs a BI from them."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P
from chanlun.cl_interface import Config

CL_CONFIG = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
}

df = pd.read_parquet("tests/test_data/ETH_USDT_5m_1000.parquet")

# Create open instance to get FXes
cd_o = CL_O("test", "test", config=CL_CONFIG)
cd_o.process_klines(df)
fxs = cd_o.get_fxs()

# Create pyarmor instance normally
cd_p = CL_P("test", "test", config=CL_CONFIG)
cd_p.process_klines(df)
bis_p = cd_p.get_bis()

qj = cd_o.fx_qj
qy = cd_o.fx_qy

# For the key pair: di@337 → ding@344
# Check if this is part of a confirming BI or a primary BI in pyarmor
print("=== Pyarmor BI chain around bi[26] ===")
for bi in bis_p[24:30]:
    cl_gap = bi.end.k.index - bi.start.k.index
    k_gap = bi.end.k.k_index - bi.start.k.k_index
    print(f"  bi[{bi.index}] {bi.type} k={bi.start.k.k_index}→{bi.end.k.k_index} cl_gap={cl_gap} k_gap={k_gap}")

# Pyarmor bi[26] = down k=333→337, bi[27] = up k=337→362
# So the CONFIRMING BI for bi[26] is bi[27] = up from di@337 to ding@362
# ding@344 is NOT itself the confirmation endpoint - ding@362 is
# But for confirmation, we only need the FIRST valid continuation from di@337

# Wait - the algorithm works differently:
# When scanning, we have end_fx=di@337 and then look for confirmation.
# Confirmation means ANY valid BI starting from di@337.
# The first ding after di@337 is ding@339 (cl_gap=1, rejected).
# Then ding@341 (cl_gap=3, rejected).
# Then ding@344 (cl_gap=6, k_gap=7 - strict check fails in open).
# In pyarmor, ding@344 must pass (because bi[27] starts at 337 and goes to 362).
# But ding@344 doesn't need to be the END of the confirming BI - just needs to pass
# the _bi_fx_valid check.

# Let me check: in pyarmor's bi list, bi[27] goes from 337 to 362.
# This means the first ACCEPTED confirmation was some ding where _bi_fx_valid(di@337, ding@X) = True.
# What is first ding X after di@337 where pyarmor accepts?

# Let me check ALL dings after 337 and before 362 in pyarmor's own FX list
fxs_p = cd_p.get_fxs()
print("\n=== Pyarmor FXes after k=337, before k=362 ===")
for fx in fxs_p:
    if 337 < fx.k.k_index < 365 and fx.type == "ding":
        print(f"  ding@{fx.k.k_index} val={fx.val:.2f} "
              f"high={fx.high(qj,qy):.2f} low={fx.low(qj,qy):.2f}")

# The confirmation check in pyarmor's _build_bis: 
# - end_fx = di@337 (candidate bottom)
# - Scan forward for opposite type (ding)
# - First ding that passes _bi_fx_valid(di@337, ding@X) confirms the current BI

# So pyarmor must have a _bi_fx_valid that accepts di@337→ding@344 
# (or some earlier ding that we missed, or it uses a different algorithm entirely)

# Let me trace: what is the FIRST ding after di@337 in BOTH FX lists?
print("\n=== Open: dings after di@337 ===")
for fx in fxs:
    if fx.k.k_index > 337 and fx.type == "ding" and fx.k.k_index < 365:
        cl_gap = fx.k.index - [f for f in fxs if f.type == "di" and f.k.k_index == 337][0].k.index
        k_gap = fx.k.k_index - 337
        valid = cd_o._bi_fx_valid([f for f in fxs if f.type == "di" and f.k.k_index == 337][0], fx)
        print(f"  ding@{fx.k.k_index} val={fx.val:.2f} cl_gap={cl_gap} k_gap={k_gap} valid={valid}")

# Now check if pyarmor's bi[27] boundaries actually work
# bi[27] is up from k=337 to k=362
# The fact that pyarmor creates this BI means pyarmor's algorithm found an end_fx=ding@362
# AND got it confirmed. But the first step is finding end_fx.
# For up BI from di@337, we need _bi_fx_valid(di@337, ding@X) = True for some X.
# The FIRST valid ding X is the end_fx candidate.

# Actually wait - pyarmor might NOT need confirmation for the BI before.
# In pyarmor, bi[26] = down 333→337 with bi[27] = up 337→362
# For bi[26] to be confirmed, we need the NEXT BI (up from 337) to be valid.
# The confirmation check is: _bi_fx_valid(di@337, ding@X) for some X after 337.

# Let me also check: what if pyarmor's _build_bis doesn't require confirmation 
# for BIs that have a large enough interval? Or what if the confirmation uses
# a different threshold?

# Let me trace the BTC60 case too
print("\n\n=== BTC60 Case ===")
df2 = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cd_o2 = CL_O("test", "test", config=CL_CONFIG)
cd_o2.process_klines(df2)
cd_p2 = CL_P("test", "test", config=CL_CONFIG)
cd_p2.process_klines(df2)

bis_p2 = cd_p2.get_bis()
print("Pyarmor BIs [14:18]:")
for bi in bis_p2[14:18]:
    cl_gap = bi.end.k.index - bi.start.k.index
    k_gap = bi.end.k.k_index - bi.start.k.k_index
    print(f"  bi[{bi.index}] {bi.type} k={bi.start.k.k_index}→{bi.end.k.k_index} cl_gap={cl_gap} k_gap={k_gap}")

# Pyarmor bi[15] = up k=299→304, bi[16] = down k=304→309
# For bi[15] to exist, _bi_fx_valid(di@299, ding@304) must return True
# But we know cl_gap=1 < 4, so this FAILS the gap check in open.
# Pyarmor must use k_gap instead of cl_gap for the gap check.

# Key insight: pyarmor uses k_gap for gap check AND must also change the strict check
# Let me test: k_gap for gap + ck_three for strict
print("\n\n=== Combined test: k_gap for gap + ck_three for strict ===")

def variant_k_gap_ck_three(self, start_fx, end_fx):
    if start_fx.type == end_fx.type: return False
    cl_gap = end_fx.k.index - start_fx.k.index
    k_gap = end_fx.k.k_index - start_fx.k.k_index
    if self.bi_type == Config.BI_TYPE_DD.value:
        if cl_gap < 1: return False
    elif self.bi_type == Config.BI_TYPE_JDB.value:
        if k_gap < 4: return False
    else:
        if k_gap < 4: return False  # k_gap instead of cl_gap
    if k_gap < self.fx_check_k_nums:
        if self.allow_bi_fx_strict:
            qj = Config.FX_QJ_CK.value  # ck instead of k
            qy = Config.FX_QY_THREE.value
            if start_fx.type == "ding" and end_fx.type == "di":
                if start_fx.low(qj, qy) < end_fx.low(qj, qy): return False
                if end_fx.high(qj, qy) > start_fx.high(qj, qy): return False
            elif start_fx.type == "di" and end_fx.type == "ding":
                if start_fx.high(qj, qy) > end_fx.high(qj, qy): return False
                if end_fx.low(qj, qy) < start_fx.low(qj, qy): return False
    return True

original = CL_O._bi_fx_valid

for name, path in [
    ("BTCd", "tests/test_data/BTC_USDT_d_500.parquet"),
    ("ETH60", "tests/test_data/ETH_USDT_60m_1000.parquet"),
    ("BTC60", "tests/test_data/BTC_USDT_60m_1000.parquet"),
    ("BTC5m", "tests/test_data/BTC_USDT_5m_1000.parquet"),
    ("ETH5m", "tests/test_data/ETH_USDT_5m_1000.parquet"),
]:
    df = pd.read_parquet(path)
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    n_p = len(cd_p.get_bis())
    
    CL_O._bi_fx_valid = variant_k_gap_ck_three
    cd_o = CL_O("test", "test", config=CL_CONFIG)
    cd_o.process_klines(df)
    n_o = len(cd_o.get_bis())
    
    marker = "✅" if n_o == n_p else "❌"
    print(f"  {name}: pyarmor={n_p}, k_gap+ck_three={n_o} {marker}")

CL_O._bi_fx_valid = original
