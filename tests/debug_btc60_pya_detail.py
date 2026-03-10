"""Check pyarmor BIs around BTC60 divergence and ETH5m divergence."""
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

# BTC60
print("=== BTC60: Pyarmor BIs [14:21] ===")
df = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cd_p = CL_P("test", "test", config=CL_CONFIG)
cd_p.process_klines(df)
for bi in cd_p.get_bis()[14:21]:
    cl_gap = bi.end.k.index - bi.start.k.index
    k_gap = bi.end.k.k_index - bi.start.k.k_index
    print(f"  bi[{bi.index}] {bi.type} k={bi.start.k.k_index}→{bi.end.k.k_index} cl={cl_gap} k={k_gap}")

# ETH5m
print("\n=== ETH5m: Pyarmor BIs [25:33] ===")
df = pd.read_parquet("tests/test_data/ETH_USDT_5m_1000.parquet")
cd_p2 = CL_P("test", "test", config=CL_CONFIG)
cd_p2.process_klines(df)
for bi in cd_p2.get_bis()[25:33]:
    cl_gap = bi.end.k.index - bi.start.k.index
    k_gap = bi.end.k.k_index - bi.start.k.k_index
    print(f"  bi[{bi.index}] {bi.type} k={bi.start.k.k_index}→{bi.end.k.k_index} cl={cl_gap} k={k_gap}")

# For BTC60 ding@304: what is the confirmation in pyarmor?
# bi[16] = down 304→X. The confirmation must come from a ding after X
# In the deep compare, all confirmations from ding@304 fail strict!
# ding@304 confirmations: di@306 strict=False, di@309 strict=False, di@315 strict=False
# But pyarmor has bi[16] = down 304→X, so it DID form a BI!

# Check: pyarmor's FXes might differ from open's FXes?
cd_o = CL_O("test", "test", config=CL_CONFIG)
df = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cd_o.process_klines(df)

fxs_o = cd_o.get_fxs()
fxs_p = cd_p.get_fxs()

print(f"\n=== BTC60 FX count: Open={len(fxs_o)} Pyarmor={len(fxs_p)} ===")

# Check if FXes around 299-315 are identical
print("\nOpen FXes around 299-320:")
for fx in fxs_o:
    if 295 <= fx.k.k_index <= 320:
        print(f"  {fx.type:>4}@{fx.k.k_index} val={fx.val:.2f} ck_idx={fx.k.index}")

print("\nPyarmor FXes around 299-320:")
for fx in fxs_p:
    if 295 <= fx.k.k_index <= 320:
        print(f"  {fx.type:>4}@{fx.k.k_index} val={fx.val:.2f} ck_idx={fx.k.index}")
