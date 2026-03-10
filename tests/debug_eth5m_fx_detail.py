"""Check raw K-lines in di@337 and ding@344 to understand strict check failure."""
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

df = pd.read_parquet("tests/test_data/ETH_USDT_5m_1000.parquet")
cd_o = CL_O("test", "test", config=CL_CONFIG)
cd_o.process_klines(df)

fxs = cd_o.get_fxs()
cks = cd_o.get_klines()

# Find di@337 and ding@344
di337 = None
ding344 = None
for fx in fxs:
    if fx.type == "di" and fx.k.k_index == 337:
        di337 = fx
    if fx.type == "ding" and fx.k.k_index == 344:
        ding344 = fx

qj = cd_o.fx_qj
qy = cd_o.fx_qy

print("=== di@337 (bottom FX) ===")
print(f"  val={di337.val:.2f}")
print(f"  high({qj},{qy})={di337.high(qj,qy):.2f}")
print(f"  low({qj},{qy})={di337.low(qj,qy):.2f}")
print(f"  FX has {len(di337.klines)} merged klines:")
for kl in di337.klines:
    n_sub = len(kl.klines) if hasattr(kl, 'klines') and kl.klines else 0
    k_idx = kl.k_index if hasattr(kl, 'k_index') else '?'
    print(f"    ck k_index={k_idx} h={kl.h:.2f} l={kl.l:.2f} (n_sub={n_sub})")
    if hasattr(kl, 'klines') and kl.klines:
        for sub in kl.klines:
            sub_k_idx = sub.k_index if hasattr(sub, 'k_index') else '?'
            print(f"      raw k_index={sub_k_idx} h={sub.h:.2f} l={sub.l:.2f}")

print(f"\n=== ding@344 (top FX) ===")
print(f"  val={ding344.val:.2f}")
print(f"  high({qj},{qy})={ding344.high(qj,qy):.2f}")
print(f"  low({qj},{qy})={ding344.low(qj,qy):.2f}")
print(f"  FX has {len(ding344.klines)} merged klines:")
for kl in ding344.klines:
    n_sub = len(kl.klines) if hasattr(kl, 'klines') and kl.klines else 0
    k_idx = kl.k_index if hasattr(kl, 'k_index') else '?'
    print(f"    ck k_index={k_idx} h={kl.h:.2f} l={kl.l:.2f} (n_sub={n_sub})")
    if hasattr(kl, 'klines') and kl.klines:
        for sub in kl.klines:
            sub_k_idx = sub.k_index if hasattr(sub, 'k_index') else '?'
            print(f"      raw k_index={sub_k_idx} h={sub.h:.2f} l={sub.l:.2f}")

# Now check pyarmor FX.high/low for comparison
cd_p = CL_P("test", "test", config=CL_CONFIG)
cd_p.process_klines(df)
fxs_p = cd_p.get_fxs()

di337_p = None
ding344_p = None
for fx in fxs_p:
    if fx.type == "di" and fx.k.k_index == 337:
        di337_p = fx
    if fx.type == "ding" and fx.k.k_index == 344:
        ding344_p = fx

print(f"\n=== Pyarmor comparison ===")
print(f"di@337 pyarmor: high={di337_p.high(qj,qy):.2f} low={di337_p.low(qj,qy):.2f}")
print(f"ding@344 pyarmor: high={ding344_p.high(qj,qy):.2f} low={ding344_p.low(qj,qy):.2f}")

# Check if pyarmor's _bi_fx_valid accepts di@337 → ding@344
v_p = cd_p._bi_fx_valid(di337_p, ding344_p)
print(f"\npyarmor._bi_fx_valid(di@337, ding@344) = {v_p}")

# Also check pyarmor's strict FX computation
print(f"\nUp BI strict check (di→ding):")
print(f"  di337_p.high > ding344_p.high? {di337_p.high(qj,qy)} > {ding344_p.high(qj,qy)} = {di337_p.high(qj,qy) > ding344_p.high(qj,qy)}")
print(f"  ding344_p.low < di337_p.low? {ding344_p.low(qj,qy)} < {di337_p.low(qj,qy)} = {ding344_p.low(qj,qy) < di337_p.low(qj,qy)}")
