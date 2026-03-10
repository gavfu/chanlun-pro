"""
Deep investigation of ETH5m K-line merging divergence around ck=55-56.
"""
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
cd_p = CL_P("test", "test", config=CL_CONFIG)
cd_p.process_klines(df)

# Get merged K-lines
cklines_o = cd_o.get_cl_klines()
cklines_p = cd_p.get_cl_klines()

print(f"Open:    {len(cklines_o)} merged K-lines")
print(f"Pyarmor: {len(cklines_p)} merged K-lines")

# Find divergence in merged K-lines
print(f"\n=== Merged K-line comparison ===")
for i in range(min(len(cklines_o), len(cklines_p))):
    co = cklines_o[i]
    cp = cklines_p[i]
    if co.h != cp.h or co.l != cp.l or co.index != cp.index:
        # Show context
        for k in range(max(0, i-3), i):
            ck_o = cklines_o[k]
            ck_p = cklines_p[k]
            print(f"  ck[{k:>4}] o: k_idx={ck_o.k_index:>4} h={ck_o.h:<10.2f} l={ck_o.l:<10.2f} n_klines={len(ck_o.klines)}  "
                  f"| p: k_idx={ck_p.k_index:>4} h={ck_p.h:<10.2f} l={ck_p.l:<10.2f} n_klines={len(ck_p.klines)}")
        print(f"  --- CK DIVERGENCE AT INDEX {i} ---")
        for k in range(i, min(i+8, min(len(cklines_o), len(cklines_p)))):
            ck_o = cklines_o[k]
            ck_p = cklines_p[k]
            match_h = "✓" if ck_o.h == ck_p.h else "✗"
            match_l = "✓" if ck_o.l == ck_p.l else "✗"
            print(f"  ck[{k:>4}] o: k_idx={ck_o.k_index:>4} h={ck_o.h:<10.2f} l={ck_o.l:<10.2f} n={len(ck_o.klines)} q={ck_o.q}  "
                  f"| p: k_idx={ck_p.k_index:>4} h={ck_p.h:<10.2f} l={ck_p.l:<10.2f} n={len(ck_p.klines)} q={ck_p.q}  "
                  f"h{match_h} l{match_l}")
        break

# Show raw K-lines around the divergence point
print(f"\n=== Raw K-lines around divergence ===")
raw_klines = df.values
col_names = list(df.columns)
print(f"  Columns: {col_names}")
# Assume typical OHLCV columns: date, open, high, low, close, volume
# Show raw K-lines around index 50-60
for i in range(50, min(65, len(raw_klines))):
    row = raw_klines[i]
    print(f"  raw[{i:>4}] h={float(row[2]):<10.2f} l={float(row[3]):<10.2f}")

# Also compare FX around divergence
fxs_o = cd_o.get_fxs()
fxs_p = cd_p.get_fxs()

print(f"\n=== FX comparison around divergence ===")
for i in range(15, min(25, max(len(fxs_o), len(fxs_p)))):
    fo = fxs_o[i] if i < len(fxs_o) else None
    fp = fxs_p[i] if i < len(fxs_p) else None
    if fo and fp:
        match = "✓" if fo.k.k_index == fp.k.k_index else "✗"
        o_str = f"{fo.type:4s} ck={fo.k.k_index:>4} val={fo.val:<10.2f} k_idx={fo.k.index}"
        p_str = f"{fp.type:4s} ck={fp.k.k_index:>4} val={fp.val:<10.2f} k_idx={fp.k.index}"
        print(f"  fx[{i:>3}] {match} o={o_str} | p={p_str}")
