"""
Detailed check of ETH5m pyarmor bis — full list to verify reconstruction.
Also show cl_open _build_bis (pre-split) for the same range.
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

df = pd.read_parquet('tests/test_data/ETH_USDT_5m_1000.parquet')

co = CL_O("test", "test", config=CL_CONFIG)
co.process_klines(df)
cp = CL_P("test", "test", config=CL_CONFIG)
cp.process_klines(df)

bis_p = cp.get_bis()
pre_split_o = co._build_bis(co.get_fxs())

print("=== ALL pyarmor bis (ETH5m) === (filtering k=250..400)")
for i, bi in enumerate(bis_p):
    if 250 <= bi.start.k.k_index <= 400 or 250 <= bi.end.k.k_index <= 400:
        split = "SPLIT" if bi.is_split else ""
        print(f"  bi[{i:>3}] {bi.type:>4} k={bi.start.k.k_index:>4}→{bi.end.k.k_index:>4} {split} {bi.is_split[:50] if bi.is_split else ''}")

print(f"\n=== cl_open pre-split bis (ETH5m) === (filtering k=250..400)")
for i, bi in enumerate(pre_split_o):
    if 250 <= bi.start.k.k_index <= 400 or 250 <= bi.end.k.k_index <= 400:
        print(f"  bi[{i:>3}] {bi.type:>4} k={bi.start.k.k_index:>4}→{bi.end.k.k_index:>4}")

# Manual reconstruction of pyarmor pre-split
print(f"\n=== Manual pyarmor pre-split reconstruction ===")
i = 0
recon = []
while i < len(bis_p):
    bi = bis_p[i]
    if bi.is_split and i + 2 < len(bis_p):
        merged_end = bis_p[i + 2].end.k.k_index
        recon.append((bi.type, bi.start.k.k_index, merged_end, 'MERGED'))
        i += 3
    else:
        recon.append((bi.type, bi.start.k.k_index, bi.end.k.k_index, ''))
        i += 1

for j, r in enumerate(recon):
    if 250 <= r[1] <= 400 or 250 <= r[2] <= 400:
        print(f"  bi[{j:>3}] {r[0]:>4} k={r[1]:>4}→{r[2]:>4} {r[3]}")
