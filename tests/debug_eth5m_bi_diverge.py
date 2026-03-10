"""
Investigate ETH5m BI count mismatch: 71 vs 73.
Find where the extra 2 BIs appear/disappear.
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

bis_o = cd_o.get_bis()
bis_p = cd_p.get_bis()
fxs_o = cd_o.get_fxs()
fxs_p = cd_p.get_fxs()

print(f"Open:    {len(fxs_o)} FXes, {len(bis_o)} BIs")
print(f"Pyarmor: {len(fxs_p)} FXes, {len(bis_p)} BIs")

# Compare BIs by aligning on start k_index
print(f"\n=== BI alignment by start k_index ===")
o_idx = 0
p_idx = 0
divergence_count = 0

while o_idx < len(bis_o) and p_idx < len(bis_p):
    bo = bis_o[o_idx]
    bp = bis_p[p_idx]
    
    o_start = bo.start.k.k_index
    p_start = bp.start.k.k_index
    o_end = bo.end.k.k_index
    p_end = bp.end.k.k_index
    
    if o_start == p_start and o_end == p_end:
        o_idx += 1
        p_idx += 1
    else:
        # Show divergence area
        if divergence_count == 0:
            # Show a few matching BIs before the divergence
            for k in range(max(0, o_idx - 2), o_idx):
                bk = bis_o[k]
                print(f"  bi_o[{k:>3}] {bk.type:>4} k={bk.start.k.k_index}→{bk.end.k.k_index} h={bk.high:.2f} l={bk.low:.2f}")
            print(f"  --- DIVERGENCE STARTS ---")
        
        divergence_count += 1
        o_str = f"bi_o[{o_idx:>3}] {bo.type:>4} k={o_start}→{o_end} h={bo.high:.2f} l={bo.low:.2f}"
        p_str = f"bi_p[{p_idx:>3}] {bp.type:>4} k={p_start}→{p_end} h={bp.high:.2f} l={bp.low:.2f}"
        
        if o_start == p_start:
            # Same start, different end
            print(f"  {o_str}  <<<  {p_str}  END_DIFF")
            if o_end < p_end:
                o_idx += 1  # open's BI is shorter, advance it
            else:
                p_idx += 1
        elif o_start < p_start:
            # Open has an extra BI
            print(f"  {o_str}  <<<  EXTRA OPEN")
            o_idx += 1
        else:
            # Pyarmor has an extra BI
            print(f"  {'':60s}  >>>  {p_str}  EXTRA PYARMOR")
            p_idx += 1
        
        if divergence_count > 20:
            print("  ... (truncated)")
            break

# Show remaining
while o_idx < len(bis_o):
    bo = bis_o[o_idx]
    print(f"  bi_o[{o_idx:>3}] {bo.type:>4} k={bo.start.k.k_index}→{bo.end.k.k_index} EXTRA OPEN")
    o_idx += 1
    if o_idx > len(bis_o) - 5:
        break

while p_idx < len(bis_p):
    bp = bis_p[p_idx]
    print(f"  {'':60s}  EXTRA PYARMOR bi_p[{p_idx:>3}] {bp.type:>4} k={bp.start.k.k_index}→{bp.end.k.k_index}")
    p_idx += 1
    if p_idx > len(bis_p) - 5:
        break

print(f"\nTotal divergences: {divergence_count}")

# Also check FX (fractal) differences around the divergence
print(f"\n=== FX comparison ===")
fx_diverge = 0
for i in range(max(len(fxs_o), len(fxs_p))):
    fo = fxs_o[i] if i < len(fxs_o) else None
    fp = fxs_p[i] if i < len(fxs_p) else None
    if fo and fp:
        if fo.k.k_index != fp.k.k_index or fo.type != fp.type:
            if fx_diverge == 0:
                # Show context before
                for k in range(max(0, i-2), i):
                    fk = fxs_o[k]
                    pk = fxs_p[k]
                    print(f"  fx[{k:>3}] o={fk.type:4s} ck={fk.k.k_index:>4} | p={pk.type:4s} ck={pk.k.k_index:>4}")
                print(f"  --- FX DIVERGENCE ---")
            fx_diverge += 1
            o_str = f"o={fo.type:4s} ck={fo.k.k_index:>4} val={fo.val:.2f}" if fo else "N/A"
            p_str = f"p={fp.type:4s} ck={fp.k.k_index:>4} val={fp.val:.2f}" if fp else "N/A"
            print(f"  fx[{i:>3}] {o_str} | {p_str}")
            if fx_diverge > 15:
                print("  ... (truncated)")
                break
    elif fo:
        fx_diverge += 1
        print(f"  fx[{i:>3}] EXTRA OPEN: {fo.type} ck={fo.k.k_index}")
    elif fp:
        fx_diverge += 1
        print(f"  fx[{i:>3}] EXTRA PYARMOR: {fp.type} ck={fp.k.k_index}")
