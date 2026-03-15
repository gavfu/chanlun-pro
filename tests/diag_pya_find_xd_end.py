"""
直接调用 cl_pyarmor 的 _find_xd_end 以对比行为。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
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

df = pd.read_parquet('tests/test_data/BTC_USDT_4h_5000.parquet')
cl_p = CL_P("test", "test", CL_CONFIG)
cl_p.process_klines(df)
xds_p = cl_p.get_xds()

# Re-index
for i, xd in enumerate(xds_p):
    xd.index = i

print(f"cl_pyarmor XDs: {len(xds_p)}")

# Call cl_pyarmor's methods directly
print("\n--- cl_pyarmor._find_first_xd_start(xds) ---")
result = cl_p._find_first_xd_start(xds_p)
print(f"  → {result}")

if result:
    xd_type, start_idx = result
    print(f"\n--- cl_pyarmor._find_xd_end(xds, {start_idx}, '{xd_type}') ---")
    r = cl_p._find_xd_end(xds_p, start_idx, xd_type)
    if r:
        end_idx, ding_fx, di_fx, tzxls = r
        print(f"  → end_idx={end_idx}")
        print(f"     UP ZSD: {xd_type} xd[{start_idx}→{end_idx}]")
        # Now DOWN ZSD from end_idx+1
        next_start = end_idx + 1
        next_type = "down" if xd_type == "up" else "up"
        print(f"\n--- cl_pyarmor._find_xd_end(xds, {next_start}, '{next_type}') ---")
        r2 = cl_p._find_xd_end(xds_p, next_start, next_type)
        if r2:
            end_idx2, _, _, _ = r2
            print(f"  → end_idx={end_idx2}")
            print(f"     DOWN ZSD: {next_type} xd[{next_start}→{end_idx2}]")
        else:
            print(f"  → None (no end found, trailing partial)")
            # Trailing partial: last xd of next_type
            trail_end = len(xds_p) - 1
            if xds_p[trail_end].type != next_type:
                trail_end -= 1
            print(f"     Trailing partial: {next_type} xd[{next_start}→{trail_end}]")
    else:
        print(f"  → None")

# Also check cl_open's behavior for comparison
print("\n\n--- OPEN comparison ---")
from chanlun.cl_open import CL as CL_O
cl_o = CL_O("test", "test", CL_CONFIG)
cl_o.process_klines(df)
# Inject pyarmor XDs
for i, xd in enumerate(xds_p):
    xd.index = i
cl_o.xds = list(xds_p)

print("cl_open._find_first_xd_start(pyarmor_xds):")
r_o = cl_o._find_first_xd_start(xds_p)
print(f"  → {r_o}")

if r_o:
    xd_type_o, start_o = r_o
    print(f"cl_open._find_xd_end(pyarmor_xds, {start_o}, '{xd_type_o}'):")
    r_o2 = cl_o._find_xd_end(xds_p, start_o, xd_type_o)
    if r_o2:
        end_o, _, _, _ = r_o2
        print(f"  → end={end_o}  (UP ZSD: xd[{start_o}→{end_o}])")
        # DOWN ZSD
        ns = end_o + 1
        nt = "down" if xd_type_o == "up" else "up"
        print(f"cl_open._find_xd_end(pyarmor_xds, {ns}, '{nt}'):")
        r_o3 = cl_o._find_xd_end(xds_p, ns, nt)
        if r_o3:
            end_o3, _, _, _ = r_o3
            print(f"  → end={end_o3}  (DOWN ZSD: xd[{ns}→{end_o3}])")
        else:
            print(f"  → None (trailing partial)")
    else:
        print(f"  → None")
