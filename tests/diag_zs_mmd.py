"""Compare ZS (pivots) between cl_open and cl_pyarmor in detail."""
import sys; sys.path.insert(0, 'src')
import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')

cd_o = CL_O('BTC/USDT', '60m')
cd_o.process_klines(df)

cd_p = CL_P('BTC/USDT', '60m')
cd_p.process_klines(df)

print("=== BI ZS (笔中枢) ===")
print(f"Open: {len(cd_o.bi_zss)}, Pyarmor: {len(cd_p.bi_zss)}")
for i, zs in enumerate(cd_o.bi_zss):
    lines = f"lines=[{zs.lines[0].index}→{zs.lines[-1].index}]" if zs.lines else ""
    print(f"  O[{i}] {zs.zs_type} ZG={zs.zg:.1f} ZD={zs.zd:.1f} {lines}")
for i, zs in enumerate(cd_p.bi_zss):
    lines = f"lines=[{zs.lines[0].index}→{zs.lines[-1].index}]" if zs.lines else ""
    print(f"  P[{i}] {zs.zs_type} ZG={zs.zg:.1f} ZD={zs.zd:.1f} {lines}")

print("\n=== XD ZS (线段中枢) ===")
print(f"Open: {len(cd_o.xd_zss)}, Pyarmor: {len(cd_p.xd_zss)}")
for i, zs in enumerate(cd_o.xd_zss):
    lines = f"lines=[{zs.lines[0].index}→{zs.lines[-1].index}]" if zs.lines else ""
    print(f"  O[{i}] {zs.zs_type} ZG={zs.zg:.1f} ZD={zs.zd:.1f} {lines}")
for i, zs in enumerate(cd_p.xd_zss):
    lines = f"lines=[{zs.lines[0].index}→{zs.lines[-1].index}]" if zs.lines else ""
    print(f"  P[{i}] {zs.zs_type} ZG={zs.zg:.1f} ZD={zs.zd:.1f} {lines}")

# With only 3 XDs, there shouldn't be an XD ZS because you need at least 3 overlapping XDs
# to form a ZS. Let me check the ZS construction logic.
print("\n=== XD details ===")
for x in cd_o.xds:
    print(f"  [{x.index}] {x.type} bi[{x.start_line.index}]→bi[{x.end_line.index}] h={x.high:.1f} l={x.low:.1f}")

# ZS check: 3 consecutive XDs with an overlapping price range
# XD[0] DOWN: high area, XD[1] UP: to high, XD[2] DOWN: from high
# For XD ZS: need XD[1] and XD[2] to have overlapping range with XD[0]
# Actually, ZS is formed by the overlapping part of the SECOND and THIRD lines
# ZG = min(high of lines[1], high of lines[2])
# ZD = max(low of lines[1], low of lines[2])
# If ZG > ZD, we have a ZS

print("\n=== BI MMD/BC details ===")
for i in range(len(cd_o.bis)):
    o_bi = cd_o.bis[i]
    p_bi = cd_p.bis[i]
    
    o_mmd = sorted([m.name for m in o_bi.mmds]) if o_bi.mmds else []
    p_mmd = sorted([m.name for m in p_bi.mmds]) if p_bi.mmds else []
    
    o_bc = sorted(list(set(bc.type for bc in o_bi.bcs))) if o_bi.bcs else []
    p_bc = sorted(list(set(bc.type for bc in p_bi.bcs))) if p_bi.bcs else []
    
    if o_mmd != p_mmd or o_bc != p_bc:
        print(f"  bi[{i}] {o_bi.type}")
        if o_mmd != p_mmd:
            print(f"    MMD: open={o_mmd} pyarmor={p_mmd}")
        if o_bc != p_bc:
            print(f"    BC:  open={o_bc} pyarmor={p_bc}")
