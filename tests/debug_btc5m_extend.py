"""Check what happens with extension and no-check-extension variants in BTC5m."""
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

df = pd.read_parquet("tests/test_data/BTC_USDT_5m_1000.parquet")
cd = CL_O("test", "test", config=CL_CONFIG)
cd.process_klines(df)
cd_p = CL_P("test", "test", config=CL_CONFIG)
cd_p.process_klines(df)

fxs = cd.get_fxs()
qj = cd.fx_qj
qy = cd.fx_qy

fx_map = {}
for fx in fxs:
    fx_map[fx.k.k_index] = fx

# Pyarmor has bi[61] = up 875→889, bi[62] = down 889→899
# Let's check FXes around 889-905
print("=== FXes from k=887 to k=910 ===")
for fx in fxs:
    if 887 <= fx.k.k_index <= 910:
        print(f"  {fx.type:>4}@{fx.k.k_index} val={fx.val:.2f} "
              f"high={fx.high(qj,qy):.2f} low={fx.low(qj,qy):.2f} ck_idx={fx.k.index}")

# Now simulate: what if extension doesn't check _bi_fx_valid?
# start=di@875, end_fx=ding@879
# ding@884: val < ding@879, no extension
# ding@889: val > ding@879, extension WITHOUT primary check → end_fx=ding@889
# Then confirmation scan continues from ding@889:
print("\n=== Simulation: NO primary check for extension ===")
print("start_fx = di@875, end_fx = ding@879")
print("ding@884 val=67321.10 < 67329.70, no extension")
print("ding@889 val=67399.80 > 67329.70, EXTEND → end_fx = ding@889")
print("\nNow confirming from ding@889:")

ding889 = fx_map[889]
for fx in fxs:
    if fx.k.k_index > 889 and fx.k.k_index <= 910 and fx.type == "di":
        cl_gap = fx.k.index - ding889.k.index
        k_gap = fx.k.k_index - ding889.k.k_index
        
        # Standard _bi_fx_valid check
        valid_cl = cl_gap >= 4
        valid_k = k_gap >= 4
        
        # Strict check (down BI: ding→di)
        c1 = ding889.low(qj, qy) < fx.low(qj, qy)
        c2 = fx.high(qj, qy) > ding889.high(qj, qy)
        strict_pass = not c1 and not c2
        in_strict_range = k_gap < 13
        
        overall_valid_cl = valid_cl and (strict_pass if in_strict_range else True)
        overall_valid_k = valid_k and (strict_pass if in_strict_range else True)
        
        print(f"  di@{fx.k.k_index}: cl_gap={cl_gap} k_gap={k_gap} "
              f"valid_cl={valid_cl} valid_k={valid_k} "
              f"strict_pass={strict_pass} in_range={in_strict_range}")
        print(f"    overall(cl_gap): {overall_valid_cl} overall(k_gap): {overall_valid_k}")
        print(f"    ding889.low={ding889.low(qj,qy):.2f} di.low={fx.low(qj,qy):.2f}")
        print(f"    ding889.high={ding889.high(qj,qy):.2f} di.high={fx.high(qj,qy):.2f}")

# Also, pyarmor bi[62] = down 889→899. Check what di is at k=899.
print("\n=== Pyarmor BIs around this region ===")
for bi in cd_p.get_bis()[60:66]:
    print(f"  bi[{bi.index}] {bi.type} k={bi.start.k.k_index}→{bi.end.k.k_index}")

# What if confirmation from ding@879 is rejected by a DIFFERENT strict check?
# What about checking whether confirmation FX (di) creates a "valid" BI from end_fx,
# but also checking if there's a higher ding between end_fx and confirmation?
print("\n=== Check: any higher ding between ding@879 and di@887? ===")
for fx in fxs:
    if 879 < fx.k.k_index < 887 and fx.type == "ding":
        print(f"  ding@{fx.k.k_index} val={fx.val:.2f} vs ding@879 val={fx_map[879].val:.2f}")
        print(f"    Higher? {fx.val > fx_map[879].val}")

# What if confirmation requires NO higher same-type FX between end_fx and confirm_fx?
# (i.e., di@887 must be the LOWEST di between ding@879 and di@887)
print("\n=== Check: lowest di between ding@879 and di@887? ===")
dis_between = []
for fx in fxs:
    if 879 < fx.k.k_index <= 887 and fx.type == "di":
        dis_between.append(fx)
        print(f"  di@{fx.k.k_index} val={fx.val:.2f}")
if dis_between:
    lowest = min(dis_between, key=lambda f: f.val)
    print(f"  Lowest is di@{lowest.k.k_index} val={lowest.val:.2f}")
    
# What if confirmation requires that the confirming FX not just passes _bi_fx_valid,
# but also that it's NOT the last end_fx replacement?
# i.e., if there's a same-type FX after end_fx that's more extreme (even if it didn't pass
# extension check), the confirmation should be skipped?

print("\n=== All ding FXes between di@875 and di@891 ===")
for fx in fxs:
    if 875 < fx.k.k_index < 891 and fx.type == "ding":
        valid_primary = cd._bi_fx_valid(fx_map[875], fx)
        print(f"  ding@{fx.k.k_index} val={fx.val:.2f} "
              f"valid(di@875→ding): {valid_primary}")
