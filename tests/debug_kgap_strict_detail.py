"""Check strict check status for all k_gap=4 pyarmor BIs.
Also check the REJECTED (extra) BIs from k_gap variant — do they fail strict?"""
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

datasets = {
    "BTC60": "tests/test_data/BTC_USDT_60m_1000.parquet",
    "ETH60": "tests/test_data/ETH_USDT_60m_1000.parquet",
    "BTC5m": "tests/test_data/BTC_USDT_5m_1000.parquet",
    "ETH5m": "tests/test_data/ETH_USDT_5m_1000.parquet",
}

for name, path in datasets.items():
    df = pd.read_parquet(path)
    cd_o = CL_O("test", "test", config=CL_CONFIG)
    cd_o.process_klines(df)
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    
    fxs = cd_o.get_fxs()
    bis_pya = cd_p.get_bis()
    qj = cd_o.fx_qj; qy = cd_o.fx_qy
    
    print(f"\n{'='*60}")
    print(f"=== {name}: k_gap=4 pyarmor BIs strict analysis ===")
    
    for bi in bis_pya:
        cl = bi.end.k.index - bi.start.k.index
        k = bi.end.k.k_index - bi.start.k.k_index
        if k <= 5:  # check small k_gap BIs
            sfx = bi.start; efx = bi.end
            h_s = sfx.high(qj, qy); l_s = sfx.low(qj, qy)
            h_e = efx.high(qj, qy); l_e = efx.low(qj, qy)
            
            if sfx.type == "di" and efx.type == "ding":
                c1 = h_s > h_e  # start.high > end.high
                c2 = l_e < l_s  # end.low < start.low
            else:  # ding → di
                c1 = l_s < l_e  # start.low < end.low  
                c2 = h_e > h_s  # end.high > start.high
            
            strict_fail = c1 or c2
            
            print(f"  bi[{bi.index}]: {bi.type:4s} {sfx.k.k_index}→{efx.k.k_index} "
                  f"cl={cl} k={k} strict={'FAIL' if strict_fail else 'PASS'} "
                  f"(C1={c1}, C2={c2})")
            
            # Also show the SHARE info: do they share klines?
            s_ck_indices = set(ck.index for ck in sfx.klines)
            e_ck_indices = set(ck.index for ck in efx.klines)
            shared = s_ck_indices & e_ck_indices
            if shared:
                print(f"    Shared klines: {shared}")
            
            # Show with klines[1:] (right-half)
            if len(sfx.klines) > 1:
                rh_h = max([rk.h for ck in sfx.klines[1:] for rk in ck.klines])
                rh_l = min([rk.l for ck in sfx.klines[1:] for rk in ck.klines])
                if sfx.type == "di" and efx.type == "ding":
                    c1r = rh_h > h_e
                    c2r = l_e < rh_l
                else:
                    c1r = rh_l < l_e
                    c2r = h_e > rh_h
                strict_fail_rh = c1r or c2r
                print(f"    right-half strict: {'FAIL' if strict_fail_rh else 'PASS'} "
                      f"(C1={c1r}, C2={c2r})")

print(f"\n\n{'='*60}")
print("=== BTC60: What BIs does k_gap=4 create that pyarmor doesn't? ===")

# Check BTC60 specifically — k_gap variant creates bi[2] down 69→95 but pyarmor has down 69→101
# What's the strict check for the extra intermediate FX?

df = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cd_o = CL_O("test", "test", config=CL_CONFIG)
cd_o.process_klines(df)
fxs = cd_o.get_fxs()
qj = cd_o.fx_qj; qy = cd_o.fx_qy

# Find the FXes between 69 and 101
print(f"\nFXes between k=60 and k=105:")
for fx in fxs:
    if 60 <= fx.k.k_index <= 105:
        print(f"  {fx.type:4s} k={fx.k.k_index} (ck_idx={fx.k.index}) val={fx.val:.2f}")
        # Check FX interval
        h = fx.high(qj, qy)
        l = fx.low(qj, qy)
        print(f"    h={h:.2f}, l={l:.2f}")

# Check what makes k_gap variant accept bi from ding@69 to di@95
# Find ding@69 and di@95
for fx in fxs:
    if fx.k.k_index == 69 and fx.type == "ding":
        ding69 = fx
    if fx.k.k_index == 95 and fx.type == "di":
        di95 = fx
    if fx.k.k_index == 99 and fx.type == "ding":
        ding99 = fx

print(f"\n=== ding@69 → di@95 ===")
cl = di95.k.index - ding69.k.index
k = di95.k.k_index - ding69.k.k_index
print(f"  cl_gap={cl}, k_gap={k}")
# Strict: ding→di (down bi)
# C1: start.low < end.low
# C2: end.high > start.high
h_s = ding69.high(qj, qy); l_s = ding69.low(qj, qy)
h_e = di95.high(qj, qy); l_e = di95.low(qj, qy)
c1 = l_s < l_e
c2 = h_e > h_s
print(f"  start.h={h_s:.2f}, l={l_s:.2f}")
print(f"  end.h={h_e:.2f}, l={l_e:.2f}")
print(f"  C1(start.l<end.l): {l_s:.2f}<{l_e:.2f} = {c1}")
print(f"  C2(end.h>start.h): {h_e:.2f}>{h_s:.2f} = {c2}")
print(f"  strict: {'FAIL' if c1 or c2 else 'PASS'}")

# Check di@95 → ding@99 (confirmation)
print(f"\n=== di@95 → ding@99 (confirmation) ===")
cl2 = ding99.k.index - di95.k.index
k2 = ding99.k.k_index - di95.k.k_index
print(f"  cl_gap={cl2}, k_gap={k2}")
h2_s = di95.high(qj, qy); l2_s = di95.low(qj, qy)
h2_e = ding99.high(qj, qy); l2_e = ding99.low(qj, qy)
c1_2 = h2_s > h2_e
c2_2 = l2_e < l2_s
print(f"  start.h={h2_s:.2f}, l={l2_s:.2f}")
print(f"  end.h={h2_e:.2f}, l={l2_e:.2f}")
print(f"  C1(start.h>end.h): {h2_s:.2f}>{h2_e:.2f} = {c1_2}")
print(f"  C2(end.l<start.l): {l2_e:.2f}<{l2_s:.2f} = {c2_2}")
print(f"  strict: {'FAIL' if c1_2 or c2_2 else 'PASS'}")
