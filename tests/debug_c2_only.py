"""HYPOTHESIS: Strict check uses ONLY C2, not C1.

For UP BI (di→ding): only check end.low < start.low (C2)
For DOWN BI (ding→di): only check end.high > start.high (C2)

C1 (the "same-direction overlap") is SKIPPED.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P
from chanlun.cl_interface import FX, BI

CL_CONFIG = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
}

def build_bis_c2_only(cd, fxs):
    """Build BIs with strict check using ONLY C2."""
    bis = []
    if len(fxs) < 2: return bis
    qj = cd.fx_qj; qy = cd.fx_qy
    
    def check_valid(sfx, efx):
        if sfx.type == efx.type: return False
        cl_gap = efx.k.index - sfx.k.index
        k_gap = efx.k.k_index - sfx.k.k_index
        if cl_gap < 4: return False
        if k_gap < 13:
            if sfx.type == "ding" and efx.type == "di":
                # DOWN BI: only check C2 = end.high > start.high
                if efx.high(qj, qy) > sfx.high(qj, qy): return False
            elif sfx.type == "di" and efx.type == "ding":
                # UP BI: only check C2 = end.low < start.low
                if efx.low(qj, qy) < sfx.low(qj, qy): return False
        return True
    
    start_fx = fxs[0]; start_idx = 0
    end_fx = None; end_idx = -1; has_confirmed = False
    i = 1
    while i < len(fxs):
        cur_fx = fxs[i]
        if end_fx is None:
            if cur_fx.type == start_fx.type:
                if not has_confirmed:
                    if start_fx.type == "ding" and cur_fx.val > start_fx.val:
                        start_fx = cur_fx; start_idx = i
                    elif start_fx.type == "di" and cur_fx.val < start_fx.val:
                        start_fx = cur_fx; start_idx = i
            else:
                if check_valid(start_fx, cur_fx):
                    end_fx = cur_fx; end_idx = i
            i += 1
        else:
            if cur_fx.type == end_fx.type:
                if check_valid(start_fx, cur_fx):
                    if end_fx.type == "di" and cur_fx.val <= end_fx.val:
                        end_fx = cur_fx; end_idx = i
                    elif end_fx.type == "ding" and cur_fx.val >= end_fx.val:
                        end_fx = cur_fx; end_idx = i
                i += 1
            else:
                if check_valid(end_fx, cur_fx):
                    bi_type = "down" if start_fx.type == "ding" else "up"
                    bis.append(BI(start=start_fx, end=end_fx, _type=bi_type,
                                  index=len(bis), default_zs_type="zs_type_bz"))
                    has_confirmed = True
                    start_fx = end_fx; start_idx = end_idx
                    end_fx = None; end_idx = -1
                    i = start_idx + 1
                else:
                    i += 1
    if end_fx is not None:
        bi_type = "down" if start_fx.type == "ding" else "up"
        bis.append(BI(start=start_fx, end=end_fx, _type=bi_type,
                      index=len(bis), default_zs_type="zs_type_bz"))
    return bis


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
    bis_new = build_bis_c2_only(cd_o, fxs)
    bis_pya = cd_p.get_bis()
    bis_open = cd_o.get_bis()
    
    print(f"\n{'='*60}")
    print(f"=== {name} ===")
    print(f"  pyarmor: {len(bis_pya)} BIs")
    print(f"  baseline: {len(bis_open)} BIs")
    print(f"  c2-only: {len(bis_new)} BIs")
    
    bound_match = 0
    bound_total = min(len(bis_new), len(bis_pya))
    for j in range(bound_total):
        bk = bis_new[j]; bp = bis_pya[j]
        if bk.start.k.k_index == bp.start.k.k_index and bk.end.k.k_index == bp.end.k.k_index:
            bound_match += 1
    print(f"  c2-only boundary matches: {bound_match}/{bound_total}")
    
    base_match = 0
    base_total = min(len(bis_open), len(bis_pya))
    for j in range(base_total):
        bo = bis_open[j]; bp = bis_pya[j]
        if bo.start.k.k_index == bp.start.k.k_index and bo.end.k.k_index == bp.end.k.k_index:
            base_match += 1
    print(f"  baseline boundary matches: {base_match}/{base_total}")
    
    # Show first divergence
    for j in range(min(len(bis_new), len(bis_pya))):
        bk = bis_new[j]; bp = bis_pya[j]
        if bk.start.k.k_index != bp.start.k.k_index or bk.end.k.k_index != bp.end.k.k_index:
            print(f"  First divergence at bi[{j}]:")
            for k in range(j, min(j+5, min(len(bis_new), len(bis_pya)))):
                bn = bis_new[k]; bp2 = bis_pya[k]
                print(f"    [{k}] c2only={bn.type} {bn.start.k.k_index}→{bn.end.k.k_index}  "
                      f"pya={bp2.type} {bp2.start.k.k_index}→{bp2.end.k.k_index}")
            break
