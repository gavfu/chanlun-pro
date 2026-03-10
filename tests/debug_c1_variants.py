"""Test C1-only strict, and also: strict with C1 using right-half for start_fx only.

For UP BI (di→ding): C1 = start.high > end.high
For DOWN BI (ding→di): C1 = start.low < end.low

Test variants:
1. C1-only with standard FX interval
2. C1-only with right-half (klines[1:]) for start_fx
3. No strict at all but with BOTH cl_gap AND k_gap checks
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

def build_bis_variant(cd, fxs, variant):
    """Build BIs with different strict check variants."""
    bis = []
    if len(fxs) < 2: return bis
    qj = cd.fx_qj; qy = cd.fx_qy
    
    def get_high_low(fx, use_rh_start=False):
        if use_rh_start and len(fx.klines) > 1:
            h = max([rk.h for ck in fx.klines[1:] for rk in ck.klines])
            l = min([rk.l for ck in fx.klines[1:] for rk in ck.klines])
        else:
            h = fx.high(qj, qy)
            l = fx.low(qj, qy)
        return h, l
    
    def check_valid(sfx, efx):
        if sfx.type == efx.type: return False
        cl_gap = efx.k.index - sfx.k.index
        k_gap = efx.k.k_index - sfx.k.k_index
        
        if variant == "c1_only":
            if cl_gap < 4: return False
            if k_gap < 13:
                h_s, l_s = get_high_low(sfx)
                h_e, l_e = get_high_low(efx)
                if sfx.type == "ding" and efx.type == "di":
                    if l_s < l_e: return False  # C1 only
                elif sfx.type == "di" and efx.type == "ding":
                    if h_s > h_e: return False  # C1 only
        elif variant == "c1_rh_start":
            if cl_gap < 4: return False
            if k_gap < 13:
                h_s, l_s = get_high_low(sfx, use_rh_start=True)
                h_e, l_e = get_high_low(efx)
                if sfx.type == "ding" and efx.type == "di":
                    if l_s < l_e: return False  # C1 only, right-half start
                elif sfx.type == "di" and efx.type == "ding":
                    if h_s > h_e: return False  # C1 only, right-half start
        elif variant == "no_strict_dual_gap":
            # Both cl_gap >= 4 AND k_gap >= 4 required, no strict
            if cl_gap < 4 or k_gap < 4: return False
        elif variant == "cl_and_k_gap_strict":
            # cl_gap >= 4 AND k_gap >= 4 required, with standard strict
            if cl_gap < 4 or k_gap < 4: return False
            if k_gap < 13:
                h_s, l_s = get_high_low(sfx)
                h_e, l_e = get_high_low(efx)
                if sfx.type == "ding" and efx.type == "di":
                    if l_s < l_e: return False
                    if h_e > h_s: return False
                elif sfx.type == "di" and efx.type == "ding":
                    if h_s > h_e: return False
                    if l_e < l_s: return False
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

variants = ["c1_only", "c1_rh_start", "no_strict_dual_gap", "cl_and_k_gap_strict"]

for name, path in datasets.items():
    df = pd.read_parquet(path)
    cd_o = CL_O("test", "test", config=CL_CONFIG)
    cd_o.process_klines(df)
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    
    fxs = cd_o.get_fxs()
    bis_pya = cd_p.get_bis()
    bis_open = cd_o.get_bis()
    
    print(f"\n{'='*60}")
    print(f"=== {name}: pyarmor={len(bis_pya)}, baseline={len(bis_open)} ===")
    
    base_match = sum(1 for j in range(min(len(bis_open), len(bis_pya)))
                     if bis_open[j].start.k.k_index == bis_pya[j].start.k.k_index
                     and bis_open[j].end.k.k_index == bis_pya[j].end.k.k_index)
    print(f"  baseline: {len(bis_open)} BIs, {base_match}/{min(len(bis_open),len(bis_pya))} matches")
    
    for v in variants:
        bis_new = build_bis_variant(cd_o, fxs, v)
        bound_match = sum(1 for j in range(min(len(bis_new), len(bis_pya)))
                          if bis_new[j].start.k.k_index == bis_pya[j].start.k.k_index
                          and bis_new[j].end.k.k_index == bis_pya[j].end.k.k_index)
        print(f"  {v:25s}: {len(bis_new):3d} BIs, {bound_match}/{min(len(bis_new),len(bis_pya))} matches")
