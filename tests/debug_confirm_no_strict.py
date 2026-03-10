"""HYPOTHESIS: Strict check applied ONLY to primary candidate, NOT confirmation.
Gap check applied to BOTH.

This means:
- Primary: start_fx → end_fx: gap check + strict check
- Extension: start_fx → new_end_fx: gap check + strict check  
- Confirmation: end_fx → confirm_fx: gap check ONLY (no strict)
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

def build_bis_confirm_no_strict(cd, fxs):
    """Build BIs where confirmation uses gap-only check (no strict)."""
    bis = []
    if len(fxs) < 2: return bis
    qj = cd.fx_qj; qy = cd.fx_qy
    
    def check_full(sfx, efx):
        """Full check: gap + strict (for primary and extension)."""
        if sfx.type == efx.type: return False
        cl_gap = efx.k.index - sfx.k.index
        k_gap = efx.k.k_index - sfx.k.k_index
        if cl_gap < 4: return False
        if k_gap < 13:
            if sfx.type == "ding" and efx.type == "di":
                if sfx.low(qj, qy) < efx.low(qj, qy): return False
                if efx.high(qj, qy) > sfx.high(qj, qy): return False
            elif sfx.type == "di" and efx.type == "ding":
                if sfx.high(qj, qy) > efx.high(qj, qy): return False
                if efx.low(qj, qy) < sfx.low(qj, qy): return False
        return True
    
    def check_gap_only(sfx, efx):
        """Gap-only check (for confirmation)."""
        if sfx.type == efx.type: return False
        cl_gap = efx.k.index - sfx.k.index
        return cl_gap >= 4
    
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
                if check_full(start_fx, cur_fx):  # primary: full check
                    end_fx = cur_fx; end_idx = i
            i += 1
        else:
            if cur_fx.type == end_fx.type:
                if check_full(start_fx, cur_fx):  # extension: full check
                    if end_fx.type == "di" and cur_fx.val <= end_fx.val:
                        end_fx = cur_fx; end_idx = i
                    elif end_fx.type == "ding" and cur_fx.val >= end_fx.val:
                        end_fx = cur_fx; end_idx = i
                i += 1
            else:
                if check_gap_only(end_fx, cur_fx):  # confirmation: gap only!
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
    bis_new = build_bis_confirm_no_strict(cd_o, fxs)
    bis_pya = cd_p.get_bis()
    bis_open = cd_o.get_bis()
    
    print(f"\n{'='*60}")
    print(f"=== {name} ===")
    print(f"  pyarmor: {len(bis_pya)} BIs")
    print(f"  baseline: {len(bis_open)} BIs")
    print(f"  confirm-no-strict: {len(bis_new)} BIs")
    
    bound_match = 0
    bound_total = min(len(bis_new), len(bis_pya))
    for j in range(bound_total):
        bk = bis_new[j]; bp = bis_pya[j]
        if bk.start.k.k_index == bp.start.k.k_index and bk.end.k.k_index == bp.end.k.k_index:
            bound_match += 1
    print(f"  confirm-no-strict boundary matches: {bound_match}/{bound_total}")
    
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
            bo = bis_open[j] if j < len(bis_open) else None
            bo_s = f"{bo.type} {bo.start.k.k_index}→{bo.end.k.k_index}" if bo else "---"
            print(f"  First divergence at bi[{j}]:")
            for k in range(j, min(j+5, min(len(bis_new), len(bis_pya)))):
                bn = bis_new[k]; bp2 = bis_pya[k]
                bo2 = bis_open[k] if k < len(bis_open) else None
                bo_s2 = f"{bo2.type} {bo2.start.k.k_index}→{bo2.end.k.k_index}" if bo2 else "---"
                print(f"    [{k}] new={bn.type} {bn.start.k.k_index}→{bn.end.k.k_index}  "
                      f"pya={bp2.type} {bp2.start.k.k_index}→{bp2.end.k.k_index}  "
                      f"open={bo_s2}")
            break
