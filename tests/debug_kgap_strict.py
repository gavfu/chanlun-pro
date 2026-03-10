"""Test: k_gap >= 4 as gap threshold (instead of cl_gap >= 4) with standard strict check.

Key insight: ALL pyarmor BIs have k_gap >= 4. Previous k_gap test used k_gap WITHOUT
strict check and created 14 extras. Now testing k_gap WITH strict check.
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

def build_bis_kgap(cd, fxs):
    """Build BIs using k_gap >= 4 as gap check + standard strict."""
    bis = []
    if len(fxs) < 2: return bis
    qj = cd.fx_qj; qy = cd.fx_qy
    
    def check_valid(sfx, efx):
        if sfx.type == efx.type: return False
        k_gap = efx.k.k_index - sfx.k.k_index
        # Use k_gap instead of cl_gap
        if k_gap < 4: return False
        # Standard strict check (unchanged)
        if k_gap < 13:
            if sfx.type == "ding" and efx.type == "di":
                if sfx.low(qj, qy) < efx.low(qj, qy): return False
                if efx.high(qj, qy) > sfx.high(qj, qy): return False
            elif sfx.type == "di" and efx.type == "ding":
                if sfx.high(qj, qy) > efx.high(qj, qy): return False
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
    bis_kgap = build_bis_kgap(cd_o, fxs)
    bis_pya = cd_p.get_bis()
    bis_open = cd_o.get_bis()
    
    print(f"\n{'='*60}")
    print(f"=== {name} ===")
    print(f"  pyarmor: {len(bis_pya)} BIs")
    print(f"  baseline (cl_gap): {len(bis_open)} BIs")
    print(f"  k_gap variant: {len(bis_kgap)} BIs")
    
    # Count matches with pyarmor
    bound_match = 0
    bound_total = min(len(bis_kgap), len(bis_pya))
    for j in range(bound_total):
        bk = bis_kgap[j]; bp = bis_pya[j]
        if bk.start.k.k_index == bp.start.k.k_index and bk.end.k.k_index == bp.end.k.k_index:
            bound_match += 1
    print(f"  k_gap boundary matches: {bound_match}/{bound_total}")
    
    # Count baseline matches
    base_match = 0
    base_total = min(len(bis_open), len(bis_pya))
    for j in range(base_total):
        bo = bis_open[j]; bp = bis_pya[j]
        if bo.start.k.k_index == bp.start.k.k_index and bo.end.k.k_index == bp.end.k.k_index:
            base_match += 1
    print(f"  baseline boundary matches: {base_match}/{base_total}")
    
    # Show divergences
    first_div = True
    for j in range(min(len(bis_kgap), len(bis_pya))):
        bk = bis_kgap[j]; bp = bis_pya[j]
        if bk.start.k.k_index != bp.start.k.k_index or bk.end.k.k_index != bp.end.k.k_index:
            if first_div:
                print(f"  First divergence at bi[{j}]:")
                first_div = False
            if j < min(len(bis_kgap), len(bis_pya)) and j - (0 if first_div else 0) < 5:
                bo = bis_open[j] if j < len(bis_open) else None
                bo_s = f"{bo.type} {bo.start.k.k_index}→{bo.end.k.k_index}" if bo else "---"
                print(f"    [{j}] kgap={bk.type} {bk.start.k.k_index}→{bk.end.k.k_index}  "
                      f"pya={bp.type} {bp.start.k.k_index}→{bp.end.k.k_index}  "
                      f"open={bo_s}")
