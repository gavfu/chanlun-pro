"""Test a completely different algorithm variant: no_check extension.

In this variant:
- Primary check: still uses _bi_fx_valid (with some gap+strict combo)
- Extension: ALWAYS extend to more extreme same-type FX, WITHOUT checking _bi_fx_valid
- Confirmation: still uses _bi_fx_valid

Combined with different gap+strict variants."""
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

DATASETS = [
    ("BTCd", "tests/test_data/BTC_USDT_d_500.parquet"),
    ("ETH60", "tests/test_data/ETH_USDT_60m_1000.parquet"),
    ("BTC60", "tests/test_data/BTC_USDT_60m_1000.parquet"),
    ("BTC5m", "tests/test_data/BTC_USDT_5m_1000.parquet"),
    ("ETH5m", "tests/test_data/ETH_USDT_5m_1000.parquet"),
]

def build_bis_nocheck_ext(cd, fxs, gap_mode="cl", strict=True):
    """Build BIs where extension doesn't check _bi_fx_valid."""
    bis = []
    if len(fxs) < 2: return bis
    
    qj = cd.fx_qj
    qy = cd.fx_qy
    
    def check_valid(sfx, efx):
        if sfx.type == efx.type: return False
        cl_gap = efx.k.index - sfx.k.index
        k_gap = efx.k.k_index - sfx.k.k_index
        if gap_mode == "cl":
            if cl_gap < 4: return False
        else:
            if k_gap < 4: return False
        if strict and k_gap < 13:
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
                # NO CHECK extension - just extend if more extreme
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

VARIANTS = [
    ("cl", True, "no-check-ext + cl + strict"),
    ("cl", False, "no-check-ext + cl + no strict"),
    ("k", True, "no-check-ext + k_gap + strict"),
    ("k", False, "no-check-ext + k_gap + no strict"),
]

for name, path in DATASETS:
    df = pd.read_parquet(path)
    cd_o = CL_O("test", "test", config=CL_CONFIG)
    cd_o.process_klines(df)
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    
    fxs = cd_o.get_fxs()
    bis_pya = cd_p.get_bis()
    n_pya = len(bis_pya)
    
    print(f"\n{'='*60}")
    print(f"=== {name}: Pyarmor={n_pya} ===")
    
    for gap_mode, strict, desc in VARIANTS:
        bis = build_bis_nocheck_ext(cd_o, fxs, gap_mode, strict)
        cd_o.bis = bis
        bis = cd_o._bi_special_bi_split(bis)
        
        match = "✅" if len(bis) == n_pya else "❌"
        min_len = min(len(bis), len(bis_pya))
        b_match = sum(1 for j in range(min_len)
                     if bis[j].start.k.k_index == bis_pya[j].start.k.k_index
                     and bis[j].end.k.k_index == bis_pya[j].end.k.k_index)
        
        print(f"  {desc:40s}: {len(bis):3d} {match}  bounds={b_match}/{min_len}")
