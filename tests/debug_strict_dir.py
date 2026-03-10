"""Test directional strict: use facing sides of FXes for strict check.

For a BI going from start_fx to end_fx (left→right in time):
- start_fx "faces right": use klines[1:] (center + right kline)
- end_fx "faces left": use klines[:2] (left kline + center)

This means we use the INNER parts of each FX (the sides facing each other)."""
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

def fx_range(fx, side):
    """Get high/low from a specific part of the FX using raw K-lines.
    side: 'all', 'right' (klines[1:]), 'left' (klines[:2]), 'center' (klines[1])
    """
    if side == "all":
        klines_to_use = [ck for ck in fx.klines if ck is not None]
    elif side == "right":
        klines_to_use = [ck for ck in fx.klines[1:] if ck is not None]
    elif side == "left":
        klines_to_use = [ck for ck in fx.klines[:2] if ck is not None]
    elif side == "center":
        ck = fx.klines[1] if fx.klines[1] is not None else fx.k
        klines_to_use = [ck]
    
    highs = [k.h for ck in klines_to_use for k in ck.klines]
    lows = [k.l for ck in klines_to_use for k in ck.klines]
    return max(highs), min(lows)


def _check_valid_dir(cd, start_fx, end_fx, gap_mode, start_side, end_side):
    """Check BI validity with directional strict."""
    if start_fx.type == end_fx.type:
        return False
    
    cl_gap = end_fx.k.index - start_fx.k.index
    k_gap = end_fx.k.k_index - start_fx.k.k_index
    
    if gap_mode == "cl":
        if cl_gap < 4:
            return False
    elif gap_mode == "k":
        if k_gap < 4:
            return False
    
    if k_gap < 13:
        sh, sl = fx_range(start_fx, start_side)
        eh, el = fx_range(end_fx, end_side)
        
        if start_fx.type == "ding" and end_fx.type == "di":
            if sl < el: return False
            if eh > sh: return False
        elif start_fx.type == "di" and end_fx.type == "ding":
            if sh > eh: return False
            if el < sl: return False
    
    return True


def build_bis_dir(cd, fxs, gap_mode, start_side, end_side):
    """Build BIs with directional strict checking."""
    bis = []
    if len(fxs) < 2:
        return bis
    
    start_fx = fxs[0]
    start_idx = 0
    end_fx = None
    end_idx = -1
    has_confirmed = False
    
    i = 1
    while i < len(fxs):
        cur_fx = fxs[i]
        
        if end_fx is None:
            if cur_fx.type == start_fx.type:
                if not has_confirmed:
                    if start_fx.type == "ding" and cur_fx.val > start_fx.val:
                        start_fx = cur_fx
                        start_idx = i
                    elif start_fx.type == "di" and cur_fx.val < start_fx.val:
                        start_fx = cur_fx
                        start_idx = i
            else:
                if _check_valid_dir(cd, start_fx, cur_fx, gap_mode, start_side, end_side):
                    end_fx = cur_fx
                    end_idx = i
            i += 1
        else:
            if cur_fx.type == end_fx.type:
                if end_fx.type == "di" and cur_fx.val <= end_fx.val:
                    if _check_valid_dir(cd, start_fx, cur_fx, gap_mode, start_side, end_side):
                        end_fx = cur_fx
                        end_idx = i
                elif end_fx.type == "ding" and cur_fx.val >= end_fx.val:
                    if _check_valid_dir(cd, start_fx, cur_fx, gap_mode, start_side, end_side):
                        end_fx = cur_fx
                        end_idx = i
                i += 1
            else:
                confirm = _check_valid_dir(cd, end_fx, cur_fx, gap_mode, start_side, end_side)
                if confirm:
                    bi_type = "down" if start_fx.type == "ding" else "up"
                    bi = BI(start=start_fx, end=end_fx, _type=bi_type,
                            index=len(bis), default_zs_type="zs_type_bz")
                    bis.append(bi)
                    has_confirmed = True
                    start_fx = end_fx
                    start_idx = end_idx
                    end_fx = None
                    end_idx = -1
                    i = start_idx + 1
                else:
                    i += 1
    
    if end_fx is not None:
        bi_type = "down" if start_fx.type == "ding" else "up"
        bi = BI(start=start_fx, end=end_fx, _type=bi_type,
                index=len(bis), default_zs_type="zs_type_bz")
        bis.append(bi)
    
    return bis


VARIANTS = [
    # gap_mode, start_side, end_side, description
    ("cl", "all", "all", "baseline"),
    ("cl", "right", "left", "cl + facing sides"),
    ("cl", "right", "all", "cl + start-right, end-all"),
    ("cl", "all", "left", "cl + start-all, end-left"),
    ("cl", "center", "center", "cl + center/center"),
    ("k", "right", "left", "k_gap + facing sides"),
    ("k", "center", "center", "k_gap + center/center"),
    # What if no strict for primary but strict for confirm?
    # Or what if gap=cl but for pairs with cl_gap<4 but k_gap>=4, use no strict?
]

for name, path in DATASETS:
    df = pd.read_parquet(path)
    cd_o = CL_O("test", "test", config=CL_CONFIG)
    cd_o.process_klines(df)
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    
    fxs = cd_o.get_fxs()
    bis_p = cd_p.get_bis()
    n_pya = len(bis_p)
    
    print(f"\n{'='*60}")
    print(f"=== {name}: Pyarmor={n_pya} ===")
    
    for gap_mode, ss, es, desc in VARIANTS:
        bis = build_bis_dir(cd_o, fxs, gap_mode, ss, es)
        cd_o.bis = bis
        bis = cd_o._bi_special_bi_split(bis)
        
        match = "✅" if len(bis) == n_pya else "❌"
        
        min_len = min(len(bis), len(bis_p))
        b_match = sum(1 for j in range(min_len)
                     if bis[j].start.k.k_index == bis_p[j].start.k.k_index
                     and bis[j].end.k.k_index == bis_p[j].end.k.k_index)
        
        print(f"  {desc:40s}: {len(bis):3d} {match}  bounds={b_match}/{min_len}")
