"""Test hypothesis: strict check uses directional FX intervals.

When checking start_fx → end_fx for a BI going RIGHT:
- For start_fx: use klines[1:] (center + right) — the part facing the BI direction
- For end_fx: use klines[:2] (left + center) — the part facing the BI direction (back toward start)

Or simpler: just use center kline (klines[1]) for strict check.
Or: use the STANDARD h/l from the FX itself (CK-based, not raw K-line based).

Let's test multiple strict interval variants."""
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

def fx_high_variant(fx, variant):
    """Compute FX high using different k-line ranges."""
    qj, qy = 'fx_qj_k', 'fx_qy_three'
    if variant == "standard":
        return fx.high(qj, qy)
    elif variant == "center":
        # Only center merged K-line's raw K-lines
        ck = fx.klines[1] if fx.klines[1] is not None else fx.k
        return max(k.h for k in ck.klines)
    elif variant == "right":
        # Center + right (klines[1:])
        highs = []
        for ck in fx.klines[1:]:
            if ck is not None:
                highs.extend(k.h for k in ck.klines)
        return max(highs) if highs else fx.high(qj, qy)
    elif variant == "left":
        # Left + center (klines[:2])
        highs = []
        for ck in fx.klines[:2]:
            if ck is not None:
                highs.extend(k.h for k in ck.klines)
        return max(highs) if highs else fx.high(qj, qy)
    elif variant == "ck_three":
        # Use CK (merged K-line) h/l instead of raw K-line
        return max(ck.h for ck in fx.klines if ck is not None)

def fx_low_variant(fx, variant):
    """Compute FX low using different k-line ranges."""
    qj, qy = 'fx_qj_k', 'fx_qy_three'
    if variant == "standard":
        return fx.low(qj, qy)
    elif variant == "center":
        ck = fx.klines[1] if fx.klines[1] is not None else fx.k
        return min(k.l for k in ck.klines)
    elif variant == "right":
        lows = []
        for ck in fx.klines[1:]:
            if ck is not None:
                lows.extend(k.l for k in ck.klines)
        return min(lows) if lows else fx.low(qj, qy)
    elif variant == "left":
        lows = []
        for ck in fx.klines[:2]:
            if ck is not None:
                lows.extend(k.l for k in ck.klines)
        return min(lows) if lows else fx.low(qj, qy)
    elif variant == "ck_three":
        return min(ck.l for ck in fx.klines if ck is not None)


def build_bis_variant(cd, fxs, gap_mode, strict_variant):
    """Build BIs with variant gap and strict checking."""
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
                if _check_valid(cd, start_fx, cur_fx, gap_mode, strict_variant):
                    end_fx = cur_fx
                    end_idx = i
            i += 1
        else:
            if cur_fx.type == end_fx.type:
                if end_fx.type == "di" and cur_fx.val <= end_fx.val:
                    if _check_valid(cd, start_fx, cur_fx, gap_mode, strict_variant):
                        end_fx = cur_fx
                        end_idx = i
                elif end_fx.type == "ding" and cur_fx.val >= end_fx.val:
                    if _check_valid(cd, start_fx, cur_fx, gap_mode, strict_variant):
                        end_fx = cur_fx
                        end_idx = i
                i += 1
            else:
                confirm = _check_valid(cd, end_fx, cur_fx, gap_mode, strict_variant)
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


def _check_valid(cd, start_fx, end_fx, gap_mode, strict_variant):
    """Check BI validity with variant gap and strict."""
    if start_fx.type == end_fx.type:
        return False
    
    cl_gap = end_fx.k.index - start_fx.k.index
    k_gap = end_fx.k.k_index - start_fx.k.k_index
    
    # Gap check
    if gap_mode == "cl":
        if cl_gap < 4:
            return False
    elif gap_mode == "k":
        if k_gap < 4:
            return False
    
    # Strict check
    if strict_variant == "none":
        return True
    
    if k_gap < 13:
        sh = fx_high_variant(start_fx, strict_variant)
        sl = fx_low_variant(start_fx, strict_variant)
        eh = fx_high_variant(end_fx, strict_variant)
        el = fx_low_variant(end_fx, strict_variant)
        
        if start_fx.type == "ding" and end_fx.type == "di":
            if sl < el:
                return False
            if eh > sh:
                return False
        elif start_fx.type == "di" and end_fx.type == "ding":
            if sh > eh:
                return False
            if el < sl:
                return False
    
    return True


VARIANTS = [
    ("cl", "standard", "baseline (cl+standard)"),
    ("k", "standard", "k_gap + standard strict"),
    ("k", "center", "k_gap + center-only strict"),
    ("k", "right", "k_gap + right-half strict"),
    ("k", "none", "k_gap + no strict"),
    ("cl", "center", "cl_gap + center-only strict"),
    ("cl", "right", "cl_gap + right-half strict"),
]

for name, path in DATASETS:
    df = pd.read_parquet(path)
    cd_o = CL_O("test", "test", config=CL_CONFIG)
    cd_o.process_klines(df)
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    
    fxs = cd_o.get_fxs()
    n_pya = len(cd_p.get_bis())
    
    print(f"\n{'='*60}")
    print(f"=== {name}: Pyarmor={n_pya} ===")
    
    for gap_mode, strict_var, desc in VARIANTS:
        bis = build_bis_variant(cd_o, fxs, gap_mode, strict_var)
        # Apply bi_split after building
        cd_o.bis = bis
        bis = cd_o._bi_special_bi_split(bis)
        
        match = "✅" if len(bis) == n_pya else "❌"
        
        # Count boundary matches
        bis_p = cd_p.get_bis()
        min_len = min(len(bis), len(bis_p))
        b_match = sum(1 for j in range(min_len)
                     if bis[j].start.k.k_index == bis_p[j].start.k.k_index
                     and bis[j].end.k.k_index == bis_p[j].end.k.k_index)
        
        print(f"  {desc:40s}: {len(bis):3d} {match}  bounds={b_match}/{min_len}")
