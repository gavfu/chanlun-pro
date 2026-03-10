"""Investigate why cl+start-right breaks ETH60 and BTC60.
Find the FIRST divergence and analyze it."""
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

def fx_right_high(fx):
    """High from klines[1:] raw K-lines"""
    klines_right = [ck for ck in fx.klines[1:] if ck is not None]
    if not klines_right:
        return fx.high('fx_qj_k', 'fx_qy_three')
    return max(k.h for ck in klines_right for k in ck.klines)

def fx_right_low(fx):
    """Low from klines[1:] raw K-lines"""
    klines_right = [ck for ck in fx.klines[1:] if ck is not None]
    if not klines_right:
        return fx.low('fx_qj_k', 'fx_qy_three')
    return min(k.l for ck in klines_right for k in ck.klines)

def check_valid_right(start_fx, end_fx, qj, qy):
    """Check validity with cl_gap and start-right strict."""
    if start_fx.type == end_fx.type:
        return False
    cl_gap = end_fx.k.index - start_fx.k.index
    k_gap = end_fx.k.k_index - start_fx.k.k_index
    if cl_gap < 4:
        return False
    if k_gap < 13:
        sh = fx_right_high(start_fx)
        sl = fx_right_low(start_fx)
        eh = end_fx.high(qj, qy)
        el = end_fx.low(qj, qy)
        if start_fx.type == "ding" and end_fx.type == "di":
            if sl < el: return False
            if eh > sh: return False
        elif start_fx.type == "di" and end_fx.type == "ding":
            if sh > eh: return False
            if el < sl: return False
    return True

def build_bis_right(cd, fxs, qj, qy):
    bis = []
    if len(fxs) < 2: return bis
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
                if check_valid_right(start_fx, cur_fx, qj, qy):
                    end_fx = cur_fx; end_idx = i
            i += 1
        else:
            if cur_fx.type == end_fx.type:
                if end_fx.type == "di" and cur_fx.val <= end_fx.val:
                    if check_valid_right(start_fx, cur_fx, qj, qy):
                        end_fx = cur_fx; end_idx = i
                elif end_fx.type == "ding" and cur_fx.val >= end_fx.val:
                    if check_valid_right(start_fx, cur_fx, qj, qy):
                        end_fx = cur_fx; end_idx = i
                i += 1
            else:
                if check_valid_right(end_fx, cur_fx, qj, qy):
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

# Test ETH60
for dataset, path in [("ETH60", "tests/test_data/ETH_USDT_60m_1000.parquet"),
                       ("BTC60", "tests/test_data/BTC_USDT_60m_1000.parquet")]:
    df = pd.read_parquet(path)
    cd_o = CL_O("test", "test", config=CL_CONFIG)
    cd_o.process_klines(df)
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    
    fxs = cd_o.get_fxs()
    qj = cd_o.fx_qj
    qy = cd_o.fx_qy
    
    bis_right = build_bis_right(cd_o, fxs, qj, qy)
    bis_baseline = cd_o.get_bis()
    bis_pya = cd_p.get_bis()
    
    print(f"\n{'='*60}")
    print(f"=== {dataset}: baseline={len(bis_baseline)} right={len(bis_right)} pyarmor={len(bis_pya)} ===")
    
    # Find first divergence between right and baseline
    for j in range(min(len(bis_right), len(bis_baseline))):
        br = bis_right[j]
        bb = bis_baseline[j]
        if br.start.k.k_index != bb.start.k.k_index or br.end.k.k_index != bb.end.k.k_index:
            print(f"\n  First diff at bi[{j}]:")
            print(f"    baseline: {bb.type} {bb.start.k.k_index}→{bb.end.k.k_index}")
            print(f"    right:    {br.type} {br.start.k.k_index}→{br.end.k.k_index}")
            if j < len(bis_pya):
                bp = bis_pya[j]
                print(f"    pyarmor:  {bp.type} {bp.start.k.k_index}→{bp.end.k.k_index}")
            
            # Show context
            for k in range(max(0,j-2), min(min(len(bis_right), len(bis_baseline)), j+5)):
                bb2 = bis_baseline[k] if k < len(bis_baseline) else None
                br2 = bis_right[k] if k < len(bis_right) else None
                bp2 = bis_pya[k] if k < len(bis_pya) else None
                baseline_s = f"{bb2.type} {bb2.start.k.k_index}→{bb2.end.k.k_index}" if bb2 else "---"
                right_s = f"{br2.type} {br2.start.k.k_index}→{br2.end.k.k_index}" if br2 else "---"
                pya_s = f"{bp2.type} {bp2.start.k.k_index}→{bp2.end.k.k_index}" if bp2 else "---"
                print(f"    [{k}] base={baseline_s}  right={right_s}  pya={pya_s}")
            
            # Why does right differ? Check the specific FX pair
            # The right variant creates an extra confirmation
            # Show the FXes in the region
            start_k = min(bb.start.k.k_index, br.start.k.k_index)
            end_k = max(bb.end.k.k_index, br.end.k.k_index) + 5
            
            print(f"\n  FXes from k={start_k} to k={end_k}:")
            for fx in fxs:
                if start_k <= fx.k.k_index <= end_k:
                    h_all = fx.high(qj, qy)
                    l_all = fx.low(qj, qy)
                    h_right = fx_right_high(fx)
                    l_right = fx_right_low(fx)
                    print(f"    {fx.type:>4}@{fx.k.k_index} val={fx.val:.2f} "
                          f"h_all={h_all:.2f} h_right={h_right:.2f} "
                          f"l_all={l_all:.2f} l_right={l_right:.2f}")
            
            break
