"""Check no-check-ext for BTC5m around bi[61]."""
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

def build_bis_nocheck_ext(cd, fxs):
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
                # NO CHECK extension
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

df = pd.read_parquet("tests/test_data/BTC_USDT_5m_1000.parquet")
cd_o = CL_O("test", "test", config=CL_CONFIG)
cd_o.process_klines(df)
cd_p = CL_P("test", "test", config=CL_CONFIG)
cd_p.process_klines(df)

fxs = cd_o.get_fxs()
bis_baseline = cd_o.get_bis()
bis_nocheck = build_bis_nocheck_ext(cd_o, fxs)
bis_pya = cd_p.get_bis()

print("=== BTC5m: Comparing around bi[59-66] ===")
for j in range(59, min(66, len(bis_baseline))):
    bb = bis_baseline[j]
    bn = bis_nocheck[j] if j < len(bis_nocheck) else None
    bp = bis_pya[j] if j < len(bis_pya) else None
    base_s = f"{bb.type} {bb.start.k.k_index}→{bb.end.k.k_index}"
    nc_s = f"{bn.type} {bn.start.k.k_index}→{bn.end.k.k_index}" if bn else "---"
    pya_s = f"{bp.type} {bp.start.k.k_index}→{bp.end.k.k_index}" if bp else "---"
    
    cl_b = bb.end.k.index - bb.start.k.index
    k_b = bb.end.k.k_index - bb.start.k.k_index
    
    print(f"  [{j}] base={base_s:25s} nocheck={nc_s:25s} pya={pya_s}")

# Check which boundaries changed
print(f"\n=== Boundary diffs: baseline vs nocheck ===")
for j in range(min(len(bis_baseline), len(bis_nocheck))):
    bb = bis_baseline[j]
    bn = bis_nocheck[j]
    if bb.start.k.k_index != bn.start.k.k_index or bb.end.k.k_index != bn.end.k.k_index:
        bp = bis_pya[j] if j < len(bis_pya) else None
        pya_s = f"{bp.type} {bp.start.k.k_index}→{bp.end.k.k_index}" if bp else "---"
        print(f"  [{j}] base={bb.type} {bb.start.k.k_index}→{bb.end.k.k_index} "
              f"→ nocheck={bn.type} {bn.start.k.k_index}→{bn.end.k.k_index} "
              f"pya={pya_s}")
        # Did nocheck match pyarmor?
        if bp and bn.start.k.k_index == bp.start.k.k_index and bn.end.k.k_index == bp.end.k.k_index:
            print(f"    ^^^ MATCHES PYARMOR!")
