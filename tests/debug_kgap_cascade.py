"""Check how k_gap variant affects the ENTIRE BI sequence, especially around k=875 in BTC5m.
Also check how many ADDITIONAL BIs the k_gap variant creates vs baseline."""
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

def build_bis_variant(cd, fxs, use_k_gap=False, no_strict=False):
    bis = []
    if len(fxs) < 2:
        return bis
    
    start_fx = fxs[0]
    start_idx = 0
    end_fx = None
    end_idx = -1
    has_confirmed = False
    qj = cd.fx_qj
    qy = cd.fx_qy
    
    def check_valid(sfx, efx):
        if sfx.type == efx.type:
            return False
        cl_gap = efx.k.index - sfx.k.index
        k_gap = efx.k.k_index - sfx.k.k_index
        
        if use_k_gap:
            if k_gap < 4: return False
        else:
            if cl_gap < 4: return False
        
        if no_strict:
            return True
        
        if k_gap < 13:
            if sfx.type == "ding" and efx.type == "di":
                if sfx.low(qj, qy) < efx.low(qj, qy): return False
                if efx.high(qj, qy) > sfx.high(qj, qy): return False
            elif sfx.type == "di" and efx.type == "ding":
                if sfx.high(qj, qy) > efx.high(qj, qy): return False
                if efx.low(qj, qy) < sfx.low(qj, qy): return False
        return True
    
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
                if end_fx.type == "di" and cur_fx.val <= end_fx.val:
                    if check_valid(start_fx, cur_fx):
                        end_fx = cur_fx; end_idx = i
                elif end_fx.type == "ding" and cur_fx.val >= end_fx.val:
                    if check_valid(start_fx, cur_fx):
                        end_fx = cur_fx; end_idx = i
                i += 1
            else:
                if check_valid(end_fx, cur_fx):
                    bi_type = "down" if start_fx.type == "ding" else "up"
                    bi = BI(start=start_fx, end=end_fx, _type=bi_type,
                            index=len(bis), default_zs_type="zs_type_bz")
                    bis.append(bi)
                    has_confirmed = True
                    start_fx = end_fx; start_idx = end_idx
                    end_fx = None; end_idx = -1
                    i = start_idx + 1
                else:
                    i += 1
    
    if end_fx is not None:
        bi_type = "down" if start_fx.type == "ding" else "up"
        bi = BI(start=start_fx, end=end_fx, _type=bi_type,
                index=len(bis), default_zs_type="zs_type_bz")
        bis.append(bi)
    
    return bis

# Focus on BTC5m only
df = pd.read_parquet("tests/test_data/BTC_USDT_5m_1000.parquet")
cd_o = CL_O("test", "test", config=CL_CONFIG)
cd_o.process_klines(df)
cd_p = CL_P("test", "test", config=CL_CONFIG)
cd_p.process_klines(df)

fxs = cd_o.get_fxs()

# Build with baseline and k_gap
bis_baseline = build_bis_variant(cd_o, fxs, use_k_gap=False)
bis_kgap = build_bis_variant(cd_o, fxs, use_k_gap=True)
bis_pya = cd_p.get_bis()

print(f"BTC5m: baseline={len(bis_baseline)} k_gap={len(bis_kgap)} pyarmor={len(bis_pya)}")

# Find where baseline and k_gap first diverge
print("\n=== First divergence between baseline and k_gap ===")
for j in range(min(len(bis_baseline), len(bis_kgap))):
    bb = bis_baseline[j]
    bk = bis_kgap[j]
    if bb.start.k.k_index != bk.start.k.k_index or bb.end.k.k_index != bk.end.k.k_index:
        print(f"  bi[{j}]: baseline={bb.type} {bb.start.k.k_index}→{bb.end.k.k_index}"
              f"  k_gap={bk.type} {bk.start.k.k_index}→{bk.end.k.k_index}")
        # Show surrounding BIs
        for k in range(max(0, j-2), min(len(bis_baseline), j+5)):
            if k < len(bis_baseline) and k < len(bis_kgap):
                print(f"    [{k}] base={bis_baseline[k].type} {bis_baseline[k].start.k.k_index}→{bis_baseline[k].end.k.k_index}"
                      f"  kgap={bis_kgap[k].type} {bis_kgap[k].start.k.k_index}→{bis_kgap[k].end.k.k_index}"
                      f"  pya={bis_pya[k].type} {bis_pya[k].start.k.k_index}→{bis_pya[k].end.k.k_index}" if k < len(bis_pya) else "")
        break

# Now check: do the EXTRA k_gap BIs overlap with pyarmor's BIs?
# Find all k_gap BIs that don't exist in baseline — these are the "extra" BIs
print(f"\n=== k_gap extra BIs not in baseline ===")
baseline_set = set((bi.start.k.k_index, bi.end.k.k_index) for bi in bis_baseline)
kgap_extra = [(i, bi) for i, bi in enumerate(bis_kgap) 
              if (bi.start.k.k_index, bi.end.k.k_index) not in baseline_set]

pya_set = set((bi.start.k.k_index, bi.end.k.k_index) for bi in bis_pya)
for i, bi in kgap_extra[:10]:
    in_pya = (bi.start.k.k_index, bi.end.k.k_index) in pya_set
    cl_gap = bi.end.k.index - bi.start.k.index
    k_gap = bi.end.k.k_index - bi.start.k.k_index
    print(f"  [{i}] {bi.type} {bi.start.k.k_index}→{bi.end.k.k_index} cl={cl_gap} k={k_gap} in_pyarmor={in_pya}")

print(f"\nTotal k_gap extra BIs: {len(kgap_extra)}")
print(f"k_gap extra BIs also in pyarmor: {sum(1 for i,bi in kgap_extra if (bi.start.k.k_index, bi.end.k.k_index) in pya_set)}")
