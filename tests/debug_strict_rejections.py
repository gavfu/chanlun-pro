"""Instrument _build_bis to log every _bi_fx_valid call, then compare with pyarmor BIs
to find which specific rejected calls cause divergence."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P
from chanlun.cl_interface import FX, BI, Config

CL_CONFIG = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
}

# For each divergent bi, trace what pyarmor accepted that open rejected
# Key: find the first FX pair that pyarmor accepts but open rejects

datasets = [
    ("ETH5m", "tests/test_data/ETH_USDT_5m_1000.parquet"),
]

for name, path in datasets:
    print(f"\n{'='*60}")
    print(f" {name}: Tracing ALL strict check rejections that pyarmor would accept")
    print(f"{'='*60}")
    
    df = pd.read_parquet(path)
    cd_o = CL_O("test", "test", config=CL_CONFIG)
    cd_o.process_klines(df)
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    
    fxs = cd_o.get_fxs()
    bis_p = cd_p.get_bis()
    
    qj = cd_o.fx_qj
    qy = cd_o.fx_qy
    
    # Build a set of pyarmor BI endpoints: (start_k_index, end_k_index)
    pyarmor_bis = set()
    for bi in bis_p:
        pyarmor_bis.add((bi.start.k.k_index, bi.end.k.k_index))
    
    # Now, for every pair of FXes (fx_i, fx_j) where fx_i and fx_j are alternating types,
    # check if _bi_fx_valid rejects it due to strict check, and if pyarmor has a BI 
    # starting at or near fx_i
    
    # Actually, let's just find ALL pairs where strict check causes rejection
    # and their cl_gap, k_gap, and which condition fails
    rejected = []
    for i in range(len(fxs)):
        for j in range(i+1, min(i+15, len(fxs))):  # only nearby pairs
            fx_a = fxs[i]
            fx_b = fxs[j]
            if fx_a.type == fx_b.type:
                continue
            cl_gap = fx_b.k.index - fx_a.k.index
            k_gap = fx_b.k.k_index - fx_a.k.k_index
            
            # Would pass gap check?
            if cl_gap < 4:
                continue  # gap check rejects, not strict
            
            # Would be in strict check range?
            if k_gap >= cd_o.fx_check_k_nums:
                continue  # not in strict range
            
            # Check strict conditions
            if cd_o.allow_bi_fx_strict:
                if fx_a.type == "ding" and fx_b.type == "di":
                    cond1 = fx_a.low(qj, qy) < fx_b.low(qj, qy)
                    cond2 = fx_b.high(qj, qy) > fx_a.high(qj, qy)
                elif fx_a.type == "di" and fx_b.type == "ding":
                    cond1 = fx_a.high(qj, qy) > fx_b.high(qj, qy)
                    cond2 = fx_b.low(qj, qy) < fx_a.low(qj, qy)
                else:
                    continue
                
                if cond1 or cond2:
                    rejected.append({
                        'fx_a': fx_a, 'fx_b': fx_b,
                        'cl_gap': cl_gap, 'k_gap': k_gap,
                        'cond1': cond1, 'cond2': cond2,
                        'fxi': i, 'fxj': j,
                    })
    
    print(f"\nTotal strict-check rejections: {len(rejected)}")
    print(f"\nDetails:")
    for r in rejected:
        fa, fb = r['fx_a'], r['fx_b']
        c1, c2 = r['cond1'], r['cond2']
        
        bi_dir = "down" if fa.type == "ding" else "up"
        
        # Check if this rejected pair's fxes appear as endpoints in pyarmor
        # Actually check if pyarmor has a BI spanning these particular fxes
        in_pyarmor = (fa.k.k_index, fb.k.k_index) in pyarmor_bis
        
        label = "IN_PYARMOR" if in_pyarmor else ""
        
        if fa.type == "ding":
            fa_val_str = f"fa.low={fa.low(qj,qy):.2f} fb.low={fb.low(qj,qy):.2f}"
            fb_val_str = f"fb.high={fb.high(qj,qy):.2f} fa.high={fa.high(qj,qy):.2f}"
        else:
            fa_val_str = f"fa.high={fa.high(qj,qy):.2f} fb.high={fb.high(qj,qy):.2f}"
            fb_val_str = f"fb.low={fb.low(qj,qy):.2f} fa.low={fa.low(qj,qy):.2f}"
        
        fails = []
        if c1: fails.append("C1")
        if c2: fails.append("C2")
        
        print(f"  {bi_dir:>4} {fa.type}@{fa.k.k_index}→{fb.type}@{fb.k.k_index} "
              f"cl={r['cl_gap']} k={r['k_gap']} fails={'+'.join(fails)} "
              f"[{fa_val_str}] [{fb_val_str}] {label}")
