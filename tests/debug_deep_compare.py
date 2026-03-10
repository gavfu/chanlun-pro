"""Deep compare: for each pyarmor BI, check if open's _bi_fx_valid agrees.
For each open BI, check if pyarmor has the same BI.
Focus on the FIRST divergence point in each dataset."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

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
    ("BTC60", "tests/test_data/BTC_USDT_60m_1000.parquet"),
    ("BTC5m", "tests/test_data/BTC_USDT_5m_1000.parquet"),
    ("ETH5m", "tests/test_data/ETH_USDT_5m_1000.parquet"),
]

for name, path in DATASETS:
    df = pd.read_parquet(path)
    cd_o = CL_O("test", "test", config=CL_CONFIG)
    cd_o.process_klines(df)
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    
    bis_o = cd_o.get_bis()
    bis_p = cd_p.get_bis()
    fxs = cd_o.get_fxs()
    qj = cd_o.fx_qj
    qy = cd_o.fx_qy
    
    print(f"\n{'='*70}")
    print(f"=== {name}: Open={len(bis_o)} BIs, Pyarmor={len(bis_p)} BIs ===")
    
    # Find first divergent BI
    min_len = min(len(bis_o), len(bis_p))
    for j in range(min_len):
        bo = bis_o[j]
        bp = bis_p[j]
        if bo.start.k.k_index != bp.start.k.k_index or bo.end.k.k_index != bp.end.k.k_index:
            print(f"\nFirst divergence at bi[{j}]:")
            print(f"  Open:   {bo.type} k={bo.start.k.k_index}→{bo.end.k.k_index}")
            print(f"  Pyarmor:{bp.type} k={bp.start.k.k_index}→{bp.end.k.k_index}")
            
            # Show preceding BI for context
            if j > 0:
                print(f"  Prev Open:   bi[{j-1}] {bis_o[j-1].type} k={bis_o[j-1].start.k.k_index}→{bis_o[j-1].end.k.k_index}")
                print(f"  Prev Pyarmor:bi[{j-1}] {bis_p[j-1].type} k={bis_p[j-1].start.k.k_index}→{bis_p[j-1].end.k.k_index}")
            
            # Both start at the same point. Show all FXes in the region
            start_k = bp.start.k.k_index
            end_k = max(bo.end.k.k_index, bp.end.k.k_index) + 10
            
            print(f"\n  FXes from k={start_k} to k={end_k}:")
            start_fx_obj = None
            region_fxs = []
            for fx in fxs:
                if fx.k.k_index == start_k and fx.type == bp.start.type:
                    start_fx_obj = fx
                if start_k <= fx.k.k_index <= end_k:
                    region_fxs.append(fx)
                    cl_idx = fx.k.index
                    print(f"    {fx.type:>4}@{fx.k.k_index} val={fx.val:.2f} ck={cl_idx} "
                          f"high={fx.high(qj,qy):.2f} low={fx.low(qj,qy):.2f}")
            
            if start_fx_obj is None:
                print("  Could not find start_fx!")
                break
            
            # For each opposite-type FX, check _bi_fx_valid from start
            print(f"\n  Primary checks from {start_fx_obj.type}@{start_fx_obj.k.k_index}:")
            for fx in region_fxs:
                if fx.type != start_fx_obj.type:
                    cl_gap = fx.k.index - start_fx_obj.k.index
                    k_gap = fx.k.k_index - start_fx_obj.k.k_index
                    valid = cd_o._bi_fx_valid(start_fx_obj, fx)
                    
                    # Manually check individual conditions
                    gap_cl = cl_gap >= 4
                    gap_k = k_gap >= 4
                    
                    # Strict check
                    strict = True
                    strict_reason = ""
                    if k_gap < 13:
                        if start_fx_obj.type == "di" and fx.type == "ding":
                            if start_fx_obj.high(qj, qy) > fx.high(qj, qy):
                                strict = False
                                strict_reason = f"C1: start.high({start_fx_obj.high(qj,qy):.2f}) > end.high({fx.high(qj,qy):.2f})"
                            elif fx.low(qj, qy) < start_fx_obj.low(qj, qy):
                                strict = False
                                strict_reason = f"C2: end.low({fx.low(qj,qy):.2f}) < start.low({start_fx_obj.low(qj,qy):.2f})"
                        elif start_fx_obj.type == "ding" and fx.type == "di":
                            if start_fx_obj.low(qj, qy) < fx.low(qj, qy):
                                strict = False
                                strict_reason = f"C1: start.low({start_fx_obj.low(qj,qy):.2f}) < end.low({fx.low(qj,qy):.2f})"
                            elif fx.high(qj, qy) > start_fx_obj.high(qj, qy):
                                strict = False
                                strict_reason = f"C2: end.high({fx.high(qj,qy):.2f}) > start.high({start_fx_obj.high(qj,qy):.2f})"
                    
                    print(f"    → {fx.type}@{fx.k.k_index}: cl={cl_gap} k={k_gap} "
                          f"gap_cl={gap_cl} gap_k={gap_k} strict={strict} valid={valid}")
                    if not strict:
                        print(f"      {strict_reason}")
            
            # For pyarmor's end_fx, check confirmation from that end_fx
            pya_end_k = bp.end.k.k_index
            pya_end_fx = None
            for fx in fxs:
                if fx.k.k_index == pya_end_k:
                    pya_end_fx = fx
                    break
            
            if pya_end_fx:
                print(f"\n  Confirmation checks from {pya_end_fx.type}@{pya_end_fx.k.k_index}:")
                for fx in region_fxs:
                    if fx.k.k_index > pya_end_k and fx.type != pya_end_fx.type:
                        cl_gap_c = fx.k.index - pya_end_fx.k.index
                        k_gap_c = fx.k.k_index - pya_end_fx.k.k_index
                        valid_c = cd_o._bi_fx_valid(pya_end_fx, fx)
                        
                        gap_cl_c = cl_gap_c >= 4
                        gap_k_c = k_gap_c >= 4
                        
                        strict_c = True
                        if k_gap_c < 13:
                            if pya_end_fx.type == "ding" and fx.type == "di":
                                if pya_end_fx.low(qj, qy) < fx.low(qj, qy):
                                    strict_c = False
                                elif fx.high(qj, qy) > pya_end_fx.high(qj, qy):
                                    strict_c = False
                            elif pya_end_fx.type == "di" and fx.type == "ding":
                                if pya_end_fx.high(qj, qy) > fx.high(qj, qy):
                                    strict_c = False
                                elif fx.low(qj, qy) < pya_end_fx.low(qj, qy):
                                    strict_c = False
                        
                        print(f"    → {fx.type}@{fx.k.k_index}: cl={cl_gap_c} k={k_gap_c} "
                              f"gap_cl={gap_cl_c} gap_k={gap_k_c} strict={strict_c} valid={valid_c}")
            
            break  # Only show first divergence per dataset
