"""
Test: if strict check uses FX_QJ_CK+FX_QY_MIDDLE for bi_type_old,
do the EXTRA BIs (open creates but pyarmor doesn't) fail?
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import pandas as pd
from chanlun.cl_open import CL as CLOpen
from chanlun.cl_pyarmor import CL as CLPya
from chanlun.cl_interface import Config

qj_ck = Config.FX_QJ_CK.value
qy_mid = Config.FX_QY_MIDDLE.value
qj_k = Config.FX_QJ_K.value
qy_three = Config.FX_QY_THREE.value

def strict_check_type_old(start_fx, end_fx):
    """CK+MIDDLE strict check for bi_type_old"""
    qj, qy = qj_ck, qy_mid
    if start_fx.type == "ding" and end_fx.type == "di":
        if start_fx.low(qj, qy) < end_fx.low(qj, qy):
            return False, f"ding->di: start.low({start_fx.low(qj,qy)}) < end.low({end_fx.low(qj,qy)})"
        if end_fx.high(qj, qy) > start_fx.high(qj, qy):
            return False, f"ding->di: end.high({end_fx.high(qj,qy)}) > start.high({start_fx.high(qj,qy)})"
    elif start_fx.type == "di" and end_fx.type == "ding":
        if start_fx.high(qj, qy) > end_fx.high(qj, qy):
            return False, f"di->ding: start.high({start_fx.high(qj,qy)}) > end.high({end_fx.high(qj,qy)})"
        if end_fx.low(qj, qy) < start_fx.low(qj, qy):
            return False, f"di->ding: end.low({end_fx.low(qj,qy)}) < start.low({start_fx.low(qj,qy)})"
    return True, "pass"

for fname in ["BTC_USDT_60m_500.parquet", "BTC_USDT_60m_1000.parquet"]:
    print(f"\n===== {fname} =====")
    df = pd.read_parquet(pathlib.Path(__file__).parent / "test_data" / fname)
    
    c_open = CLOpen("BTC/USDT", "60m", {})
    c_open.process_klines(df)
    
    c_pya = CLPya("BTC/USDT", "60m", {})
    c_pya.process_klines(df)
    
    open_bis = c_open.bis
    pya_bis = c_pya.bis
    
    # Find open BI endpoints not in pyarmor BIs
    pya_bi_endpoints = {(b.start.index, b.end.index) for b in pya_bis}
    open_extra = [(b, i) for i, b in enumerate(open_bis) if (b.start.index, b.end.index) not in pya_bi_endpoints]
    
    print(f"Open BIs: {len(open_bis)}, Pyarmor BIs: {len(pya_bis)}, Extra in open: {len(open_extra)}")
    
    fxs = c_open.fxs
    
    for b, bi_idx in open_extra[:10]:
        start_fx = b.start
        end_fx = b.end
        cl_gap = end_fx.k.index - start_fx.k.index
        k_gap = end_fx.k.k_index - start_fx.k.k_index
        
        # Test k_gap >= 4 (would pass gap check)
        gap_pass = k_gap >= 4
        # Test strict with CK+MIDDLE
        strict_pass, reason = strict_check_type_old(start_fx, end_fx)
        
        print(f"  Extra bi[{bi_idx}]: FX{start_fx.index}({start_fx.type})->FX{end_fx.index}({end_fx.type}) cl_gap={cl_gap} k_gap={k_gap}")
        print(f"    gap_pass(k>=4)={gap_pass}, strict_ck_mid_pass={strict_pass} ({reason})")
