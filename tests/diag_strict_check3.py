"""
Directly test _bi_fx_valid for known extra BIs to find the right rule.
Compare with known accepted BIs.
Focus: What differentiates FX0->FX1 (rejected by pyarmor) from FX99->FX100 (accepted)?
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

def analyze_fx_pair(fx_start, fx_end, label=""):
    cl_gap = fx_end.k.index - fx_start.k.index
    k_gap = fx_end.k.k_index - fx_start.k.k_index
    fx_gap = fx_end.index - fx_start.index  # FX index difference
    
    sh_ck = fx_start.high(qj_ck, qy_mid)
    sl_ck = fx_start.low(qj_ck, qy_mid)
    eh_ck = fx_end.high(qj_ck, qy_mid)
    el_ck = fx_end.low(qj_ck, qy_mid)
    
    sh_k = fx_start.high(qj_k, qy_three)
    sl_k = fx_start.low(qj_k, qy_three)
    eh_k = fx_end.high(qj_k, qy_three)
    el_k = fx_end.low(qj_k, qy_three)
    
    print(f"  {label}: FX{fx_start.index}({fx_start.type})->FX{fx_end.index}({fx_end.type})")
    print(f"    cl_gap={cl_gap}, k_gap={k_gap}, fx_gap={fx_gap}")
    print(f"    CK+MID: start_h={sh_ck}, start_l={sl_ck}, end_h={eh_ck}, end_l={el_ck}")
    print(f"    K+THREE: start_h={sh_k}, start_l={sl_k}, end_h={eh_k}, end_l={el_k}")
    
    if fx_start.type == "di" and fx_end.type == "ding":
        print(f"    CK+MID strict(up): start_h({sh_ck}) > end_h({eh_ck})? {sh_ck > eh_ck}")
        print(f"    CK+MID strict(up): end_l({el_ck}) < start_l({sl_ck})? {el_ck < sl_ck}")
    else:
        print(f"    CK+MID strict(dn): start_l({sl_ck}) < end_l({el_ck})? {sl_ck < el_ck}")
        print(f"    CK+MID strict(dn): end_h({eh_ck}) > start_h({sh_ck})? {eh_ck > sh_ck}")
    print()

print("=== 1000k dataset ===")
df = pd.read_parquet(pathlib.Path(__file__).parent / "test_data" / "BTC_USDT_60m_1000.parquet")
c_open = CLOpen("BTC/USDT", "60m", {})
c_open.process_klines(df)

fxs = c_open.fxs

print("ACCEPTED by pyarmor:")
analyze_fx_pair(fxs[99], fxs[100], "FX99->100 (1000k, bi[15] accepted)")
analyze_fx_pair(fxs[138], fxs[139], "FX138->139 (1000k, bi[24] accepted)")
analyze_fx_pair(fxs[151], fxs[152], "FX151->152 (1000k, bi[27] accepted)")
analyze_fx_pair(fxs[177], fxs[178], "FX177->178 (1000k, bi[33] accepted)")

print("EXTRA in open (rejected by pyarmor):")
analyze_fx_pair(fxs[29], fxs[30], "FX29->30 (1000k, extra bi[3])")
analyze_fx_pair(fxs[39], fxs[40], "FX39->40 (1000k, extra bi[5], cl_gap=1,k_gap=4)")
analyze_fx_pair(fxs[81], fxs[82], "FX81->82 (1000k, extra bi[15])")
analyze_fx_pair(fxs[123], fxs[124], "FX123->124 (1000k, extra bi[23])")

print("=== 500k dataset ===")
df500 = pd.read_parquet(pathlib.Path(__file__).parent / "test_data" / "BTC_USDT_60m_500.parquet")
c_open500 = CLOpen("BTC/USDT", "60m", {})
c_open500.process_klines(df500)
fxs500 = c_open500.fxs

print("EXTRA in open (rejected by pyarmor):")
analyze_fx_pair(fxs500[0], fxs500[1], "FX0->1 (500k, extra bi[0], cl_gap=3,k_gap=4)")
analyze_fx_pair(fxs500[40], fxs500[41], "FX40->41 (500k, extra bi[10], cl_gap=3,k_gap=4)")
analyze_fx_pair(fxs500[67], fxs500[68], "FX67->68 (500k, extra bi[15], cl_gap=3,k_gap=5)")
