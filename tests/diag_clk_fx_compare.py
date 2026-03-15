"""
Check if CLKlines or FX attributes differ subtly between cl_open and cl_pyarmor
for the datasets where pre-split bis diverge (BTC5m, ETH5m).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

CL_CONFIG = {
    'bi_type': 'bi_type_old', 'fx_qj': 'fx_qj_k', 'fx_qy': 'fx_qy_three',
    'bi_fx_cgd': 'bi_fx_cgd_yes', 'fx_check_k_nums': 13,
    'bi_split_k_cross_nums': '20,1', 'xd_bzh': 'xd_bzh_no',
}

for dataset_name, path in [
    ('BTC5m', 'tests/test_data/BTC_USDT_5m_1000.parquet'),
    ('ETH5m', 'tests/test_data/ETH_USDT_5m_1000.parquet'),
]:
    df = pd.read_parquet(path)
    co = CL_O("test", "test", config=CL_CONFIG)
    co.process_klines(df)
    cp = CL_P("test", "test", config=CL_CONFIG)
    cp.process_klines(df)

    # 1. Check cl_klines count and values
    clk_o = co.get_cl_klines()
    clk_p = cp.get_cl_klines()
    print(f"\n{'='*60}")
    print(f"{dataset_name}")
    print(f"  cl_klines: open={len(clk_o)} pyarmor={len(clk_p)} match={len(clk_o)==len(clk_p)}")
    
    if len(clk_o) == len(clk_p):
        clk_diffs = []
        for i in range(len(clk_o)):
            ko, kp = clk_o[i], clk_p[i]
            if ko.k_index != kp.k_index or abs(ko.h - kp.h) > 1e-10 or abs(ko.l - kp.l) > 1e-10 or ko.index != kp.index:
                clk_diffs.append(i)
        print(f"  cl_kline diffs: {len(clk_diffs)}")
        for d in clk_diffs[:5]:
            ko, kp = clk_o[d], clk_p[d]
            print(f"    CLK[{d}]: open(idx={ko.index} k_idx={ko.k_index} h={ko.h} l={ko.l}) vs pyarmor(idx={kp.index} k_idx={kp.k_index} h={kp.h} l={kp.l})")
    
    # 2. Check FX count, types, and high/low values
    fxs_o = co.get_fxs()
    fxs_p = cp.get_fxs()
    print(f"  fxs: open={len(fxs_o)} pyarmor={len(fxs_p)} match={len(fxs_o)==len(fxs_p)}")
    
    if len(fxs_o) == len(fxs_p):
        fx_diffs = []
        qj, qy = co.fx_qj, co.fx_qy
        for i in range(len(fxs_o)):
            fo, fp = fxs_o[i], fxs_p[i]
            if (fo.type != fp.type or fo.k.k_index != fp.k.k_index or fo.k.index != fp.k.index
                    or abs(fo.val - fp.val) > 1e-10
                    or abs(fo.high(qj, qy) - fp.high(qj, qy)) > 1e-10
                    or abs(fo.low(qj, qy) - fp.low(qj, qy)) > 1e-10):
                fx_diffs.append(i)
        print(f"  fx diffs (type/k_idx/CLK_idx/val/high/low): {len(fx_diffs)}")
        for d in fx_diffs[:10]:
            fo, fp = fxs_o[d], fxs_p[d]
            print(f"    fx[{d}]: open({fo.type} k={fo.k.k_index} CLK={fo.k.index} val={fo.val:.4f} h={fo.high(qj,qy):.4f} l={fo.low(qj,qy):.4f})")
            print(f"           pyar({fp.type} k={fp.k.k_index} CLK={fp.k.index} val={fp.val:.4f} h={fp.high(qj,qy):.4f} l={fp.low(qj,qy):.4f})")
    
    # 3. Check FX klines CLK indices around the divergent point
    if dataset_name == 'BTC5m':
        target_kidxs = [875, 879]
    else:
        target_kidxs = [333, 337]
    
    print(f"\n  FX klines CLK indices around divergence:")
    for kidx in target_kidxs:
        for i, (fo, fp) in enumerate(zip(fxs_o, fxs_p)):
            if fo.k.k_index == kidx:
                print(f"    fx[{i}] {fo.type} k_idx={kidx}:")
                print(f"      open  klines:", [(kl.k_index, kl.index, f"h={kl.h:.2f}", f"l={kl.l:.2f}") for kl in fo.klines if kl])
                print(f"      pyarm klines:", [(kl.k_index, kl.index, f"h={kl.h:.2f}", f"l={kl.l:.2f}") for kl in fp.klines if kl])
                break
