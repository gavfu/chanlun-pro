"""
Diagnostic: Detailed LD analysis for bi[6] and bi[4]
Check the FX indices used for MACD computation.
"""
import sys
sys.path.insert(0, 'src')
import numpy as np
import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')
config = {}
cd_open = CL_Open('BTC/USDT', '60m', config=config)
cd_open.process_klines(df)
cd_pyarmor = CL_Pyarmor('BTC/USDT', '60m', config=config)
cd_pyarmor.process_klines(df)

for bi_idx, prev_idx in [(6, 4), (4, 2)]:
    print(f"\n=== bi[{bi_idx}] DOWN: compare with bi[{prev_idx}] DOWN ===")
    
    py_bi = cd_pyarmor.bis[bi_idx]
    op_bi = cd_open.bis[bi_idx]
    py_prev = cd_pyarmor.bis[prev_idx]
    op_prev = cd_open.bis[prev_idx]
    
    print(f"  bi[{prev_idx}]:")
    print(f"    py: start_fx.k.k_index={py_prev.start.k.k_index}, end_fx.k.k_index={py_prev.end.k.k_index}")
    print(f"    op: start_fx.k.k_index={op_prev.start.k.k_index}, end_fx.k.k_index={op_prev.end.k.k_index}")
    
    print(f"  bi[{bi_idx}]:")
    print(f"    py: start_fx.k.k_index={py_bi.start.k.k_index}, end_fx.k.k_index={py_bi.end.k.k_index}")
    print(f"    op: start_fx.k.k_index={op_bi.start.k.k_index}, end_fx.k.k_index={op_bi.end.k.k_index}")
    
    # Get MACD histograms for each range
    py_hist_prev = np.array(cd_pyarmor.get_idx()['macd']['hist'][py_prev.start.k.k_index:py_prev.end.k.k_index+1])
    op_hist_prev = np.array(cd_open.get_idx()['macd']['hist'][op_prev.start.k.k_index:op_prev.end.k.k_index+1])
    
    py_hist_bi = np.array(cd_pyarmor.get_idx()['macd']['hist'][py_bi.start.k.k_index:py_bi.end.k.k_index+1])
    op_hist_bi = np.array(cd_open.get_idx()['macd']['hist'][op_bi.start.k.k_index:op_bi.end.k.k_index+1])
    
    # For DOWN line, use down_sum
    py_prev_down_sum = abs(py_hist_prev[py_hist_prev < 0].sum())
    op_prev_down_sum = abs(op_hist_prev[op_hist_prev < 0].sum())
    py_bi_down_sum = abs(py_hist_bi[py_hist_bi < 0].sum())
    op_bi_down_sum = abs(op_hist_bi[op_hist_bi < 0].sum())
    
    print(f"\n  prev down_sum: py={py_prev_down_sum:.4f}, op={op_prev_down_sum:.4f}")
    print(f"  bi   down_sum: py={py_bi_down_sum:.4f}, op={op_bi_down_sum:.4f}")
    
    py_bc = py_bi_down_sum < py_prev_down_sum  
    op_bc = op_bi_down_sum < op_prev_down_sum
    
    print(f"  beichi: py_bc={py_bc}  op_bc={op_bc}")
    
    # Find the histogram values in py vs op to check scale
    print(f"\n  py hist[{py_prev.start.k.k_index}:{py_prev.end.k.k_index}] first 5 neg values: {[h for h in py_hist_prev if h < 0][:5]}")
    print(f"  op hist[{op_prev.start.k.k_index}:{op_prev.end.k.k_index}] first 5 neg values: {[h for h in op_hist_prev if h < 0][:5]}")
    
    print(f"\n  py hist[{py_bi.start.k.k_index}:{py_bi.end.k.k_index}] first 5 neg values: {[h for h in py_hist_bi if h < 0][:5]}")
    print(f"  op hist[{op_bi.start.k.k_index}:{op_bi.end.k.k_index}] first 5 neg values: {[h for h in op_hist_bi if h < 0][:5]}")
