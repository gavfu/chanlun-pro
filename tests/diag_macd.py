"""诊断 MACD hist 差异"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

df = pd.read_parquet('tests/test_data/ETH_USDT_30m_2025.parquet')
cfg = {
    'bi_type': 'bi_type_old', 'fx_qj': 'fx_qj_k', 'fx_qy': 'fx_qy_three',
    'bi_fx_cgd': 'bi_fx_cgd_yes', 'fx_check_k_nums': 13,
    'bi_split_k_cross_nums': '20,1', 'xd_bzh': 'xd_bzh_no',
}

co = CL_O('ETH/USDT', '30m', config=cfg)
co.process_klines(df)
cp = CL_P('ETH/USDT', '30m', config=cfg)
cp.process_klines(df)

ho = np.array(co.get_idx()['macd']['hist'])
hp = np.array(cp.get_idx()['macd']['hist'])
do_ = np.array(co.get_idx()['macd']['dif'])
dp = np.array(cp.get_idx()['macd']['dif'])
eo = np.array(co.get_idx()['macd']['dea'])
ep = np.array(cp.get_idx()['macd']['dea'])

print(f"dif match: max_diff = {np.max(np.abs(do_ - dp)):.2e}")
print(f"dea match: max_diff = {np.max(np.abs(eo - ep)):.2e}")
print(f"hist match: max_diff = {np.max(np.abs(ho - hp)):.2e}")

# pyarmor hist vs 2*(dif-dea)
computed_p = 2 * (dp - ep)
print(f"\npyarmor hist vs 2*(dif-dea): max_diff = {np.max(np.abs(hp - computed_p)):.2e}")

# open hist vs 2*(dif-dea) — talib returns 2*(dif-dea) as hist
computed_o = 2 * (do_ - eo)
print(f"open hist vs 2*(dif-dea): max_diff = {np.max(np.abs(ho - computed_o)):.2e}")

# Is pyarmor hist = 2 * open hist ?
print(f"\npyarmor hist vs 2*open hist: max_diff = {np.max(np.abs(hp - 2*ho)):.2e}")

# Is open hist = 2 * pyarmor hist ?
print(f"open hist vs 2*pyarmor hist: max_diff = {np.max(np.abs(ho - 2*hp)):.2e}")

# Sample values
for i in range(len(ho)):
    if abs(ho[i]) > 1.0 and abs(hp[i]) > 1.0:
        print(f"\nSample i={i}: open hist={ho[i]:.6f}, pyarmor hist={hp[i]:.6f}, ratio p/o={hp[i]/ho[i]:.6f}")
        print(f"  open dif={do_[i]:.6f} dea={eo[i]:.6f}")
        print(f"  pyarmor dif={dp[i]:.6f} dea={ep[i]:.6f}")
        break

# Max diff location
idx = np.argmax(np.abs(ho - hp))
print(f"\nMax diff at i={idx}: open hist={ho[idx]:.6f}, pyarmor hist={hp[idx]:.6f}")
