"""诊断 MACD hist 差异 (小数据集)"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

# 用小数据集更快
df = pd.read_parquet('tests/test_data/BTC_USDT_60m_1000.parquet')
cfg = {
    'bi_type': 'bi_type_old', 'fx_qj': 'fx_qj_k', 'fx_qy': 'fx_qy_three',
    'bi_fx_cgd': 'bi_fx_cgd_yes', 'fx_check_k_nums': 13,
    'bi_split_k_cross_nums': '20,1', 'xd_bzh': 'xd_bzh_no',
}

co = CL_O('BTC/USDT', '60m', config=cfg)
co.process_klines(df)
cp = CL_P('BTC/USDT', '60m', config=cfg)
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
print(f"\npyarmor hist vs 2*(dif-dea): {np.max(np.abs(hp - computed_p)):.2e}")

# open hist (talib) vs dif-dea without factor
computed_o1 = do_ - eo
print(f"open hist vs (dif-dea): {np.max(np.abs(ho - computed_o1)):.2e}")

# So if talib hist = 2*(dif-dea) and pyarmor hist = 2*(dif-dea), both should match
# But maybe talib uses hist = 2*(dif-dea) while pyarmor uses hist = dif-dea, or vice versa

# Check specific values
for i in [40, 50, 100, 200, 500]:
    if i < len(ho):
        print(f"  i={i}: open_hist={ho[i]:.6f} pyarmor_hist={hp[i]:.6f} dif-dea={do_[i]-eo[i]:.6f} 2*(dif-dea)={2*(do_[i]-eo[i]):.6f}")
