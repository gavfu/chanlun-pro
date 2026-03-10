"""Check BTC5m around k=183: why does the k_gap variant create down 183→187 
but pyarmor does not?"""
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

df = pd.read_parquet("tests/test_data/BTC_USDT_5m_1000.parquet")
cd_o = CL_O("test", "test", config=CL_CONFIG)
cd_o.process_klines(df)
cd_p = CL_P("test", "test", config=CL_CONFIG)
cd_p.process_klines(df)

fxs = cd_o.get_fxs()
qj = cd_o.fx_qj
qy = cd_o.fx_qy

# Show pyarmor BIs around k=183
print("=== Pyarmor BIs [12:18] ===")
for bi in cd_p.get_bis()[12:18]:
    cl = bi.end.k.index - bi.start.k.index
    k = bi.end.k.k_index - bi.start.k.k_index
    print(f"  bi[{bi.index}] {bi.type} {bi.start.k.k_index}→{bi.end.k.k_index} cl={cl} k={k}")

print("\n=== Open BIs [12:18] ===")
for bi in cd_o.get_bis()[12:18]:
    cl = bi.end.k.index - bi.start.k.index
    k = bi.end.k.k_index - bi.start.k.k_index
    print(f"  bi[{bi.index}] {bi.type} {bi.start.k.k_index}→{bi.end.k.k_index} cl={cl} k={k}")

# Show FXes around k=180-210
print("\n=== FXes k=175-210 ===")
for fx in fxs:
    if 175 <= fx.k.k_index <= 210:
        print(f"  {fx.type:>4}@{fx.k.k_index} val={fx.val:.2f} ck={fx.k.index} "
              f"h={fx.high(qj,qy):.2f} l={fx.low(qj,qy):.2f}")

# Both open and pyarmor have bi[13] = up 179→183. Then:
# Pyarmor: bi[14] = down 183→207 (cl=15+, k=24)
# Open: same bi[14] = down 183→207

# So currently, pyarmor also has down 183→207 (not 183→187)!
# The k_gap variant creates 183→187 because with k_gap, ding@183→di@187 has k_gap=4.
# But with cl_gap, cl_gap = di@187.k.index - ding@183.k.index

# Let's check the specific FX pair:
for fx in fxs:
    if fx.k.k_index == 183:
        ding183 = fx
    if fx.k.k_index == 187:
        di187 = fx

print(f"\n=== ding@183 → di@187 ===")
cl_gap = di187.k.index - ding183.k.index
k_gap = 187 - 183
print(f"  cl_gap={cl_gap} k_gap={k_gap}")
print(f"  ding183.high={ding183.high(qj,qy):.2f} low={ding183.low(qj,qy):.2f}")
print(f"  di187.high={di187.high(qj,qy):.2f} low={di187.low(qj,qy):.2f}")

# Check: with cl_gap check → cl=3 < 4 → FAILS → not created as end_fx
# With k_gap check → k=4 >= 4 → PASSES → what about strict?
if k_gap < 13:
    c1 = ding183.low(qj,qy) < di187.low(qj,qy)
    c2 = di187.high(qj,qy) > ding183.high(qj,qy)
    print(f"  Strict C1: ding.low < di.low = {c1}")
    print(f"  Strict C2: di.high > ding.high = {c2}")
    if not c1 and not c2:
        print(f"  Strict: PASS → k_gap variant ACCEPTS this, creating short BI")
    else:
        print(f"  Strict: FAIL → k_gap variant still REJECTS")
