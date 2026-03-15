import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P
CL_CONFIG = {'bi_type': 'bi_type_old', 'fx_qj': 'fx_qj_k', 'fx_qy': 'fx_qy_three',
    'bi_fx_cgd': 'bi_fx_cgd_yes', 'fx_check_k_nums': 13,
    'bi_split_k_cross_nums': '20,1', 'xd_bzh': 'xd_bzh_no'}
df = pd.read_parquet('tests/test_data/BTC_USDT_60m_1000.parquet')
co = CL_O('t', 't', config=CL_CONFIG)
co.process_klines(df)
cp = CL_P('t', 't', config=CL_CONFIG)
cp.process_klines(df)

pre = co._build_bis(co.get_fxs())
print("=== BTC60 pre-split bis (k=340..420) ===")
for i, bi in enumerate(pre):
    if 340 <= bi.start.k.k_index <= 420 or 340 <= bi.end.k.k_index <= 420:
        print(f"  pre[{i:>2}] {bi.type:>4} k={bi.start.k.k_index:>4}->{bi.end.k.k_index:>4}")

print("\n=== BTC60 cl_open post-split (k=340..420) ===")
for i, bi in enumerate(co.get_bis()):
    if 340 <= bi.start.k.k_index <= 420 or 340 <= bi.end.k.k_index <= 420:
        s = ' SPLIT' if bi.is_split else ''
        print(f"  open[{i:>2}] {bi.type:>4} k={bi.start.k.k_index:>4}->{bi.end.k.k_index:>4}{s}")

print("\n=== BTC60 pyarmor post-split (k=340..420) ===")
for i, bi in enumerate(cp.get_bis()):
    if 340 <= bi.start.k.k_index <= 420 or 340 <= bi.end.k.k_index <= 420:
        s = f' SPLIT={bi.is_split[:40]}' if bi.is_split else ''
        print(f"  pyar[{i:>2}] {bi.type:>4} k={bi.start.k.k_index:>4}->{bi.end.k.k_index:>4}{s}")
