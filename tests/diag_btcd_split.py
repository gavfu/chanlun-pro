"""Debug why BTCd is being split incorrectly"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O

CL_CONFIG = {
    "bi_type": "bi_type_old", "fx_qj": "fx_qj_k", "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes", "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1", "xd_bzh": "xd_bzh_no",
}

df = pd.read_parquet("tests/test_data/BTC_USDT_d_500.parquet")
cl = CL_O("test", "test", config=CL_CONFIG)
cl.process_klines(df)

bis = cl.get_bis()
xds = cl.get_xds()

print("BIS:")
for b in bis[:25]:
    print(f"  bi[{b.index:>2}] {b.type:>4} h={b.high:>12.1f} l={b.low:>12.1f}")

print("\nXDS:")
for xd in xds:
    print(f"  xd[{xd.index:>2}] {xd.type:>4} bi[{xd.start_line.index}→{xd.end_line.index}] split='{xd.is_split}'")

# Check ZS in bi[11→21] (UP segment)
print("\nZS in bi[11→21]:")
seg_bis = bis[11:22]
for b in seg_bis:
    print(f"  bi[{b.index:>2}] {b.type:>4} h={b.high:>12.1f} l={b.low:>12.1f}")

# Try building ZS
zs_list = cl._build_zs_in_range(bis, 11, 21)
for zs in zs_list:
    print(f"  ZS bi[{zs['start_bi']}→{zs['end_bi']}] type={zs['type']} lines={zs['line_num']} dir={'SAME' if zs['type'] == 'up' else 'DIFF'}")
