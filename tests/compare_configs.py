"""Compare effective config values between cl_open and cl_pyarmor."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), 'test_data', 'ETH_USDT_5m_1000.parquet'))

# Use default config (empty) - same as test suite
cd_o = CL_Open('ETH/USDT', '5m')
cd_o.process_klines(df)

cd_p = CL_Pyarmor('ETH/USDT', '5m')
cd_p.process_klines(df)

# Config attributes to compare
config_attrs = [
    'kline_type', 'fx_qy', 'fx_qj', 'fx_bh',
    'bi_type', 'bi_bzh', 'bi_qj', 'bi_fx_cgd',
    'fx_check_k_nums', 'bi_split_k_cross_nums',
    'allow_bi_fx_strict', 'kline_qk',
    'xd_qj', 'xd_allow_bi_pohuai',
    'xd_allow_split_no_highlow', 'xd_allow_split_zs_kz',
    'xd_allow_split_zs_more_line', 'xd_allow_split_zs_no_direction',
    'xd_zs_max_lines_split',
    'zsd_qj', 'zs_bi_type', 'zs_xd_type', 'zs_qj', 'zs_cd', 'zs_wzgx',
]

print(f"{'Attribute':<40} {'cl_open':<25} {'cl_pyarmor':<25} {'Match?'}")
print("=" * 120)

for attr in config_attrs:
    o_val = getattr(cd_o, attr, '???')
    p_val = getattr(cd_p, attr, '???')
    match = "✅" if str(o_val) == str(p_val) else "❌"
    print(f"{attr:<40} {str(o_val):<25} {str(p_val):<25} {match}")

# Also check any pyarmor-specific attrs we might be missing
print("\n\nPyarmor attributes not in cl_open:")
p_attrs = set(dir(cd_p)) - set(dir(cd_o))
for a in sorted(p_attrs):
    if not a.startswith('_'):
        val = getattr(cd_p, a, '???')
        if not callable(val):
            print(f"  {a} = {val}")

print("\n\ncl_open attributes not in pyarmor:")
o_attrs = set(dir(cd_o)) - set(dir(cd_p))
for a in sorted(o_attrs):
    if not a.startswith('_'):
        val = getattr(cd_o, a, '???')
        if not callable(val):
            print(f"  {a} = {val}")

# Also check bi_split tolerance (the ",1" part)
print("\n\nbi_split_k_cross_nums config details:")
print(f"  cl_open  raw config: {cd_o.cl_config.get('bi_split_k_cross_nums', '(default)')}")
print(f"  cl_open  parsed: bi_split_k_cross_nums={cd_o.bi_split_k_cross_nums}")
# Check if pyarmor has a tolerance attribute
for attr in dir(cd_p):
    if 'toler' in attr.lower() or 'split' in attr.lower() or 'cross' in attr.lower():
        val = getattr(cd_p, attr, '???')
        if not callable(val):
            print(f"  pyarmor  {attr} = {val}")
