"""
Diagnostic: Compare BI BC (背驰) between cl_pyarmor and cl_open
"""
import sys
sys.path.insert(0, 'src')

import pandas as pd
from chanlun.cl_interface import CL_Config

cfg = CL_Config()
cfg.kline_zs_query = "cl_kline"
cfg.fx_check_k_nums = 13
cfg.bi_type = "bi_type_old"
cfg.allow_bi_fx_strict = True
cfg.bi_fx_cgd = "bi_fx_cgd_yes"
cfg.bi_split_k_cross_nums = 20
cfg.bi_split_k_cross_rate = 1
cfg.xd_qj = "xd_qj_dd"
cfg.xd_allow_bi_pohuai = "yes"
cfg.xd_allow_split_no_highlow = 1
cfg.bi_zs_type = "zs_type_bz"
cfg.zs_qj = "zs_qj_dd"
cfg.beichi_type = "bc_type_macd"

df = pd.read_parquet("tests/test_data/BTC_USDT_60m_500.parquet")
klines = []
for _, row in df.iterrows():
    from chanlun.cl_interface import Kline
    k = Kline(
        index=int(row['index']),
        date=row['date'],
        h=float(row['h']),
        l=float(row['l']),
        o=float(row['o']),
        c=float(row['c']),
        v=float(row['v']),
    )
    klines.append(k)

# Run pyarmor
from chanlun.cl_pyarmor import CL as CL_pyarmor
from chanlun.cl_open import CL as CL_open

cl_py = CL_pyarmor("BTC/USDT", "60m", cfg)
cl_py.process_klines(klines)

# Switch to cl_open
import chanlun.cl as cl_mod
cl_mod.CL = CL_open

cl_op = CL_open("BTC/USDT", "60m", cfg)
cl_op.process_klines(klines)

print(f"Pyarmor BIs: {len(cl_py.bis)}, Open BIs: {len(cl_op.bis)}")
print()

# Compare BC for each BI
print("=== BI BC Comparison ===")
for i, (bi_py, bi_op) in enumerate(zip(cl_py.bis, cl_op.bis)):
    py_bc = bi_py.bcs
    op_bc = bi_op.bcs
    
    def bc_summary(bcs):
        result = []
        for bc in bcs:
            result.append(f"{bc.bc_type}({'T' if bc.bc else 'F'})")
        return result
    
    py_sum = bc_summary(py_bc)
    op_sum = bc_summary(op_bc)
    
    if py_sum != op_sum:
        print(f"  bi[{i}] {bi_py.type} h={bi_py.high:.1f} l={bi_py.low:.1f}")
        print(f"    pyarmor: {py_sum}")
        print(f"    open:    {op_sum}")

print()
print("=== BI MMD Comparison ===")
for i, (bi_py, bi_op) in enumerate(zip(cl_py.bis, cl_op.bis)):
    py_mmd = [m.name for m in bi_py.mmds]
    op_mmd = [m.name for m in bi_op.mmds]
    if py_mmd != op_mmd:
        print(f"  bi[{i}] {bi_py.type} h={bi_py.high:.1f} l={bi_py.low:.1f}")
        print(f"    pyarmor: {py_mmd}")
        print(f"    open:    {op_mmd}")

print()
print("=== BI ZS Info ===")
for i, zs in enumerate(cl_py.bi_zss):
    print(f"  py ZS[{i}]: zg={zs.zg}, zd={zs.zd}, gg={zs.gg}, dd={zs.dd}, level={zs.level}, lines=[bi[{zs.lines[0].index}]..bi[{zs.lines[-1].index}]]")
for i, zs in enumerate(cl_op.bi_zss):
    print(f"  op ZS[{i}]: zg={zs.zg}, zd={zs.zd}, gg={zs.gg}, dd={zs.dd}, level={zs.level}, lines=[bi[{zs.lines[0].index}]..bi[{zs.lines[-1].index}]]")
