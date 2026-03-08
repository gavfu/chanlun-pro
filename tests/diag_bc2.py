"""
Diagnostic: Detailed analysis of all BI BC comparisons.
Shows compare_ld_beichi results for all same-direction pairs.
"""
import sys
sys.path.insert(0, 'src')
import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor
from chanlun.cl_interface import compare_ld_beichi

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')

config = {}
cd_open = CL_Open('BTC/USDT', '60m', config=config)
cd_open.process_klines(df)

cd_pyarmor = CL_Pyarmor('BTC/USDT', '60m', config=config)
cd_pyarmor.process_klines(df)

# Show all same-direction pairs and bc result
print("=== All same-direction BI pairs - bc result ===")
print(f"{'bi':>4} {'dir':>4} | {'new_ext':>7} | {'bi_bc(T=beichi)':>16} | {'py_has_bi_bc':>12} | {'op_has_bi_bc':>12}")
print("-"*70)

for i in range(2, len(cd_pyarmor.bis)):
    py_bi = cd_pyarmor.bis[i]
    op_bi = cd_open.bis[i]
    py_prev = cd_pyarmor.bis[i-2]
    op_prev = cd_open.bis[i-2]
    
    if py_bi.type != py_prev.type:
        continue  # skip opposite-direction pairs
    
    # new extreme?
    if py_bi.type == 'up':
        new_ext = py_bi.high > py_prev.high
    else:
        new_ext = py_bi.low < py_prev.low
    
    # bc result from pyarmor
    py_ld = py_prev.get_ld(cd_pyarmor)
    py_ld2 = py_bi.get_ld(cd_pyarmor)
    py_bc_result = compare_ld_beichi(py_ld, py_ld2, py_bi.type)
    
    # also from open
    op_ld = op_prev.get_ld(cd_open)
    op_ld2 = op_bi.get_ld(cd_open)
    op_bc_result = compare_ld_beichi(op_ld, op_ld2, op_bi.type)
    
    # does pyarmor actually have bi_bc for this?
    py_has = 'bi' in sorted(py_bi.line_bcs('|'))
    op_has = 'bi' in sorted(op_bi.line_bcs('|'))
    
    # flag differences
    flag = ''
    if py_has != op_has:
        flag = ' <== DIFF'
    if py_bc_result != op_bc_result:
        flag += ' [LD_DIFF!]'
    
    print(f"  bi[{i:2d}] {py_bi.type:>4} | {str(new_ext):>7} | bc={str(py_bc_result):>5}/{str(op_bc_result):>5} | py_has={str(py_has):>5} | op_has={str(op_has):>5}{flag}")

print()
print("=== Summary of 'pz' BC (盘整背驰) diffs ===")
for i, (op, py) in enumerate(zip(cd_open.bis, cd_pyarmor.bis)):
    py_pz = 'pz' in sorted(py.line_bcs('|'))
    op_pz = 'pz' in sorted(op.line_bcs('|'))
    if py_pz != op_pz:
        print(f"  bi[{i}] {op.type}: py_pz={py_pz}, op_pz={op_pz}")

print()
print("=== BI ZS details (for pz analysis) ===")
for i, zs in enumerate(cd_pyarmor.bi_zss):
    lines_str = f"bi[{zs.lines[0].index}..{zs.lines[-1].index}]"
    print(f"  py ZS[{i}]: {lines_str} zg={zs.zg} zd={zs.zd}")
for i, zs in enumerate(cd_open.bi_zss):
    lines_str = f"bi[{zs.lines[0].index}..{zs.lines[-1].index}]"
    print(f"  op ZS[{i}]: {lines_str} zg={zs.zg} zd={zs.zd}")
