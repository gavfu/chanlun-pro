"""
Diagnostic: MACD LD difference for bi[6] vs bi[4].
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

# Focus on bi[6] which has LD_DIFF!
print("=== bi[6] DOWN: compare with bi[4] DOWN ===")
py_bi4 = cd_pyarmor.bis[4]
py_bi6 = cd_pyarmor.bis[6]
op_bi4 = cd_open.bis[4]
op_bi6 = cd_open.bis[6]

print(f"  bi[4]: py h={py_bi4.high} l={py_bi4.low}, op h={op_bi4.high} l={op_bi4.low}")
print(f"  bi[6]: py h={py_bi6.high} l={py_bi6.low}, op h={op_bi6.high} l={op_bi6.low}")

# get the LD
py_ld4 = py_bi4.get_ld(cd_pyarmor)
py_ld6 = py_bi6.get_ld(cd_pyarmor)
op_ld4 = op_bi4.get_ld(cd_open)
op_ld6 = op_bi6.get_ld(cd_open)

print(f"\n  bi[4] LD: py={py_ld4}  op={op_ld4}")
print(f"  bi[6] LD: py={py_ld6}  op={op_ld6}")

py_result = compare_ld_beichi(py_ld4, py_ld6, 'down')
op_result = compare_ld_beichi(op_ld4, op_ld6, 'down')
print(f"\n  compare_ld_beichi: py={py_result}  op={op_result}")

# Show MACD indices for each bi
print(f"\n  bi[4] klines range: start_kl_idx={py_bi4.start_line.fx_mark}...end_kl_idx={py_bi4.end_line.fx_mark}")
print(f"  bi[6] klines range: start={py_bi6.start_line.fx_mark}...end={py_bi6.end_line.fx_mark}")

# Let's look at the LD structure
print(f"\n  py ld4 type: {type(py_ld4)}, val: {py_ld4}")
print(f"  op ld4 type: {type(op_ld4)}, val: {op_ld4}")
print(f"  py ld6 type: {type(py_ld6)}, val: {py_ld6}")
print(f"  op ld6 type: {type(op_ld6)}, val: {op_ld6}")

# Check get_ld method
import inspect
try:
    src = inspect.getsource(cd_open.bis[4].get_ld)
    print(f"\n  get_ld source:\n{src}")
except:
    print("\n  Can't get source for get_ld")
