"""
Test strict check values for FX99->FX100 with different qj/qy combos
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import pandas as pd
from chanlun.cl_open import CL as CLOpen
from chanlun.cl_interface import Config

df = pd.read_parquet(pathlib.Path(__file__).parent / "test_data" / "BTC_USDT_60m_1000.parquet")
c_open = CLOpen("BTC/USDT", "60m", {})
c_open.process_klines(df)

fxs = c_open.fxs
start_fx = fxs[99]
end_fx = fxs[100]

qj_ck = Config.FX_QJ_CK.value
qy_mid = Config.FX_QY_MIDDLE.value
qj_k = Config.FX_QJ_K.value
qy_three = Config.FX_QY_THREE.value

print(f"FX99 type={start_fx.type}, k.h={start_fx.k.h}, k.l={start_fx.k.l}")
print(f"FX100 type={end_fx.type}, k.h={end_fx.k.h}, k.l={end_fx.k.l}")
print()

cl_gap = end_fx.k.index - start_fx.k.index
k_gap = end_fx.k.k_index - start_fx.k.k_index
print(f"cl_gap={cl_gap}, k_gap={k_gap}")
print()

print("With FX_QJ_CK, FX_QY_MIDDLE (middle CLKline):")
sh = start_fx.high(qj_ck, qy_mid)
sl = start_fx.low(qj_ck, qy_mid)
eh = end_fx.high(qj_ck, qy_mid)
el = end_fx.low(qj_ck, qy_mid)
print(f"  start.high={sh}, start.low={sl}")
print(f"  end.high={eh}, end.low={el}")
# di->ding upstroke: start.high > end.high? (should be False to pass)
print(f"  FAIL if start.high({sh}) > end.high({eh})? => {sh > eh}")
print()

print("With FX_QJ_CK, FX_QY_THREE (all 3 CLKlines):")
sh = start_fx.high(qj_ck, qy_three)
sl = start_fx.low(qj_ck, qy_three)
eh = end_fx.high(qj_ck, qy_three)
el = end_fx.low(qj_ck, qy_three)
print(f"  start.high={sh}, start.low={sl}")
print(f"  end.high={eh}, end.low={el}")
print(f"  FAIL if start.high({sh}) > end.high({eh})? => {sh > eh}")
print()

print("With FX_QJ_K, FX_QY_THREE (all raw klines in 3 CLKlines):")
sh = start_fx.high(qj_k, qy_three)
sl = start_fx.low(qj_k, qy_three)
eh = end_fx.high(qj_k, qy_three)
el = end_fx.low(qj_k, qy_three)
print(f"  start.high={sh}, start.low={sl}")
print(f"  end.high={eh}, end.low={el}")
print(f"  FAIL if start.high({sh}) > end.high({eh})? => {sh > eh}")
