"""
Diagnose BI divergence in BTC/USDT 60m 1000 klines.
First divergence: bi[15] high: 71524.9 vs 70000.0
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import pandas as pd
from chanlun.cl_open import CL as CLOpen
from chanlun.cl_pyarmor import CL as CLPya

# Load test data (1000 klines)
data_path = pathlib.Path(__file__).parent / "test_data" / "BTC_USDT_60m_1000.parquet"
df = pd.read_parquet(data_path)
print(f"Loaded {len(df)} klines")

config = {}
open_cl = CLOpen("BTC/USDT", "60m", config)
open_cl.process_klines(df)

pya_cl = CLPya("BTC/USDT", "60m", config)
pya_cl.process_klines(df)

open_bis = open_cl.get_bis()
pya_bis = pya_cl.get_bis()

open_fxs = open_cl.get_fxs()
pya_fxs = pya_cl.get_fxs()
print(f"Open BIs: {len(open_bis)}, Pyarmor BIs: {len(pya_bis)}")
print(f"Open FXs: {len(open_fxs)}, Pyarmor FXs: {len(pya_fxs)}")

# Show BI details around divergence
def bi_str(b):
    return f"{b.type} h={b.high:.2f} l={b.low:.2f} start={b.start.k.date} end={b.end.k.date}"

def fx_str(fx):
    return f"{fx.type} date={fx.k.date} val={fx.val:.2f} k_h={fx.k.h:.2f} k_l={fx.k.l:.2f}"

print("\n--- Open bi[12]..bi[18] detail ---")
for i in range(12, min(len(open_bis), 19)):
    print(f"  open bi[{i}]: {bi_str(open_bis[i])}")
    
print("\n--- Pyarmor bi[12]..bi[18] detail ---")
for i in range(12, min(len(pya_bis), 19)):
    print(f"  pya  bi[{i}]: {bi_str(pya_bis[i])}")

# bi[14] end should be common - check FXs after that:
if len(pya_bis) > 14 and len(open_bis) > 14:
    bi14_end = open_bis[14].end.k.date
    print(f"\nCommon point: bi[14] end = {bi14_end}")
    print(f"Open bi[14]: {bi_str(open_bis[14])}")
    print(f"Pya  bi[14]: {bi_str(pya_bis[14])}")
    print(f"Open bi[13]: {bi_str(open_bis[13])}")
    print(f"Pya  bi[13]: {bi_str(pya_bis[13])}")

    print(f"\nOpen FXs from bi[14] end:")
    shown = 0
    for j, fx in enumerate(open_fxs):
        if fx.k.date >= bi14_end:
            print(f"  open FX[{j}]: {fx_str(fx)}")
            shown += 1
            if shown >= 12: break

    pya_bi14_end = pya_bis[14].end.k.date
    print(f"\nPyarmor FXs from pya bi[14] end ({pya_bi14_end}):")
    shown = 0
    for j, fx in enumerate(pya_fxs):
        if fx.k.date >= pya_bi14_end:
            print(f"  pya  FX[{j}]: {fx_str(fx)}")
            shown += 1
            if shown >= 12: break
