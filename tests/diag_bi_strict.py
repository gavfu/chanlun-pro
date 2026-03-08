"""
Check strict validity of specific BIs
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import pandas as pd
from chanlun.cl_open import CL as CLOpen
from chanlun.cl_pyarmor import CL as CLPya
from chanlun.cl_interface import Config

df = pd.read_parquet(pathlib.Path(__file__).parent / "test_data" / "BTC_USDT_60m_500.parquet")

c_open = CLOpen("BTC/USDT", "60m", {})
c_open.process_klines(df)

open_fxs = c_open.get_fxs()

# Check bi[0]: FX0->FX1 (ding->di, cl_gap=3, k_gap=4)
# Check bi[27]: FX129->FX130 (di->ding, cl_gap=3, k_gap=5)
# And for 1000klines: FX99->FX100 (di->ding, cl_gap=1, k_gap=5)

def check_bi_strict(start_fx, end_fx, label):
    cl_gap = end_fx.k.index - start_fx.k.index
    k_gap = end_fx.k.k_index - start_fx.k.k_index
    print(f"\n=== {label} ===")
    print(f"start: {start_fx.type} val={start_fx.val:.2f} k_index={start_fx.k.k_index}")
    print(f"end:   {end_fx.type} val={end_fx.val:.2f} k_index={end_fx.k.k_index}")
    print(f"cl_gap={cl_gap} k_gap={k_gap}")
    
    qj = c_open.fx_qj
    qy = c_open.fx_qy
    print(f"qj={qj!r} qy={qy!r} fx_check_k_nums={c_open.fx_check_k_nums}")
    
    if start_fx.type == "ding" and end_fx.type == "di":
        sfl = start_fx.low(qj, qy)
        efl = end_fx.low(qj, qy)
        sfh = start_fx.high(qj, qy)
        efh = end_fx.high(qj, qy)
        print(f"Down stroke strict check:")
        print(f"  start_low={sfl:.2f} end_low={efl:.2f} (start_low < end_low → invalid: {sfl < efl})")
        print(f"  end_high={efh:.2f} start_high={sfh:.2f} (end_high > start_high → invalid: {efh > sfh})")
    elif start_fx.type == "di" and end_fx.type == "ding":
        sfh = start_fx.high(qj, qy)
        efh = end_fx.high(qj, qy)
        sfl = start_fx.low(qj, qy)
        efl = end_fx.low(qj, qy)
        print(f"Up stroke strict check:")
        print(f"  start_high={sfh:.2f} end_high={efh:.2f} (start_high > end_high → invalid: {sfh > efh})")
        print(f"  end_low={efl:.2f} start_low={sfl:.2f} (end_low < start_low → invalid: {efl < sfl})")
    
    result = c_open._bi_fx_valid(start_fx, end_fx)
    print(f"_bi_fx_valid result: {result}")

# bi[0]: FX[0]=ding, FX[1]=di
check_bi_strict(open_fxs[0], open_fxs[1], "bi[0] FX0->FX1 (ding->di, cl_gap=3, k_gap=4)")

# bi[27]: FX[129]=di, FX[130]=ding
check_bi_strict(open_fxs[129], open_fxs[130], "bi[27] FX129->FX130 (di->ding, cl_gap=3, k_gap=5)")

# Compare with longer BIs that pyarmor uses:
# pya bi[0]: FX0->FX5 (ding->di)
check_bi_strict(open_fxs[0], open_fxs[5], "pya bi[0] FX0->FX5 (ding->di)")

# Now load 1000k and check FX99->FX100
df1000 = pd.read_parquet(pathlib.Path(__file__).parent / "test_data" / "BTC_USDT_60m_1000.parquet")
c1000 = CLOpen("BTC/USDT", "60m", {})
c1000.process_klines(df1000)
fxs1000 = c1000.get_fxs()
check_bi_strict(fxs1000[99], fxs1000[100], "1000k FX99->FX100 (di->ding, cl_gap=1, k_gap=5)")
