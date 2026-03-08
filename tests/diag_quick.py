"""Quick test of _bi_fx_valid for key FX pairs"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
import pandas as pd
from chanlun.cl_open import CL as CLOpen

df = pd.read_parquet(pathlib.Path(__file__).parent / "test_data" / "BTC_USDT_60m_500.parquet")
c = CLOpen("BTC/USDT", "60m", {})
c.process_klines(df)

fxs = c.fxs

def check(a, b, note=""):
    cl_gap = fxs[b].k.index - fxs[a].k.index
    k_gap = fxs[b].k.k_index - fxs[a].k.k_index
    valid = c._bi_fx_valid(fxs[a], fxs[b])
    print(f"_bi_fx_valid(FX{a},{b}) cl_gap={cl_gap} k_gap={k_gap} => {valid}  {note}")

# Divergence #1: why does FX22->FX23 confirm bi FX19->FX22?
print("Divergence #1 (bi[5]):")
check(19, 22, "open's bi[5] end")
check(22, 23, "confirm check: if True, bi[5]=FX19->FX22 gets confirmed")
check(19, 26, "pyarmor's bi[5] end")

print("\nDivergence #2 (bi[12]):")
check(54, 57, "open's bi[14] end")
check(57, 58, "confirm check after FX57")
check(54, 69, "pyarmor's bi[12] end")

print("\nDivergence #3 (bi[21]):")
check(129, 130, "open's bi[27] end")
check(130, 131, "confirm check after FX130")
check(129, 136, "pyarmor's bi[21] end")
