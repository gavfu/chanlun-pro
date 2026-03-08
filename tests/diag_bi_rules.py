"""
Map out all FX pairs that make up pyarmor's BIs - deduce the validity rule.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import pandas as pd
from chanlun.cl_pyarmor import CL as CLPya

# Test multiple datasets
datasets = [
    ("BTC/USDT", "60m", "BTC_USDT_60m_500.parquet", 500),
    ("BTC/USDT", "60m", "BTC_USDT_60m_1000.parquet", 1000),
]

all_pairs = []  # (cl_gap, k_gap, accepted)

for symbol, freq, fname, limit in datasets:
    df = pd.read_parquet(pathlib.Path(__file__).parent / "test_data" / fname)
    c_pya = CLPya(symbol, freq, {})
    c_pya.process_klines(df)
    
    pya_bis = c_pya.get_bis()
    pya_fxs = c_pya.get_fxs()
    
    # Collect all FX pairs that form BIs (accepted) and their gaps
    accepted_pairs = set()
    for b in pya_bis:
        si = b.start.index
        ei = b.end.index
        cl_gap = b.end.k.index - b.start.k.index
        k_gap = b.end.k.k_index - b.start.k.k_index
        all_pairs.append((cl_gap, k_gap, True, f"{fname} bi[{b.index}] FX{si}->{ei}"))
    
    print(f"\n{fname} BIs (accepted pairs):")
    for b in pya_bis:
        cl_gap = b.end.k.index - b.start.k.index
        k_gap = b.end.k.k_index - b.start.k.k_index
        print(f"  bi[{b.index}]: FX{b.start.index}→{b.end.index} cl_gap={cl_gap} k_gap={k_gap}")
    
    # Now check FX pairs BETWEEN start and end of each BI that were REJECTED
    print(f"\n{fname} Rejected FX pairs within BI:")
    for b in pya_bis:
        si = b.start.index
        ei = b.end.index
        # All intermediate FX pairs: si→si+1, si→si+2, ..., but only if si+1 to si+N-1 are skipped
        for j in range(si+1, ei):
            mid_fx = pya_fxs[j]
            if mid_fx.type != b.start.type:  # opposite type → potential end FX
                cl_gap = mid_fx.k.index - b.start.k.index
                k_gap = mid_fx.k.k_index - b.start.k.k_index
                print(f"  Rejected: FX{si}({b.start.type})→FX{j}({mid_fx.type}) cl_gap={cl_gap} k_gap={k_gap}")
                all_pairs.append((cl_gap, k_gap, False, f"{fname} rejected in bi[{b.index}]"))

# Summary: show all accepted with small cl_gap or k_gap
print("\n\n=== Summary: Pairs with cl_gap < 5 or k_gap < 5 ===")
for cl_gap, k_gap, accepted, label in sorted(all_pairs, key=lambda x: (x[0], x[1])):
    if cl_gap < 5 or k_gap < 5:
        status = "ACCEPTED" if accepted else "rejected"
        print(f"  cl_gap={cl_gap} k_gap={k_gap} → {status}  [{label}]")
