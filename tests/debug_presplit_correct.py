"""
Properly reconstruct pyarmor's pre-split segments by merging consecutive split XDs.
Then compare _find_xd_end results against pre-split expected ends.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor
from chanlun.cl_interface import TZXL

config = {
    "bi_type": "bi_type_old", "fx_qj": "fx_qj_k", "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes", "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1", "xd_bzh": "xd_bzh_no",
    "xd_zs_max_lines_split": 11, "xd_allow_split_no_highlow": 1,
    "xd_allow_split_zs_kz": 0, "xd_allow_split_zs_more_line": 1,
    "xd_allow_split_zs_no_direction": 1,
}

def reconstruct_presplit(xds):
    """Merge consecutive splits back into pre-split segments"""
    presplit = []
    i = 0
    while i < len(xds):
        xd = xds[i]
        if not xd.is_split:
            presplit.append({
                'type': xd.type,
                'start': xd.start_line.index,
                'end': xd.end_line.index,
                'split': False,
            })
            i += 1
        else:
            # Find all consecutive splits
            start = xd.start_line.index
            xd_type = xd.type
            j = i
            while j < len(xds) and xds[j].is_split:
                j += 1
            end = xds[j-1].end_line.index
            
            # The pre-split segment covers from first split's start to last split's end
            # But we need to determine the CORRECT type.
            # If the prev non-split segment exists, the pre-split type alternates.
            # Actually, the split replaces a single segment, so its type should be
            # determined by position in the alternating sequence.
            if presplit:
                prev_type = presplit[-1]['type']
                xd_type = "down" if prev_type == "up" else "up"
            
            presplit.append({
                'type': xd_type,
                'start': start,
                'end': end,
                'split': True,
                'split_parts': [(xds[k].type, xds[k].start_line.index, xds[k].end_line.index, xds[k].is_split) for k in range(i, j)],
            })
            i = j
    return presplit


cases = [
    ("BTCd", "BTC_USDT_d_500.parquet", "d"),
    ("ETH60", "ETH_USDT_60m_1000.parquet", "60m"),
    ("BTC60", "BTC_USDT_60m_1000.parquet", "60m"),
    ("BTC5m", "BTC_USDT_5m_1000.parquet", "5m"),
    ("ETH5m", "ETH_USDT_5m_1000.parquet", "5m"),
]

for name, file, freq in cases:
    df = pd.read_parquet(f"tests/test_data/{file}")
    
    cl_p = CL_Pyarmor(name, freq, config)
    cl_p.process_klines(df)
    xds_p = cl_p.get_xds()
    
    cl_o = CL_Open(name, freq, config)
    cl_o.process_klines(df)
    
    presplit = reconstruct_presplit(xds_p)
    
    print(f"\n{'='*80}")
    print(f"  {name} — Pyarmor pre-split segments:")
    for i, ps in enumerate(presplit):
        split_str = ""
        if ps['split']:
            parts = " → ".join([f"{t} [{s}→{e}]({sp})" for t, s, e, sp in ps['split_parts']])
            split_str = f"  SPLIT: {parts}"
        print(f"  [{i}] {ps['type']} bi[{ps['start']}→{ps['end']}]{split_str}")
    
    # Now Compare _find_xd_end for each pre-split start
    bis = cl_o.get_bis()
    print(f"\n  _find_xd_end comparison:")
    for ps in presplit:
        start_bi = ps['start']
        xd_type = ps['type']
        expected_end = ps['end']
        
        result = cl_o._find_xd_end(bis, start_bi, xd_type)
        our_end = result[0] if result else None
        
        match = "✅" if our_end == expected_end else "❌"
        print(f"    {xd_type:4s} bi[{start_bi:2d}] expected={expected_end:3d}  ours={str(our_end):>4s} {match}")
