"""
Check: across ALL datasets, what TZXL positions do bad FXes appear at?
This validates whether a position-based is_line_bad rule (pos >= 3 → not bad) is safe.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_interface import TZXL, BI
from typing import List

config = {
    "bi_type": "bi_type_old", "fx_qj": "fx_qj_k", "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes", "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1", "xd_bzh": "xd_bzh_no",
    "xd_zs_max_lines_split": 11, "xd_allow_split_no_highlow": 1,
    "xd_allow_split_zs_kz": 0, "xd_allow_split_zs_more_line": 1,
    "xd_allow_split_zs_no_direction": 1,
}

datasets = [
    ("BTCd", "tests/test_data/BTC_USDT_d_500.parquet"),
    ("ETH60", "tests/test_data/ETH_USDT_60m_1000.parquet"),
    ("BTC60", "tests/test_data/BTC_USDT_60m_1000.parquet"),
    ("BTC5m", "tests/test_data/BTC_USDT_5m_1000.parquet"),
    ("ETH5m", "tests/test_data/ETH_USDT_5m_1000.parquet"),
]

for ds_name, ds_path in datasets:
    df = pd.read_parquet(ds_path)
    cl_o = CL_Open("test", "60m", config)
    cl_o.process_klines(df)
    bis_o = cl_o.get_bis()
    
    print(f"\n{'='*60}")
    print(f"{ds_name}: {len(bis_o)} BIs")
    print(f"{'='*60}")
    
    # Trace all _find_xd_end calls and the TZXL positions of bad FXes
    xd_type, start_bi_idx = cl_o._find_first_xd_start(bis_o)
    xd_idx = 0
    
    while start_bi_idx < len(bis_o):
        tzxl_bi_type = "down" if xd_type == "up" else "up"
        bh_direction = "up" if xd_type == "up" else "down"
        target_fx_type = "ding" if xd_type == "up" else "di"
        
        # Build TZXLs
        tzxl_bis = [bi for bi in bis_o[start_bi_idx:] if bi.type == tzxl_bi_type]
        if len(tzxl_bis) < 3:
            break
            
        tzxls: List[TZXL] = []
        for bi in tzxl_bis:
            pre_line = bis_o[bi.index - 1] if bi.index > 0 else bi
            done = bi.is_done()
            new_tzxl = TZXL(bh_direction=bh_direction, line=bi, pre_line=pre_line, line_bad=False, done=done)
            if len(tzxls) == 0:
                tzxls.append(new_tzxl)
                continue
            last_tzxl = tzxls[-1]
            if last_tzxl.max >= new_tzxl.max and last_tzxl.min <= new_tzxl.min:
                last_tzxl.lines.append(bi)
                last_tzxl.done = done
                last_tzxl.line_bad = False
                last_tzxl.update_maxmin()
            elif new_tzxl.max >= last_tzxl.max and new_tzxl.min <= last_tzxl.min:
                new_tzxl.line_bad = True
                tzxls.append(new_tzxl)
            else:
                tzxls.append(new_tzxl)
        
        # Find first bad FX
        found_bad = False
        for i in range(1, len(tzxls) - 1):
            curr = tzxls[i]
            prev = tzxls[i - 1]
            nxt = tzxls[i + 1]
            
            is_fx = False
            if target_fx_type == "ding":
                is_fx = curr.max > prev.max and curr.max > nxt.max
            else:
                is_fx = curr.min < prev.min and curr.min < nxt.min
            
            if is_fx and curr.line_bad:
                bi_indices = [l.index for l in curr.lines]
                
                # Find if there's a later non-bad FX that's more extreme
                later_non_bad = None
                for j in range(i + 1, len(tzxls) - 1):
                    c2 = tzxls[j]
                    p2 = tzxls[j - 1]
                    n2 = tzxls[j + 1]
                    is_fx2 = False
                    if target_fx_type == "ding":
                        is_fx2 = c2.max > p2.max and c2.max > n2.max
                    else:
                        is_fx2 = c2.min < p2.min and c2.min < n2.min
                    if is_fx2 and not c2.line_bad:
                        # Check if more extreme
                        more_extreme = (
                            (target_fx_type == "ding" and c2.max > curr.max) or
                            (target_fx_type == "di" and c2.min < curr.min)
                        )
                        if more_extreme:
                            later_non_bad = (j, [l.index for l in c2.lines])
                        break
                
                impact = ""
                if later_non_bad:
                    impact = f"→ WOULD SKIP, use TZXL[{later_non_bad[0]}] bi{later_non_bad[1]}"
                else:
                    impact = "→ no better non-bad FX, would use this one"
                
                print(f"  xd[{xd_idx}] {xd_type}[{start_bi_idx}→]: "
                      f"BAD FX at TZXL[{i}] bi{bi_indices} {target_fx_type} "
                      f"{impact}")
                found_bad = True
                break
        
        # Use the actual _find_xd_end to get the result
        result = cl_o._find_xd_end(bis_o, start_bi_idx, xd_type)
        if result is None:
            break
        end_bi_idx = result[0]
        start_bi_idx = end_bi_idx + 1
        xd_type = "down" if xd_type == "up" else "up"
        xd_idx += 1
