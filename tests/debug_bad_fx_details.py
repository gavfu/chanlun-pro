"""
Compare bad FX details for the 3 divergent cases:
- ETH60 up bi[31]: needs more-extreme (end=41)
- BTC60 down bi[28]: needs always-first (end=38) 
- BTC5m up bi[3]: needs more-extreme (end=9)

Show the TZXLs, which are bad, the first bad FX, and the more-extreme non-bad FX.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_interface import TZXL, BI

config = {
    "bi_type": "bi_type_old", "fx_qj": "fx_qj_k", "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes", "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1", "xd_bzh": "xd_bzh_no",
    "xd_zs_max_lines_split": 11, "xd_allow_split_no_highlow": 1,
    "xd_allow_split_zs_kz": 0, "xd_allow_split_zs_more_line": 1,
    "xd_allow_split_zs_no_direction": 1,
}

def analyze_find_xd_end(cl, bis, start_bi_idx, xd_type, label):
    """Detailed analysis of _find_xd_end logic"""
    tzxl_bi_type = "down" if xd_type == "up" else "up"
    bh_direction = "up" if xd_type == "up" else "down"
    target_fx_type = "ding" if xd_type == "up" else "di"

    tzxl_bis = []
    for i in range(start_bi_idx, len(bis)):
        if bis[i].type == tzxl_bi_type:
            tzxl_bis.append(bis[i])

    # Build TZXLs
    tzxls = []
    for bi in tzxl_bis:
        pre_line = bis[bi.index - 1] if bi.index > 0 else bi
        done = bi.is_done()
        new_tzxl = TZXL(
            bh_direction=bh_direction, line=bi, pre_line=pre_line,
            line_bad=False, done=done,
        )
        if len(tzxls) == 0:
            tzxls.append(new_tzxl)
            continue
        last_tzxl = tzxls[-1]
        old_contains_new = last_tzxl.max >= new_tzxl.max and last_tzxl.min <= new_tzxl.min
        new_contains_old = new_tzxl.max >= last_tzxl.max and new_tzxl.min <= last_tzxl.min
        if old_contains_new:
            last_tzxl.lines.append(bi)
            last_tzxl.done = done
            last_tzxl.line_bad = False
            last_tzxl.update_maxmin()
        elif new_contains_old:
            new_tzxl.line_bad = True
            tzxls.append(new_tzxl)
        else:
            tzxls.append(new_tzxl)

    print(f"\n{'='*80}")
    print(f"  {label}: {xd_type} bi[{start_bi_idx}], target_fx={target_fx_type}")
    print(f"  TZXLs ({len(tzxls)}):")
    for i, xl in enumerate(tzxls):
        bis_str = ",".join([str(l.index) for l in xl.lines])
        print(f"    [{i}] bi[{bis_str}] max={xl.max:.1f} min={xl.min:.1f} bad={xl.line_bad}")

    # Find FX
    print(f"\n  FX search:")
    first_bad = None
    first_bad_extreme = None
    for i in range(1, len(tzxls) - 1):
        curr_xl = tzxls[i]
        prev_xl = tzxls[i - 1]
        next_xl = tzxls[i + 1]

        is_fx = False
        if target_fx_type == "ding":
            if curr_xl.max > prev_xl.max and curr_xl.max > next_xl.max:
                is_fx = True
        else:
            if curr_xl.min < prev_xl.min and curr_xl.min < next_xl.min:
                is_fx = True

        if not is_fx:
            continue

        # Check bi_pohuai
        pohuai = cl._check_xd_bi_pohuai(bis, start_bi_idx, curr_xl, xd_type)
        
        bis_str = ",".join([str(l.index) for l in curr_xl.lines])
        extreme_val = curr_xl.max if target_fx_type == "ding" else curr_xl.min
        
        # Compute end_bi
        if xd_type == "up":
            end_bi = max(curr_xl.lines, key=lambda l: l.high)
        else:
            end_bi = min(curr_xl.lines, key=lambda l: l.low)
        end_idx = end_bi.index
        if end_bi.type == "up":
            end_idx -= 1
        
        status = ""
        if pohuai:
            status = "REJECTED(pohuai)"
        elif curr_xl.line_bad:
            if first_bad is None:
                first_bad = (i, bis_str, extreme_val, end_idx)
                first_bad_extreme = extreme_val
                status = "BAD(first)"
            else:
                status = "BAD(later)"
        else:
            if first_bad is not None:
                more_extreme = (extreme_val > first_bad_extreme) if target_fx_type == "ding" else (extreme_val < first_bad_extreme)
                if more_extreme:
                    status = f"NON-BAD, MORE EXTREME than bad({first_bad_extreme:.1f}) → CURRENT uses THIS end={end_idx}"
                else:
                    status = f"NON-BAD, LESS EXTREME than bad({first_bad_extreme:.1f}) → CURRENT uses BAD end={first_bad[3]}"
            else:
                status = f"NON-BAD(first, no prior bad) → end={end_idx}"
        
        print(f"    TZXL[{i}] bi[{bis_str}] bad={curr_xl.line_bad} extreme={extreme_val:.1f} end={end_idx} pohuai={pohuai} → {status}")
    
    if first_bad:
        print(f"\n  First bad FX: TZXL[{first_bad[0]}] bi[{first_bad[1]}] extreme={first_bad[2]:.1f} end={first_bad[3]}")


# --------- ETH60 up bi[31] ---------
df = pd.read_parquet("tests/test_data/ETH_USDT_60m_1000.parquet")
cl = CL_Open("ETH60", "60m", config)
cl.process_klines(df)
bis = cl.get_bis()
analyze_find_xd_end(cl, bis, 31, "up", "ETH60")

# --------- BTC60 down bi[28] ---------
df = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cl = CL_Open("BTC60", "60m", config)
cl.process_klines(df)
bis = cl.get_bis()
analyze_find_xd_end(cl, bis, 28, "down", "BTC60")

# --------- BTC5m up bi[3] ---------
df = pd.read_parquet("tests/test_data/BTC_USDT_5m_1000.parquet")
cl = CL_Open("BTC5m", "5m", config)
cl.process_klines(df)
bis = cl.get_bis()
analyze_find_xd_end(cl, bis, 3, "up", "BTC5m")
