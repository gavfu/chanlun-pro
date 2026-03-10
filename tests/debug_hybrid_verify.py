"""
Verify: for ALL cases where no_bh first FX is bad, does bh first give the correct answer?
And for cases where no_bh first is non-bad, does the current approach already give correct?

Test ALL pre-split segment starts.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor
from chanlun.cl_interface import TZXL, BI

config = {
    "bi_type": "bi_type_old", "fx_qj": "fx_qj_k", "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes", "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1", "xd_bzh": "xd_bzh_no",
    "xd_zs_max_lines_split": 11, "xd_allow_split_no_highlow": 1,
    "xd_allow_split_zs_kz": 0, "xd_allow_split_zs_more_line": 1,
    "xd_allow_split_zs_no_direction": 1,
}

def build_tzxls(bis, start_bi_idx, xd_type, bh_mode):
    tzxl_bi_type = "down" if xd_type == "up" else "up"
    bh_direction = "up" if xd_type == "up" else "down"

    tzxl_bis = []
    for i in range(start_bi_idx, len(bis)):
        if bis[i].type == tzxl_bi_type:
            tzxl_bis.append(bis[i])

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
        
        if bh_mode:
            if old_contains_new or new_contains_old:
                last_tzxl.lines.append(bi)
                last_tzxl.done = done
                last_tzxl.line_bad = False
                last_tzxl.update_maxmin()
            else:
                tzxls.append(new_tzxl)
        else:
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
    return tzxls

def find_first_fx_end(cl, bis, tzxls, start_bi_idx, xd_type, target_fx_type):
    """Find first FX end index"""
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

        if is_fx:
            if not cl._check_xd_bi_pohuai(bis, start_bi_idx, curr_xl, xd_type):
                if xd_type == "up":
                    end_bi = max(curr_xl.lines, key=lambda l: l.high)
                    end_bi_idx = end_bi.index
                    if bis[end_bi_idx].type == "down" and end_bi_idx > 0:
                        end_bi_idx -= 1
                else:
                    end_bi = min(curr_xl.lines, key=lambda l: l.low)
                    end_bi_idx = end_bi.index
                    if bis[end_bi_idx].type == "up" and end_bi_idx > 0:
                        end_bi_idx -= 1
                
                if end_bi_idx - start_bi_idx >= 2:
                    return end_bi_idx, curr_xl.line_bad
    return None, None


cases = [
    ("BTCd", "BTC_USDT_d_500.parquet", "d"),
    ("ETH60", "ETH_USDT_60m_1000.parquet", "60m"),
    ("BTC60", "BTC_USDT_60m_1000.parquet", "60m"),
    ("BTC5m", "BTC_USDT_5m_1000.parquet", "5m"),
    ("ETH5m", "ETH_USDT_5m_1000.parquet", "5m"),
]

for name, file, freq in cases:
    df = pd.read_parquet(f"tests/test_data/{file}")
    
    # Get pyarmor pre-split segments
    cl_p = CL_Pyarmor(name, freq, config)
    cl_p.process_klines(df)
    
    # Get our segments (use current code, just for BI data)
    cl_o = CL_Open(name, freq, config)
    cl_o.process_klines(df)
    bis = cl_o.get_bis()
    
    # Get pyarmor pre-split: use xds but ignore splits
    xds_p = cl_p.get_xds()
    # Get pre-split by merging split segments
    presplit_p = []
    i = 0
    while i < len(xds_p):
        xd = xds_p[i]
        if not xd.is_split:
            presplit_p.append((xd.type, xd.start_line.index, xd.end_line.index))
            i += 1
        else:
            # Find all consecutive splits
            start = xd.start_line.index
            j = i
            while j < len(xds_p) and xds_p[j].is_split:
                j += 1
            end = xds_p[j-1].end_line.index
            xd_type = xds_p[i].type if xds_p[i].start_line.index == start else xds_p[i-1].type
            # Actually for pre-split, the merged segment type is the type that covers start→end
            # Let's use xds_p[i].type for the first split
            presplit_p.append((xds_p[i].type, start, end))
            i = j
    
    # Also get current pre-split (our code)
    # We can hook _split_xds but that's complex. Let's just test each pyarmor pre-split start.
    
    print(f"\n{'='*80}")
    print(f"  {name}")
    
    for xd_type, start_bi, expected_end in presplit_p:
        target_fx_type = "ding" if xd_type == "up" else "di"
        
        tzxls_nobh = build_tzxls(bis, start_bi, xd_type, bh_mode=False)
        tzxls_bh = build_tzxls(bis, start_bi, xd_type, bh_mode=True)
        
        nobh_end, nobh_bad = find_first_fx_end(cl_o, bis, tzxls_nobh, start_bi, xd_type, target_fx_type)
        bh_end, bh_bad = find_first_fx_end(cl_o, bis, tzxls_bh, start_bi, xd_type, target_fx_type)
        
        # Current code's result (with more-extreme rule)
        curr_result = cl_o._find_xd_end(bis, start_bi, xd_type)
        curr_end = curr_result[0] if curr_result else None
        
        # Proposed hybrid: if no_bh first is bad, use bh first
        hybrid_end = nobh_end
        if nobh_bad:
            hybrid_end = bh_end
        
        match_curr = "✅" if curr_end == expected_end else "❌"
        match_hybrid = "✅" if hybrid_end == expected_end else "❌"
        match_nobh = "✅" if nobh_end == expected_end else "❌"
        match_bh = "✅" if bh_end == expected_end else "❌"
        
        bad_str = " BAD" if nobh_bad else ""
        
        expected_s = str(expected_end)
        if curr_end != expected_end or hybrid_end != expected_end:
            print(f"  {xd_type:4s} bi[{start_bi:2d}] expected={expected_s:>3s}  curr={str(curr_end):>4s}{match_curr}  nobh_first={str(nobh_end):>4s}({match_nobh}{bad_str})  bh_first={str(bh_end):>4s}({match_bh})  HYBRID={str(hybrid_end):>4s}{match_hybrid}")
        else:
            print(f"  {xd_type:4s} bi[{start_bi:2d}] expected={expected_s:>3s}  curr={str(curr_end):>4s}{match_curr}  HYBRID={str(hybrid_end):>4s}{match_hybrid}")
