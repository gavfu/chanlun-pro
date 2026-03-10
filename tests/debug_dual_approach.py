"""
Test dual bh/no_bh approach in _find_xd_end.

Strategy: 
1. Try "bh" mode (merge both OLD⊃NEW and NEW⊃OLD) → find FIRST valid FX
2. Try "no_bh" mode (current behavior: NEW⊃OLD → separate, line_bad=True) → find FX with "more extreme" rule
3. Choose between them:
   - If bh finds a FX → use it (it's the standard TZXL approach)
   - If bh doesn't find a FX → fall back to no_bh result

This matches pyarmor's approach where bh_di/bh_ding are computed alongside di/ding.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor
from chanlun.cl_interface import TZXL, XLFX, BI
from typing import List, Tuple, Union

# Save the original
original_find_xd_end = CL_Open._find_xd_end

def _build_tzxls_bh(bis, start_bi_idx, xd_type):
    """Build TZXLs with bh mode (merge both OLD⊃NEW and NEW⊃OLD)."""
    tzxl_bi_type = "down" if xd_type == "up" else "up"
    bh_direction = "up" if xd_type == "up" else "down"
    
    tzxl_bis = [bi for bi in bis[start_bi_idx:] if bi.type == tzxl_bi_type]
    if len(tzxl_bis) < 3:
        return None, []
    
    tzxls = []
    for bi in tzxl_bis:
        pre_line = bis[bi.index - 1] if bi.index > 0 else bi
        done = bi.is_done()
        new_tzxl = TZXL(
            bh_direction=bh_direction,
            line=bi,
            pre_line=pre_line,
            line_bad=False,
            done=done,
        )
        
        if len(tzxls) == 0:
            tzxls.append(new_tzxl)
            continue
        
        last_tzxl = tzxls[-1]
        old_contains_new = last_tzxl.max >= new_tzxl.max and last_tzxl.min <= new_tzxl.min
        new_contains_old = new_tzxl.max >= last_tzxl.max and new_tzxl.min <= last_tzxl.min
        
        if old_contains_new or new_contains_old:
            # BH mode: merge BOTH directions
            last_tzxl.lines.append(bi)
            last_tzxl.done = done
            last_tzxl.line_bad = False
            last_tzxl.update_maxmin()
        else:
            tzxls.append(new_tzxl)
    
    return tzxls, tzxl_bis

def patched_find_xd_end(self, bis, start_bi_idx, xd_type):
    """Dual approach: try bh mode first, fall back to no_bh mode."""
    target_fx_type = "ding" if xd_type == "up" else "di"
    
    # === Approach 1: BH mode (merge all containment) ===
    tzxls_bh, _ = _build_tzxls_bh(bis, start_bi_idx, xd_type)
    
    bh_result = None
    if tzxls_bh and len(tzxls_bh) >= 3:
        for i in range(1, len(tzxls_bh) - 1):
            curr_xl = tzxls_bh[i]
            prev_xl = tzxls_bh[i - 1]
            next_xl = tzxls_bh[i + 1]
            
            is_fx = False
            if target_fx_type == "ding":
                if curr_xl.max > prev_xl.max and curr_xl.max > next_xl.max:
                    is_fx = True
            else:
                if curr_xl.min < prev_xl.min and curr_xl.min < next_xl.min:
                    is_fx = True
            
            if is_fx:
                if not self._check_xd_bi_pohuai(bis, start_bi_idx, curr_xl, xd_type):
                    result = self._build_xd_fx_result(
                        bis, start_bi_idx, xd_type, target_fx_type,
                        curr_xl, prev_xl, next_xl, tzxls_bh,
                    )
                    if result is not None:
                        bh_result = result
                        break
    
    if bh_result is not None:
        return bh_result
    
    # === Approach 2: Fall back to no_bh mode (original behavior) ===
    return original_find_xd_end(self, bis, start_bi_idx, xd_type)

CL_Open._find_xd_end = patched_find_xd_end

# Test all cases
test_cases = [
    ("BTCd", "d", "tests/test_data/BTC_USDT_d_500.parquet"),
    ("ETH60", "60m", "tests/test_data/ETH_USDT_60m_1000.parquet"),
    ("BTC60", "60m", "tests/test_data/BTC_USDT_60m_1000.parquet"),
    ("BTC5m", "5m", "tests/test_data/BTC_USDT_5m_1000.parquet"),
]

config = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
    "xd_zs_max_lines_split": 11,
    "xd_allow_split_no_highlow": 1,
    "xd_allow_split_zs_kz": 0,
    "xd_allow_split_zs_more_line": 1,
    "xd_allow_split_zs_no_direction": 1,
}

for name, freq, path in test_cases:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    
    df = pd.read_parquet(path)
    
    cl_open = CL_Open("test", freq, config)
    cl_open.process_klines(df)
    open_xds = cl_open.get_xds()
    
    cl_pyarmor = CL_Pyarmor("test", freq, config)
    cl_pyarmor.process_klines(df)
    pyarmor_xds = cl_pyarmor.get_xds()
    
    print(f"  Open XD count: {len(open_xds)}, Pyarmor XD count: {len(pyarmor_xds)}")
    
    max_len = max(len(open_xds), len(pyarmor_xds))
    content_matches = 0
    for i in range(max_len):
        o = open_xds[i] if i < len(open_xds) else None
        p = pyarmor_xds[i] if i < len(pyarmor_xds) else None
        
        if o and p:
            o_start = o.start_line.index
            o_end = o.end_line.index
            p_start = p.start_line.index
            p_end = p.end_line.index
            
            match = o_start == p_start and o_end == p_end and o.type == p.type
            
            status = "✅" if match else "❌"
            detail = ""
            if not match:
                diffs = []
                if o.type != p.type:
                    diffs.append(f"TYPE({o.type}→{p.type})")
                if o_start != p_start:
                    diffs.append(f"START({o_start}→{p_start})")
                if o_end != p_end:
                    diffs.append(f"END({o_end}→{p_end})")
                detail = " " + " ".join(diffs)
            else:
                content_matches += 1
            
            print(f"  xd[{i}] {status} {o.type} bi[{o_start}→{o_end}] vs {p.type} bi[{p_start}→{p_end}]{detail}")
        elif o:
            print(f"  xd[{i}] ❌ EXTRA in open: {o.type} bi[{o.start_line.index}→{o.end_line.index}]")
        else:
            print(f"  xd[{i}] ❌ MISSING: pyarmor has {p.type} bi[{p.start_line.index}→{p.end_line.index}]")
    
    count_match = "✅" if len(open_xds) == len(pyarmor_xds) else "❌"
    print(f"\n  Count: {count_match} ({len(open_xds)}/{len(pyarmor_xds)}), Content: {content_matches}/{max_len}")
