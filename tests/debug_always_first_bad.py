"""Simulate 'always first bad' (when bad exists, always use it) for ETH60"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_interface import TZXL, XLFX

CL_CONFIG = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
}

original_find_xd_end = CL_O._find_xd_end

def always_first_bad_find_xd_end(self, bis, start_bi_idx, xd_type):
    """If first bad FX exists, always use it (no 'more extreme' comparison)"""
    tzxl_bi_type = "down" if xd_type == "up" else "up"
    bh_direction = "up" if xd_type == "up" else "down"
    target_fx_type = "ding" if xd_type == "up" else "di"
    
    tzxl_bis = [bi for bi in bis[start_bi_idx:] if bi.type == tzxl_bi_type]
    if len(tzxl_bis) < 3:
        return None
    
    tzxls = []
    for bi in tzxl_bis:
        pre_line = bis[bi.index - 1] if bi.index > 0 else bi
        new_tzxl = TZXL(bh_direction=bh_direction, line=bi, pre_line=pre_line,
                       line_bad=False, done=bi.is_done())
        if not tzxls:
            tzxls.append(new_tzxl)
            continue
        last = tzxls[-1]
        old_new = last.max >= new_tzxl.max and last.min <= new_tzxl.min
        new_old = new_tzxl.max >= last.max and new_tzxl.min <= last.min
        if old_new:
            last.lines.append(bi)
            last.done = bi.is_done()
            last.line_bad = False
            last.update_maxmin()
        elif new_old:
            new_tzxl.line_bad = True
            tzxls.append(new_tzxl)
        else:
            tzxls.append(new_tzxl)
    
    if len(tzxls) < 3:
        return None
    
    first_bad_result = None
    
    for i in range(1, len(tzxls) - 1):
        curr_xl = tzxls[i]
        prev_xl = tzxls[i - 1]
        next_xl = tzxls[i + 1]
        
        if target_fx_type == "ding":
            is_fx = curr_xl.max > prev_xl.max and curr_xl.max > next_xl.max
        else:
            is_fx = curr_xl.min < prev_xl.min and curr_xl.min < next_xl.min
        
        if is_fx:
            if not self._check_xd_bi_pohuai(bis, start_bi_idx, curr_xl, xd_type):
                result = self._build_xd_fx_result(
                    bis, start_bi_idx, xd_type, target_fx_type,
                    curr_xl, prev_xl, next_xl, tzxls
                )
                if result is not None:
                    if curr_xl.line_bad:
                        # Always use first bad
                        return result
                    
                    # Non-bad: if we have a stored bad, always use the bad
                    if first_bad_result is not None:
                        return first_bad_result
                    
                    # No prior bad, use non-bad
                    return result
                    
                    # Hmm wait, this logic is wrong - we need to store first_bad
                    # Let me fix: if bad, store and continue looking. When we find a non-bad, return the bad.
    
    if first_bad_result is not None:
        return first_bad_result
    
    return None

# Fix the logic - actually "always first bad when bad exists"
def always_first_bad_v2(self, bis, start_bi_idx, xd_type):
    """When a bad FX is found, use it immediately (return first bad)"""
    tzxl_bi_type = "down" if xd_type == "up" else "up"
    bh_direction = "up" if xd_type == "up" else "down"
    target_fx_type = "ding" if xd_type == "up" else "di"
    
    tzxl_bis = [bi for bi in bis[start_bi_idx:] if bi.type == tzxl_bi_type]
    if len(tzxl_bis) < 3:
        return None
    
    tzxls = []
    for bi in tzxl_bis:
        pre_line = bis[bi.index - 1] if bi.index > 0 else bi
        new_tzxl = TZXL(bh_direction=bh_direction, line=bi, pre_line=pre_line,
                       line_bad=False, done=bi.is_done())
        if not tzxls:
            tzxls.append(new_tzxl)
            continue
        last = tzxls[-1]
        old_new = last.max >= new_tzxl.max and last.min <= new_tzxl.min
        new_old = new_tzxl.max >= last.max and new_tzxl.min <= last.min
        if old_new:
            last.lines.append(bi)
            last.done = bi.is_done()
            last.line_bad = False
            last.update_maxmin()
        elif new_old:
            new_tzxl.line_bad = True
            tzxls.append(new_tzxl)
        else:
            tzxls.append(new_tzxl)
    
    if len(tzxls) < 3:
        return None
    
    for i in range(1, len(tzxls) - 1):
        curr_xl = tzxls[i]
        prev_xl = tzxls[i - 1]
        next_xl = tzxls[i + 1]
        
        if target_fx_type == "ding":
            is_fx = curr_xl.max > prev_xl.max and curr_xl.max > next_xl.max
        else:
            is_fx = curr_xl.min < prev_xl.min and curr_xl.min < next_xl.min
        
        if is_fx:
            if not self._check_xd_bi_pohuai(bis, start_bi_idx, curr_xl, xd_type):
                result = self._build_xd_fx_result(
                    bis, start_bi_idx, xd_type, target_fx_type,
                    curr_xl, prev_xl, next_xl, tzxls
                )
                if result is not None:
                    return result  # Return FIRST valid FX (same as "always first")
    
    return None

# Test: what if we just use "always first valid FX" but then the ETH60 difference  
# comes from splitting differently?

from chanlun.cl import CL as CL_P

# Let me determine what pyarmor produces for `up from bi[15]` (ETH60)
# We know both produce xd[1] = up bi[15→19]. So _find_xd_end(up, 15) → end=19 in both.
# What about _find_xd_end(up, 25) → ?

# Actually let me just trace the _find_xd_end calls for "always first FX" approach
CL_O._find_xd_end = always_first_bad_v2  # same as "always first"

trace_calls = []
orig_v2 = always_first_bad_v2

def trace_v2(self, bis, start_bi_idx, xd_type):
    result = orig_v2(self, bis, start_bi_idx, xd_type)
    end = result[0] if result else None
    trace_calls.append((start_bi_idx, xd_type, end))
    return result

CL_O._find_xd_end = trace_v2

df = pd.read_parquet("tests/test_data/ETH_USDT_60m_1000.parquet")
cd = CL_O("test", "test", config=CL_CONFIG)
cd.process_klines(df)

print("=== ETH60 with 'always first FX' ===")
print("\n_find_xd_end calls:")
for start, tp, end in trace_calls:
    print(f"  {tp:>4s} from bi[{start}] → end={end}")

print("\nSegments:")
for i, xd in enumerate(cd.get_xds()):
    print(f"  xd[{i}] {xd.type:>4s} bi[{xd.start_line.index}->{xd.end_line.index}] split=[{xd.is_split}]")

CL_O._find_xd_end = original_find_xd_end
