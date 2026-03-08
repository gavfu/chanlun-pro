"""Test XD: no merge when NEW contains OLD, no line_bad, normal fractal search."""
import sys; sys.path.insert(0, 'src')
import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_interface import TZXL

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')
cd_o = CL_O('BTC/USDT', '60m')
cd_o.process_klines(df)
bis = cd_o.bis

def build_tzxl_no_merge_new_contains_old(bis, start_bi_idx, xd_type):
    """Build characteristic sequence without merging when new contains old."""
    tzxl_bi_type = "down" if xd_type == "up" else "up"
    bh_direction = "up" if xd_type == "up" else "down"
    
    tzxl_bis = [bi for bi in bis[start_bi_idx:] if bi.type == tzxl_bi_type]
    if len(tzxl_bis) < 3:
        return []
    
    tzxls = []
    for bi in tzxl_bis:
        pre_line = bis[bi.index - 1] if bi.index > 0 else bi
        new_tzxl = TZXL(bh_direction=bh_direction, line=bi, pre_line=pre_line,
                        line_bad=False, done=bi.is_done())
        if not tzxls:
            tzxls.append(new_tzxl)
            continue
        
        last = tzxls[-1]
        old_c_new = last.max >= new_tzxl.max and last.min <= new_tzxl.min
        new_c_old = new_tzxl.max >= last.max and new_tzxl.min <= last.min
        
        if old_c_new:
            # OLD contains NEW → MERGE
            last.lines.append(bi)
            last.done = bi.is_done()
            last.update_maxmin()
        elif new_c_old:
            # NEW contains OLD → DON'T MERGE, keep separate
            # Set line_bad=True on the new element (the "包含" one)
            new_tzxl.line_bad = True
            tzxls.append(new_tzxl)
        else:
            # No containment
            tzxls.append(new_tzxl)
    
    return tzxls

def find_xd_end_corrected(bis, start_bi_idx, xd_type, tzxls):
    """Find XD end with corrected logic."""
    target_fx_type = "ding" if xd_type == "up" else "di"
    
    if len(tzxls) < 3:
        return None
    
    for i in range(1, len(tzxls) - 1):
        prev = tzxls[i - 1]
        curr = tzxls[i]
        nxt = tzxls[i + 1]
        
        # Skip if middle element has line_bad
        if curr.line_bad:
            continue
        
        is_fx = False
        if target_fx_type == "di":
            is_fx = curr.min < prev.min and curr.min < nxt.min
        else:
            is_fx = curr.max > prev.max and curr.max > nxt.max
        
        if is_fx:
            if xd_type == "up":
                end_bi = max(curr.lines, key=lambda l: l.high)
                end_bi_idx = end_bi.index
                if bis[end_bi_idx].type == "down" and end_bi_idx > 0:
                    end_bi_idx -= 1
            else:
                end_bi = min(curr.lines, key=lambda l: l.low)
                end_bi_idx = end_bi.index
                if bis[end_bi_idx].type == "up" and end_bi_idx > 0:
                    end_bi_idx -= 1
            
            if end_bi_idx - start_bi_idx >= 2:
                return end_bi_idx, i
    
    return None

# ===== Now test how pyarmor determines starting point =====
# Pyarmor starts at bi[2] for the first XD. 
# The rule seems to be: find the highest high / lowest low among first few strokes
# bi[0]: down, starts high at 69230.0 (ding@1)
# bi[1]: up, goes to 70110.9 (ding@20)  
# bi[1] goes HIGHER than bi[0] start → the first XD should start from ding@20 (bi[2]'s start)

# Hypothesis: first XD direction = first bi direction 
# But if the second bi exceeds the first bi's start, 
# start from the second bi instead (as a bi of opposite direction creates the real peak/trough)
# Then: start_bi_idx = index of the first bi of the determined direction AFTER the extremum

print("=== Determining first XD start ===")
bi0 = bis[0]
bi1 = bis[1]
print(f"bi[0]: {bi0.type} h={bi0.high:.1f} start={bi0.start.val:.1f}")
print(f"bi[1]: {bi1.type} h={bi1.high:.1f} start={bi1.start.val:.1f}")

# For DOWN first XD: the start should be the highest point
# bi[0] start = ding@1 val=69230.0
# bi[1] end = ding@20 val=70110.9  (higher!)
# So real start = bi[2] (the DOWN stroke from ding@20)

# Try: start at bi[2]
start_bi_idx = 2
xd_type = "down"

tzxls = build_tzxl_no_merge_new_contains_old(bis, start_bi_idx, xd_type)
print(f"\n=== TZXL from bi[{start_bi_idx}] ({len(tzxls)} elements) ===")
for i, xl in enumerate(tzxls):
    lines = [l.index for l in xl.lines]
    print(f"  [{i}] max={xl.max:.1f} min={xl.min:.1f} lines=bi{lines} bad={xl.line_bad}")

result = find_xd_end_corrected(bis, start_bi_idx, xd_type, tzxls)
if result:
    end_idx, fx_idx = result
    print(f"\n  Found XD down bi[{start_bi_idx}]→bi[{end_idx}] (fractal at tzxl[{fx_idx}])")
else:
    print(f"\n  No XD found!")

# Now also include bi[1] in the sequence (like pyarmor does)
# Pyarmor's tzxls[0] is bi[1], even though XD starts at bi[2]
# This may be because bi[1] is the pre_line of the first char-seq element (bi[3])
# Let's try including bi[1] in the char seq
print(f"\n=== TZXL from bi[0] (include bi[1]) ===")
tzxls_all = build_tzxl_no_merge_new_contains_old(bis, 0, "down")
for i, xl in enumerate(tzxls_all):
    lines = [l.index for l in xl.lines]
    print(f"  [{i}] max={xl.max:.1f} min={xl.min:.1f} lines=bi{lines} bad={xl.line_bad}")

result = find_xd_end_corrected(bis, 2, "down", tzxls_all)
if result:
    end_idx, fx_idx = result
    print(f"\n  Found XD down bi[2]→bi[{end_idx}] (fractal at tzxl[{fx_idx}])")
else:
    print(f"\n  No XD found with all tzxls!")

# For the SECOND XD: UP from bi[15]
print(f"\n{'='*60}")
print("=== Second XD: UP from bi[15] ===")
start_bi_idx = 15
xd_type = "up"
tzxls_2 = build_tzxl_no_merge_new_contains_old(bis, start_bi_idx, xd_type)
print(f"TZXL ({len(tzxls_2)} elements):")
for i, xl in enumerate(tzxls_2):
    lines = [l.index for l in xl.lines]
    print(f"  [{i}] max={xl.max:.1f} min={xl.min:.1f} lines=bi{lines} bad={xl.line_bad}")

result = find_xd_end_corrected(bis, start_bi_idx, xd_type, tzxls_2)
if result:
    end_idx, fx_idx = result
    print(f"\n  Found XD up bi[{start_bi_idx}]→bi[{end_idx}] (fractal at tzxl[{fx_idx}])")
else:
    print(f"\n  No XD found!")

print(f"\n=== Pyarmor XD[1] tzxls for comparison ===")
from chanlun.cl_pyarmor import CL as CL_P
cd_p = CL_P('BTC/USDT', '60m')
cd_p.process_klines(df)
for i, xl in enumerate(cd_p.xds[1].tzxls):
    lines = [l.index for l in xl.lines]
    print(f"  [{i}] max={xl.max:.1f} min={xl.min:.1f} lines=bi{lines} bad={xl.line_bad}")
