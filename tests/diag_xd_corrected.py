"""Test the corrected XD algorithm: 
1. No merge when NEW contains OLD (set line_bad=True instead)
2. Only merge when OLD contains NEW
3. Skip sequence fractals where middle element has line_bad=True
4. Different starting point logic
"""
import sys; sys.path.insert(0, 'src')
import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_interface import TZXL

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')
cd_o = CL_O('BTC/USDT', '60m')
cd_o.process_klines(df)
bis = cd_o.bis

def build_xd_corrected(bis, start_bi_idx, xd_type):
    """Build XD with corrected containment & line_bad logic."""
    tzxl_bi_type = "down" if xd_type == "up" else "up"
    bh_direction = "up" if xd_type == "up" else "down"
    target_fx_type = "ding" if xd_type == "up" else "di"
    
    # Collect opposite-direction strokes
    tzxl_bis = []
    for i in range(start_bi_idx, len(bis)):
        if bis[i].type == tzxl_bi_type:
            tzxl_bis.append(bis[i])
    
    if len(tzxl_bis) < 3:
        return None
    
    # Build characteristic sequence with corrected containment
    tzxls = []
    for bi in tzxl_bis:
        pre_line = bis[bi.index - 1] if bi.index > 0 else bi
        new_tzxl = TZXL(
            bh_direction=bh_direction, line=bi, pre_line=pre_line,
            line_bad=False, done=bi.is_done(),
        )
        
        if not tzxls:
            tzxls.append(new_tzxl)
            continue
        
        last = tzxls[-1]
        
        # Check containment using raw bi high/low
        old_contains_new = last.max >= new_tzxl.max and last.min <= new_tzxl.min
        new_contains_old = new_tzxl.max >= last.max and new_tzxl.min <= last.min
        
        if old_contains_new and new_contains_old:
            # Equal ranges - merge
            last.lines.append(bi)
            last.done = bi.is_done()
            last.update_maxmin()
        elif old_contains_new:
            # OLD contains NEW → MERGE (new is absorbed into old)
            last.lines.append(bi)
            last.done = bi.is_done()
            last.update_maxmin()
        elif new_contains_old:
            # NEW contains OLD → DON'T MERGE, mark new as line_bad=True
            new_tzxl.line_bad = True
            tzxls.append(new_tzxl)
        else:
            # No containment → separate element
            tzxls.append(new_tzxl)
    
    if len(tzxls) < 3:
        return None
    
    # Search for sequence fractal, skipping line_bad middle elements
    for i in range(1, len(tzxls) - 1):
        prev = tzxls[i - 1]
        curr = tzxls[i]
        nxt = tzxls[i + 1]
        
        # Skip if middle element is line_bad
        if curr.line_bad:
            continue
        
        is_fx = False
        if target_fx_type == "di":
            is_fx = curr.min < prev.min and curr.min < nxt.min
        else:
            is_fx = curr.max > prev.max and curr.max > nxt.max
        
        if is_fx:
            # Find end bi
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
                return (end_bi_idx, tzxls, i)
    
    return None

# Test first XD: DOWN starting at bi[0]
print("=== Test 1: DOWN XD from bi[0] ===")
result = build_xd_corrected(bis, 0, "down")
if result:
    end_idx, tzxls, fx_idx = result
    print(f"  XD down bi[0]→bi[{end_idx}]")
    print(f"  TZXL sequence ({len(tzxls)} elements):")
    for i, xl in enumerate(tzxls):
        lines = [l.index for l in xl.lines]
        marker = " ← FX" if i == fx_idx else ""
        print(f"    [{i}] max={xl.max:.1f} min={xl.min:.1f} lines=bi{lines} bad={xl.line_bad}{marker}")
else:
    print("  No XD found from bi[0]!")

# Test from bi[2]
print(f"\n=== Test 2: DOWN XD from bi[2] ===")
result = build_xd_corrected(bis, 2, "down")
if result:
    end_idx, tzxls, fx_idx = result
    print(f"  XD down bi[2]→bi[{end_idx}]")
    print(f"  TZXL sequence ({len(tzxls)} elements):")
    for i, xl in enumerate(tzxls):
        lines = [l.index for l in xl.lines]
        marker = " ← FX" if i == fx_idx else ""
        print(f"    [{i}] max={xl.max:.1f} min={xl.min:.1f} lines=bi{lines} bad={xl.line_bad}{marker}")
else:
    print("  No XD found from bi[2]!")

# Also verify: build sequence from bi[0] including bi[1]
print(f"\n=== All TZXL from bi[0] with corrected containment ===")
tzxl_bis = [bi for bi in bis if bi.type == "up"]
bh_direction = "down"
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
    if old_c_new and new_c_old:
        last.lines.append(bi)
        last.done = bi.is_done()
        last.update_maxmin()
    elif old_c_new:
        last.lines.append(bi)
        last.done = bi.is_done()
        last.update_maxmin()
    elif new_c_old:
        new_tzxl.line_bad = True
        tzxls.append(new_tzxl)
    else:
        tzxls.append(new_tzxl)

for i, xl in enumerate(tzxls):
    lines = [l.index for l in xl.lines]
    print(f"  [{i}] max={xl.max:.1f} min={xl.min:.1f} lines=bi{lines} bad={xl.line_bad}")

# Compare with pyarmor
print(f"\n=== Pyarmor XD[0] tzxls for comparison ===")
from chanlun.cl_pyarmor import CL as CL_P
cd_p = CL_P('BTC/USDT', '60m')
cd_p.process_klines(df)
for i, xl in enumerate(cd_p.xds[0].tzxls):
    lines = [l.index for l in xl.lines]
    print(f"  [{i}] max={xl.max:.1f} min={xl.min:.1f} lines=bi{lines} bad={xl.line_bad}")
