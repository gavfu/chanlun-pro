"""Trace XD construction step by step to understand divergence."""
import sys; sys.path.insert(0, 'src')
import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_interface import TZXL

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')
cd_o = CL_O('BTC/USDT', '60m')
cd_o.process_klines(df)

bis = cd_o.bis

print("=== TRACING FIRST XD CONSTRUCTION ===")
print(f"Total BIs: {len(bis)}")
print(f"bi[0]: {bis[0].type} {bis[0].start.k.index}→{bis[0].end.k.index}")
print()

# First XD is "down" (same as bi[0].type), starts at bi[0]
xd_type = "down"
start_bi_idx = 0

# For a DOWN XD, we collect UP strokes as characteristic sequence
tzxl_bi_type = "up"
bh_direction = "down"  # 向下包含处理
target_fx_type = "di"  # 寻找底分型

# Collect UP strokes starting from bi[0]
print("=== Characteristic Sequence Elements (UP strokes) ===")
tzxl_bis = []
for i in range(start_bi_idx, len(bis)):
    if bis[i].type == tzxl_bi_type:
        tzxl_bis.append(bis[i])
        print(f"  UP bi[{bis[i].index}] {bis[i].start.k.index}→{bis[i].end.k.index} "
              f"h={bis[i].high:.1f} l={bis[i].low:.1f}")

# Build TZXL with inclusion processing (direction = down)
print(f"\n=== Inclusion Processing (bh_direction={bh_direction}) ===")
print("For DOWN inclusion: max = min([l.high]), min = min([l.low])")
tzxls = []
for bi in tzxl_bis:
    pre_line = bis[bi.index - 1] if bi.index > 0 else bi
    new_tzxl = TZXL(
        bh_direction=bh_direction,
        line=bi,
        pre_line=pre_line,
        line_bad=False,
        done=bi.is_done(),
    )
    
    if len(tzxls) == 0:
        tzxls.append(new_tzxl)
        print(f"  [0] bi[{bi.index}] max={new_tzxl.max:.1f} min={new_tzxl.min:.1f}")
        continue
    
    last_tzxl = tzxls[-1]
    # Check containment
    is_contain = (last_tzxl.max >= new_tzxl.max and last_tzxl.min <= new_tzxl.min) or (
        new_tzxl.max >= last_tzxl.max and new_tzxl.min <= last_tzxl.min
    )
    
    if is_contain:
        print(f"  MERGE bi[{bi.index}] into [{len(tzxls)-1}]: "
              f"old max={last_tzxl.max:.1f} min={last_tzxl.min:.1f}, "
              f"new max={new_tzxl.max:.1f} min={new_tzxl.min:.1f}")
        last_tzxl.lines.append(bi)
        last_tzxl.done = bi.is_done()
        last_tzxl.update_maxmin()
        print(f"    → merged max={last_tzxl.max:.1f} min={last_tzxl.min:.1f}")
    else:
        tzxls.append(new_tzxl)
        print(f"  [{len(tzxls)-1}] bi[{bi.index}] max={new_tzxl.max:.1f} min={new_tzxl.min:.1f}")

print(f"\n=== Final TZXL elements: {len(tzxls)} ===")
for i, xl in enumerate(tzxls):
    lines_str = ",".join(str(l.index) for l in xl.lines)
    print(f"  [{i}] max={xl.max:.1f} min={xl.min:.1f} lines=[bi {lines_str}]")

print(f"\n=== Searching for {target_fx_type} sequence fractal ===")
for i in range(1, len(tzxls) - 1):
    prev_xl = tzxls[i - 1]
    curr_xl = tzxls[i]
    next_xl = tzxls[i + 1]
    
    is_fx = False
    if target_fx_type == "di":
        if curr_xl.min < prev_xl.min and curr_xl.min < next_xl.min:
            is_fx = True
    
    if is_fx:
        # Get end_bi_idx
        end_bi = curr_xl.lines[0]
        for l in curr_xl.lines:
            if l.low < end_bi.low:
                end_bi = l
        end_bi_idx = end_bi.index
        if bis[end_bi_idx].type == "up" and end_bi_idx > 0:
            end_bi_idx -= 1
        
        # Check bi_pohuai
        start_bi = bis[start_bi_idx]
        last_line_idx = curr_xl.lines[-1].index
        has_pohuai = False
        for bi_idx in range(last_line_idx + 1, len(bis)):
            bi = bis[bi_idx]
            if bi.type == "up" and bi.high > start_bi.high:
                has_pohuai = True
            break
        
        bi_count = end_bi_idx - start_bi_idx
        print(f"  FOUND at tzxl[{i}]: prev min={prev_xl.min:.1f}, curr min={curr_xl.min:.1f}, next min={next_xl.min:.1f}")
        print(f"    end_bi={end_bi.index} ({end_bi.type} {end_bi.start.k.index}→{end_bi.end.k.index})")
        print(f"    end_bi_idx={end_bi_idx}, span={bi_count} (need >= 2)")
        print(f"    bi_pohuai={has_pohuai}")
        
        if has_pohuai:
            print(f"    → REJECTED (bi_pohuai)")
        elif bi_count < 2:
            print(f"    → REJECTED (< 3 strokes)")
        else:
            print(f"    → ACCEPTED! XD down bi[{start_bi_idx}]→bi[{end_bi_idx}]")
            break
    else:
        comp = f"curr.min={curr_xl.min:.1f} vs prev.min={prev_xl.min:.1f} vs next.min={next_xl.min:.1f}"
        print(f"  [{i}] NOT fx: {comp}")
