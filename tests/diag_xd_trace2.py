"""Trace XD construction starting at bi[2] like pyarmor does."""
import sys; sys.path.insert(0, 'src')
import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_interface import TZXL

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')
cd_o = CL_O('BTC/USDT', '60m')
cd_o.process_klines(df)
bis = cd_o.bis

# ===== Simulate starting at bi[2] (like pyarmor) =====
xd_type = "down"
start_bi_idx = 2  # KEY DIFFERENCE: start at bi[2]

tzxl_bi_type = "up"  # For DOWN XD, collect UP strokes
bh_direction = "down"
target_fx_type = "di"

print(f"=== Building DOWN XD starting at bi[{start_bi_idx}] ===")
print(f"bi[{start_bi_idx}]: {bis[start_bi_idx].type} {bis[start_bi_idx].start.k.index}→{bis[start_bi_idx].end.k.index}")
print()

# Collect UP strokes from bi[2] onwards
print("=== UP strokes (characteristic sequence elements) ===")
tzxl_bis = []
for i in range(start_bi_idx, len(bis)):
    if bis[i].type == tzxl_bi_type:
        tzxl_bis.append(bis[i])
        print(f"  bi[{bis[i].index}] {bis[i].start.k.index}→{bis[i].end.k.index} "
              f"h={bis[i].high:.1f} l={bis[i].low:.1f}")

# Build TZXL with inclusion (direction = down)
print(f"\n=== Inclusion Processing (bh_direction={bh_direction}) ===")
tzxls = []
for bi in tzxl_bis:
    pre_line = bis[bi.index - 1] if bi.index > 0 else bi
    new_tzxl = TZXL(
        bh_direction=bh_direction, line=bi, pre_line=pre_line,
        line_bad=False, done=bi.is_done(),
    )
    
    if len(tzxls) == 0:
        tzxls.append(new_tzxl)
        print(f"  [{len(tzxls)-1}] bi[{bi.index}] max={new_tzxl.max:.1f} min={new_tzxl.min:.1f}")
        continue
    
    last = tzxls[-1]
    is_contain = (last.max >= new_tzxl.max and last.min <= new_tzxl.min) or (
        new_tzxl.max >= last.max and new_tzxl.min <= last.min
    )
    
    if is_contain:
        print(f"  MERGE bi[{bi.index}] into [{len(tzxls)-1}]: "
              f"[{last.max:.1f},{last.min:.1f}] + [{new_tzxl.max:.1f},{new_tzxl.min:.1f}]")
        last.lines.append(bi)
        last.done = bi.is_done()
        last.update_maxmin()
        print(f"    → [{last.max:.1f},{last.min:.1f}] lines={[l.index for l in last.lines]}")
    else:
        tzxls.append(new_tzxl)
        print(f"  [{len(tzxls)-1}] bi[{bi.index}] max={new_tzxl.max:.1f} min={new_tzxl.min:.1f}")

print(f"\n=== Final TZXL: {len(tzxls)} elements ===")
for i, xl in enumerate(tzxls):
    lines_str = [l.index for l in xl.lines]
    print(f"  [{i}] max={xl.max:.1f} min={xl.min:.1f} lines=bi{lines_str}")

print(f"\n=== Searching for '{target_fx_type}' sequence fractal ===")
for i in range(1, len(tzxls) - 1):
    prev = tzxls[i - 1]
    curr = tzxls[i]
    nxt = tzxls[i + 1]
    
    is_fx = curr.min < prev.min and curr.min < nxt.min  # di fractal
    
    if is_fx:
        # Get end_bi
        end_bi = min(curr.lines, key=lambda l: l.low)
        end_bi_idx = end_bi.index
        if bis[end_bi_idx].type == "up" and end_bi_idx > 0:
            end_bi_idx -= 1
        
        # Check bi_pohuai: does any UP stroke after curr go above start?
        start_bi = bis[start_bi_idx]
        last_line_idx = curr.lines[-1].index
        has_pohuai = False
        for bi_idx in range(last_line_idx + 1, len(bis)):
            bi = bis[bi_idx]
            if bi.type == "up" and bi.high > start_bi.high:
                has_pohuai = True
            break
        
        span = end_bi_idx - start_bi_idx
        print(f"  FOUND at tzxl[{i}]: prev.min={prev.min:.1f} curr.min={curr.min:.1f} nxt.min={nxt.min:.1f}")
        print(f"    end_bi=bi[{end_bi.index}] ({end_bi.type}) → end_bi_idx={end_bi_idx}")
        print(f"    span={span} (need >= 2)")
        
        # Check bi_pohuai in detail
        if has_pohuai:
            for bi_idx in range(last_line_idx + 1, len(bis)):
                bi = bis[bi_idx]
                if bi.type == "up" and bi.high > start_bi.high:
                    print(f"    bi_pohuai: bi[{bi.index}] ({bi.type} h={bi.high:.1f}) > start.high={start_bi.high:.1f}")
                break
            print(f"    → REJECTED (bi_pohuai)")
        elif span < 2:
            print(f"    → REJECTED (< 3 strokes)")
        else:
            print(f"    → ACCEPTED! XD down bi[{start_bi_idx}]→bi[{end_bi_idx}]")
            break
    else:
        print(f"  [{i}] NOT fx: curr.min={curr.min:.1f} prev.min={prev.min:.1f} nxt.min={nxt.min:.1f}")
