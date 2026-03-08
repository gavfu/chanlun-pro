"""Test different bi_pohuai interpretations for the first down XD."""
import sys; sys.path.insert(0, 'src')
import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_interface import TZXL

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')
cd_o = CL_O('BTC/USDT', '60m')
cd_o.process_klines(df)
bis = cd_o.bis

start_bi_idx = 2
start_bi = bis[start_bi_idx]
print(f"start_bi: bi[{start_bi_idx}] {start_bi.type} {start_bi.start.k.index}→{start_bi.end.k.index} h={start_bi.high:.1f}")

# The sequence fractal at tzxl[2] has curr_xl lines = [bi[7], bi[9]]
# last_line_idx = 9
last_line_idx = 9

print(f"\n=== Checking ALL subsequent strokes after bi[{last_line_idx}] for bi_pohuai ===")
print(f"Looking for UP strokes with high > start_bi.high = {start_bi.high:.1f}")
for bi_idx in range(last_line_idx + 1, len(bis)):
    bi = bis[bi_idx]
    if bi.type == "up":
        exceeds = bi.high > start_bi.high
        marker = " *** POHUAI ***" if exceeds else ""
        print(f"  bi[{bi.index}] {bi.type} {bi.start.k.index}→{bi.end.k.index} h={bi.high:.1f}{marker}")

# Now check with start as bi[0]
print(f"\n=== Now check bi_pohuai if start was bi[0] ===")
start0 = bis[0]
print(f"start_bi: bi[0] {start0.type} h={start0.high:.1f}")
for bi_idx in range(last_line_idx + 1, len(bis)):
    bi = bis[bi_idx]
    if bi.type == "up":
        exceeds = bi.high > start0.high
        marker = " *** POHUAI ***" if exceeds else ""
        print(f"  bi[{bi.index}] {bi.type} {bi.start.k.index}→{bi.end.k.index} h={bi.high:.1f}{marker}")

# Also: what about checking the characteristic sequence fractal "next" element  
# rather than the last line? In our implementation, the "next" TZXL after the fractal
# is tzxl[3] = bi[11]. After bi[11] would be bi[12]...
# But let me try: for DOWN xd, pohuai = any subsequent UP bi exceeds start.high
# This is a FULL forward look from the sequence fractal onwards.

print(f"\n=== Hypothesis 1: bi_pohuai = ANY future UP stroke > start.high ===")
# From the sequence fractal at tzxl[2] (bi[7,9]), looking forward
for bi_idx in range(last_line_idx + 1, len(bis)):
    bi = bis[bi_idx]
    if bi.type == "up" and bi.high > start_bi.high:
        print(f"  bi[{bi.index}] {bi.type} h={bi.high:.1f} > {start_bi.high:.1f} → POHUAI!")
        break
    elif bi.type == "up":
        print(f"  bi[{bi.index}] {bi.type} h={bi.high:.1f} < {start_bi.high:.1f} → ok")

print(f"\n=== Hypothesis 2: bi_pohuai = the NEXT opposite-type stroke ===")
# For DOWN xd, the next stroke that's the SAME type as xd_type after the fractal
# i.e., the reaction stroke
# After tzxl[2] (bi[7,9]'s fractal), the confirming tzxl is tzxl[3] = bi[11]
# After bi[11], bi[12] is DOWN. Does bi[12].low < something? 
# No, for DOWN xd, we check if UP bi > start.high

# Let me look at what the "next" TZXL's subsequent stroke does
# tzxl[3] = bi[11], so after bi[11]:
bi_after_confirm = bis[12]
print(f"  After confirming tzxl bi[11]: bi[12] = {bi_after_confirm.type} "
      f"{bi_after_confirm.start.k.index}→{bi_after_confirm.end.k.index} "
      f"h={bi_after_confirm.high:.1f} l={bi_after_confirm.low:.1f}")
# bi[12] is DOWN, so for a DOWN XD, we check UP strokes
# Next UP after bi[12] is bi[13]
bi13 = bis[13]
print(f"  bi[13] = {bi13.type} h={bi13.high:.1f} l={bi13.low:.1f}")
# Does bi[13] or any subsequent UP stroke go above start.high?

print(f"\n=== Hypothesis 3: Simple - check if next_xl's line goes beyond start ===")
# The "next" TZXL (tzxl[3]) is bi[11], UP, h=68687.0
# Does bi[11].high > start_bi.high (70110.9)?
print(f"  tzxl[3] = bi[11] h={bis[11].high:.1f} > start.high={start_bi.high:.1f}? {bis[11].high > start_bi.high}")
# No, 68687 < 70110.9. So this doesn't trigger.

print(f"\n=== Hypothesis 4: Check using start FX's high value (not bi's high) ===")
# start at bi[2] → start_fx = bi[2].start = ding@20 with val=70110.9
print(f"  start_fx: {start_bi.start.type}@{start_bi.start.k.index} val={start_bi.start.val:.1f}")
# Using start_fx.val would be the same as start_bi.high in this case

print(f"\n=== Hypothesis 5: bi_pohuai checks with the sequence fractal's HIGH, not START ===") 
# For DOWN xd at tzxl[2]: the sequence fractal has max=67299.4 (merged bi[7,9])
# After this fx, does any UP stroke exceed the fx HIGH?
# i.e., does any bi.high > 67299.4?
fx_max = 67299.4
print(f"  fx_max = {fx_max:.1f}")
for bi_idx in range(last_line_idx + 1, len(bis)):
    bi = bis[bi_idx]
    if bi.type == "up" and bi.high > fx_max:
        print(f"  bi[{bi.index}] {bi.type} h={bi.high:.1f} > fx_max → POHUAI!")
        break
    elif bi.type == "up":
        print(f"  bi[{bi.index}] {bi.type} h={bi.high:.1f} ≤ fx_max → ok")
# bi[11] h=68687 > fx_max=67299.4 → POHUAI!

print(f"\n=== Hypothesis 6: Looking from next_xl onwards for bi that breaks PREVIOUS xl ===")
# The confirming element is tzxl[3] = bi[11]. 
# Check if after bi[11], there's a stroke that goes below curr_xl.min (65595.7)
# showing the downtrend continues
curr_xl_min = 65595.7
print(f"  curr_xl.min = {curr_xl_min:.1f}")
print(f"  Checking if after bi[11], any DOWN stroke goes below {curr_xl_min:.1f}...")
for bi_idx in range(12, len(bis)):
    bi = bis[bi_idx]
    if bi.type == "down" and bi.low < curr_xl_min:
        print(f"  bi[{bi.index}] {bi.type} l={bi.low:.1f} < {curr_xl_min:.1f} → Downtrend continues!")
        break
    elif bi.type == "down":
        print(f"  bi[{bi.index}] {bi.type} l={bi.low:.1f} ≥ {curr_xl_min:.1f}")
