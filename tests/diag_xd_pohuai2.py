"""Comprehensive test of bi_pohuai variations."""
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

# Build TZXL
tzxl_bis = [bi for bi in bis[start_bi_idx:] if bi.type == "up"]
tzxls = []
for bi in tzxl_bis:
    pre_line = bis[bi.index - 1] if bi.index > 0 else bi
    new_tzxl = TZXL(bh_direction="down", line=bi, pre_line=pre_line,
                    line_bad=False, done=bi.is_done())
    if not tzxls:
        tzxls.append(new_tzxl)
        continue
    last = tzxls[-1]
    if (last.max >= new_tzxl.max and last.min <= new_tzxl.min) or \
       (new_tzxl.max >= last.max and new_tzxl.min <= last.min):
        last.lines.append(bi)
        last.done = bi.is_done()
        last.update_maxmin()
    else:
        tzxls.append(new_tzxl)

# The sequence fractal at tzxl[2]: {bi[7,9], max=67299.4, min=65595.7}
# Confirming element: nxt = tzxl[3] = {bi[11], max=68687.0, min=66915.0}
# 
# Pyarmor first XD ends at bi[14] — the end line of tzxl[4] = {bi[13,15]}
# So somehow the sequence fractal at tzxl[2] is rejected.

i = 2  # sequence fractal index
curr_xl = tzxls[i]
nxt_xl = tzxls[i + 1]

print("=== Sequence Fractal at tzxl[2] ===")
print(f"  curr: max={curr_xl.max:.1f} min={curr_xl.min:.1f} lines=bi{[l.index for l in curr_xl.lines]}")
print(f"  nxt: max={nxt_xl.max:.1f} min={nxt_xl.min:.1f} lines=bi{[l.index for l in nxt_xl.lines]}")

# Variant A: Check from nxt_xl's last line (not curr_xl's last line)
# nxt_xl = bi[11], so check from bi[12]
nxt_last_idx = nxt_xl.lines[-1].index
print(f"\n=== Variant A: Check from AFTER confirming tzxl (bi[{nxt_last_idx}+1]) ===")
for bi_idx in range(nxt_last_idx + 1, len(bis)):
    bi = bis[bi_idx]
    if bi.type == "up":
        exceeds_start = bi.high > start_bi.high
        exceeds_fx = bi.high > curr_xl.max
        print(f"  bi[{bi.index}] UP h={bi.high:.1f} > start.h={start_bi.high:.1f}? {exceeds_start} | > fx.max={curr_xl.max:.1f}? {exceeds_fx}")
        break  # only check first UP
    else:
        print(f"  bi[{bi.index}] DOWN (skip)")

# Variant B: Check nxt_xl itself - does the confirming element exceed the fx?
print(f"\n=== Variant B: Does confirming element exceed the fractal? ===")
print(f"  nxt.max={nxt_xl.max:.1f} > curr.max={curr_xl.max:.1f}? {nxt_xl.max > curr_xl.max}")
# 68687 > 67299.4 → TRUE! The confirming element's max exceeds the fractal's max.
# This could be the check: for a DOWN XD's di sequence fractal, 
# the fractal is valid only if nxt.max < curr.max (the trend continues down).
# But wait, that's opposite of what a di fractal means...

# Actually for a DI sequence fractal in a DOWN segment:
# prev, curr, nxt where curr.min is lowest
# For confirmation, we need the move AFTER curr to bounce back up (nxt.min > curr.min)
# But nxt.max > curr.max means the bounce-back goes even HIGHER than the fractal area

# Variant C: "Gapless" check - does nxt overlap with prev?
print(f"\n=== Variant C: Does nxt overlap with prev? (gap check) ===")
prev_xl = tzxls[i - 1]
print(f"  prev: max={prev_xl.max:.1f} min={prev_xl.min:.1f}")
print(f"  nxt: max={nxt_xl.max:.1f} min={nxt_xl.min:.1f}")
has_gap = nxt_xl.max < prev_xl.min or nxt_xl.min > prev_xl.max
print(f"  Has gap (no overlap)? {has_gap}")
# nxt.max=68687 vs prev.min=66588 → overlap. No gap.

# Variant D: For DOWN XD, the di fractal should have a GAP between prev and nxt
# meaning nxt.max < prev.min (nxt is entirely below prev)
print(f"  nxt.max < prev.min? {nxt_xl.max < prev_xl.min}")
# 68687 < 66588? No.

# Variant E: Check whether the segment would be valid by directional test
# For a DOWN segment: end should be lower than start
# end_bi_idx = 8 (bi[8] down 74→80)
# bi[8].low = 65595.7, start_bi.low (bi[2]) = 67785.4 → yes, endpoint is lower
print(f"\n=== Variant E: Direction validity ===")
end_bi_idx = 8  # where we'd end if fractal accepted
end_bi = bis[end_bi_idx]
print(f"  DOWN xd: end_bi.low={end_bi.low:.1f} < start_bi.start.val={start_bi.start.val:.1f}? {end_bi.low < start_bi.start.val}")

# Now let's check the SECOND fractal at tzxl[4]
print(f"\n{'='*60}")
print("=== Checking Sequence Fractal at tzxl[4] ===")
i = 4
prev_xl = tzxls[i - 1]
curr_xl = tzxls[i]
nxt_xl = tzxls[i + 1]
print(f"  prev: max={prev_xl.max:.1f} min={prev_xl.min:.1f} lines=bi{[l.index for l in prev_xl.lines]}")
print(f"  curr: max={curr_xl.max:.1f} min={curr_xl.min:.1f} lines=bi{[l.index for l in curr_xl.lines]}")
print(f"  nxt: max={nxt_xl.max:.1f} min={nxt_xl.min:.1f} lines=bi{[l.index for l in nxt_xl.lines]}")

# Check di fractal
is_fx = curr_xl.min < prev_xl.min and curr_xl.min < nxt_xl.min
print(f"  Is DI fractal? {is_fx}")

# End bi
end_bi = min(curr_xl.lines, key=lambda l: l.low)
end_bi_idx = end_bi.index
if bis[end_bi_idx].type == "up" and end_bi_idx > 0:
    end_bi_idx -= 1
print(f"  end_bi=bi[{end_bi.index}] ({end_bi.type} l={end_bi.low:.1f}) → end_bi_idx={end_bi_idx}")
# bi[15] is UP, so end_bi_idx = 14 ← THIS MATCHES PYARMOR!

# Variant B for tzxl[4]:
print(f"  nxt.max={nxt_xl.max:.1f} > curr.max={curr_xl.max:.1f}? {nxt_xl.max > curr_xl.max}")
# 68188.8 > 66240.4 → TRUE. So Variant B would also reject this.

# Let me also check the BI after nxt for all variants
nxt_last_idx = nxt_xl.lines[-1].index
print(f"\n  Strokes after nxt (bi[{nxt_last_idx}+1]):")
for bi_idx in range(nxt_last_idx + 1, min(nxt_last_idx + 5, len(bis))):
    bi = bis[bi_idx]
    print(f"    bi[{bi.index}] {bi.type} {bi.start.k.index}→{bi.end.k.index} h={bi.high:.1f} l={bi.low:.1f}")

# Let me check: does the current code use `start_bi.high` in bi_pohuai?
# start_bi = bis[2], start_bi.high = 70110.9
# From nxt_last_idx+1 = bi[20]:
print(f"\n  Pohuai check (original, from nxt last + 1):")
for bi_idx in range(nxt_last_idx + 1, len(bis)):
    bi = bis[bi_idx]
    if bi.type == "up" and bi.high > start_bi.high:
        print(f"    bi[{bi.index}] h={bi.high:.1f} > {start_bi.high:.1f} → POHUAI")
        break
    elif bi.type == "up":
        print(f"    bi[{bi.index}] h={bi.high:.1f} ≤ {start_bi.high:.1f} → ok")
        break
    else:
        print(f"    bi[{bi.index}] DOWN (skip)")

# From curr's last line + 1:
curr_last_idx = curr_xl.lines[-1].index
print(f"\n  Pohuai check (from curr last + 1 = bi[{curr_last_idx}+1]):")
for bi_idx in range(curr_last_idx + 1, len(bis)):
    bi = bis[bi_idx]
    if bi.type == "up" and bi.high > start_bi.high:
        print(f"    bi[{bi.index}] h={bi.high:.1f} > {start_bi.high:.1f} → POHUAI")
        break
    elif bi.type == "up":
        print(f"    bi[{bi.index}] h={bi.high:.1f} ≤ {start_bi.high:.1f} → ok")
        break
    else:
        print(f"    bi[{bi.index}] DOWN, skip")
