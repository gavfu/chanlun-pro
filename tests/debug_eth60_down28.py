"""
Check ETH60 _find_xd_end(28, "down") - does it have a bad FX?
Also check if "always-first" changes this.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_interface import TZXL

config = {
    "bi_type": "bi_type_old", "fx_qj": "fx_qj_k", "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes", "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1", "xd_bzh": "xd_bzh_no",
    "xd_zs_max_lines_split": 11, "xd_allow_split_no_highlow": 1,
    "xd_allow_split_zs_kz": 0, "xd_allow_split_zs_more_line": 1,
    "xd_allow_split_zs_no_direction": 1,
}

df = pd.read_parquet("tests/test_data/ETH_USDT_60m_1000.parquet")
cl = CL_Open("ETH60", "60m", config)
cl.process_klines(df)
bis = cl.get_bis()

start_bi = 28
xd_type = "down"
tzxl_bi_type = "up"
bh_direction = "down"
target_fx_type = "di"

tzxl_bis = [b for b in bis[start_bi:] if b.type == tzxl_bi_type]

tzxls = []
for bi in tzxl_bis:
    pre_line = bis[bi.index - 1] if bi.index > 0 else bi
    done = bi.is_done()
    new_tzxl = TZXL(bh_direction=bh_direction, line=bi, pre_line=pre_line, line_bad=False, done=done)
    if not tzxls:
        tzxls.append(new_tzxl)
        continue
    last = tzxls[-1]
    old_c_new = last.max >= new_tzxl.max and last.min <= new_tzxl.min
    new_c_old = new_tzxl.max >= last.max and new_tzxl.min <= last.min
    if old_c_new:
        last.lines.append(bi)
        last.done = done
        last.line_bad = False
        last.update_maxmin()
    elif new_c_old:
        new_tzxl.line_bad = True
        tzxls.append(new_tzxl)
    else:
        tzxls.append(new_tzxl)

print(f"ETH60 down bi[{start_bi}], TZXLs ({len(tzxls)}):")
for i, xl in enumerate(tzxls):
    bis_str = ",".join([str(l.index) for l in xl.lines])
    print(f"  [{i}] bi[{bis_str}] max={xl.max:.1f} min={xl.min:.1f} bad={xl.line_bad}")

# Find FX
print(f"\nFX search:")
for i in range(1, len(tzxls) - 1):
    curr = tzxls[i]
    prev = tzxls[i - 1]
    nxt = tzxls[i + 1]
    if curr.min < prev.min and curr.min < nxt.min:
        bis_str = ",".join([str(l.index) for l in curr.lines])
        pohuai = cl._check_xd_bi_pohuai(bis, start_bi, curr, xd_type)
        # compute end
        end_bi = min(curr.lines, key=lambda l: l.low)
        end_idx = end_bi.index
        if bis[end_idx].type == "up" and end_idx > 0:
            end_idx -= 1
        gap = end_idx - start_bi
        print(f"  TZXL[{i}] bi[{bis_str}] min={curr.min:.1f} bad={curr.line_bad} pohuai={pohuai} end={end_idx} gap={gap}")

# Current result
result = cl._find_xd_end(bis, start_bi, xd_type)
print(f"\nCurrent _find_xd_end result: end={result[0] if result else None}")

# Also check ETH60's up bi[31] - does _find_xd_end(31, "up") change with always-first?  
# The real question: does the FIRST down segment [28→30] change with always-first?
# Our down[28→30] has no bad FX (end=30), so always-first doesn't change it!
# The cascade issue is different: our down[28→30] is the WRONG pre-split end.
# Pyarmor has down[28→41] as pre-split.
# So the problem is NOT the bad FX rule - it's that we get end=30 for down[28] while pyarmor gets 41.
print(f"\nETH60 down[28]: ends at 30 in our code.")
print(f"Pyarmor expected pre-split: down[28→41]")
print(f"This is likely a DIFFERENT issue from the bad FX rule.")
print(f"The FX at TZXL that gives end=30 is probably non-bad.")
