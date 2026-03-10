"""
Investigate ETH60 down bi[34] segment.
Pyarmor: down bi[34→40]
Our "first FX" sim: down bi[34→36] (wrong)
Our "more extreme": down bi[34→40] (correct)

What's the difference? Check both bh and no_bh FX results.
And check what our code finds as FX candidates.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor
from chanlun.cl_interface import TZXL

df = pd.read_parquet("tests/test_data/ETH_USDT_60m_1000.parquet")
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

cl_open = CL_Open("test", "60m", config)
cl_open.process_klines(df)
cl_pyarmor = CL_Pyarmor("test", "60m", config)
cl_pyarmor.process_klines(df)
bis = cl_pyarmor.get_bis()

# For down bi[34], looking for DI FX in UP BIs
# bh_direction = "down" → TZXL uses UP BIs 
print("=== ETH60 down bi[34]: DI FX ===")
print("  Expected: end at bi[40]")
print()

# Check with full range
subset = bis[34:50]  # generous range
print("=== Pyarmor _xd_cal_line_xlfx ===")
tzxls_bh, xlfxs_bh = cl_pyarmor._xd_cal_line_xlfx(subset, 'di', 'bh')
print(f"  bh TZXLs ({len(tzxls_bh)}):")
for i, t in enumerate(tzxls_bh):
    print(f"    [{i}]: max={t.max} min={t.min} bad={t.line_bad} lines={[l.index for l in t.lines]}")
print(f"  bh XLFXs ({len(xlfxs_bh)}):")
for i, f in enumerate(xlfxs_bh[:3]):
    print(f"    [{i}]: type={f.type} bad={f.is_line_bad} xl_lines={[l.index for l in f.xl.lines]}")

tzxls_no, xlfxs_no = cl_pyarmor._xd_cal_line_xlfx(subset, 'di', 'no_bh')
print(f"\n  no_bh TZXLs ({len(tzxls_no)}):")
for i, t in enumerate(tzxls_no):
    print(f"    [{i}]: max={t.max} min={t.min} bad={t.line_bad} lines={[l.index for l in t.lines]}")
print(f"  no_bh XLFXs ({len(xlfxs_no)}):")
for i, f in enumerate(xlfxs_no[:3]):
    print(f"    [{i}]: type={f.type} bad={f.is_line_bad} xl_lines={[l.index for l in f.xl.lines]}")

# Now check what OUR code does for down bi[34]
print("\n=== Our code: TZXL analysis for down bi[34] ===")
our_bis = cl_open.get_bis()
# Build TZXL with our rules (no_bh mode)
up_bis = [bi for bi in our_bis[34:] if bi.type == 'up'][:8]
tzxls = []
for bi in up_bis:
    pre_line = our_bis[bi.index - 1] if bi.index > 0 else bi
    new_tzxl = TZXL(
        bh_direction="down",
        line=bi,
        pre_line=pre_line,
        line_bad=False,
        done=bi.is_done(),
    )
    if len(tzxls) == 0:
        tzxls.append(new_tzxl)
        continue
    
    last = tzxls[-1]
    o_c_n = last.max >= new_tzxl.max and last.min <= new_tzxl.min
    n_c_o = new_tzxl.max >= last.max and new_tzxl.min <= last.min
    
    if o_c_n:
        last.lines.append(bi)
        last.done = bi.is_done()
        last.line_bad = False
        last.update_maxmin()
        print(f"  bi[{bi.index}] → OLD⊃NEW merge, lines={[l.index for l in last.lines]}")
    elif n_c_o:
        new_tzxl.line_bad = True
        tzxls.append(new_tzxl)
        print(f"  bi[{bi.index}] → NEW⊃OLD, separate, bad=True")
    else:
        tzxls.append(new_tzxl)
        print(f"  bi[{bi.index}] → no containment")

print(f"\n  Final TZXLs ({len(tzxls)}):")
for i, t in enumerate(tzxls):
    print(f"    [{i}]: max={t.max} min={t.min} bad={t.line_bad} lines={[l.index for l in t.lines]}")

# Check FX candidates
print(f"\n  FX candidates (DI):")
for i in range(1, len(tzxls) - 1):
    curr = tzxls[i]
    prev = tzxls[i-1]
    next_ = tzxls[i+1]
    if curr.min < prev.min and curr.min < next_.min:
        print(f"    TZXL[{i}] is DI FX: min={curr.min}, bad={curr.line_bad}, lines={[l.index for l in curr.lines]}")
        # What end_bi would this give?
        # For DI FX, the end_bi is the most extreme low in the lines
        min_bi = min(curr.lines, key=lambda l: l.low)
        end_bi_idx = min_bi.index + 1  # next bi after the lowest
        print(f"      most extreme bi: [{min_bi.index}], low={min_bi.low}")
        print(f"      → segment end bi would be: {min_bi.index + 1}")
