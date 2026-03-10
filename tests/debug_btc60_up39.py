"""
Detailed bh analysis for BTC60 up bi[39]
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

df = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cl = CL_Open("BTC60", "60m", config)
cl.process_klines(df)
bis = cl.get_bis()

start_bi = 39
xd_type = "up"
tzxl_bi_type = "down"
bh_direction = "up"

# Build no_bh
tzxl_bis = [b for b in bis[start_bi:] if b.type == tzxl_bi_type]

print("no_bh construction:")
tzxls_nobh = []
for bi in tzxl_bis:
    pre_line = bis[bi.index - 1] if bi.index > 0 else bi
    done = bi.is_done()
    new_tzxl = TZXL(bh_direction=bh_direction, line=bi, pre_line=pre_line, line_bad=False, done=done)
    if not tzxls_nobh:
        tzxls_nobh.append(new_tzxl)
        bis_str = ",".join([str(l.index) for l in new_tzxl.lines])
        print(f"  First TZXL: bi[{bis_str}] max={new_tzxl.max:.1f} min={new_tzxl.min:.1f}")
        continue
    last = tzxls_nobh[-1]
    old_c_new = last.max >= new_tzxl.max and last.min <= new_tzxl.min
    new_c_old = new_tzxl.max >= last.max and new_tzxl.min <= last.min
    bis_str_new = ",".join([str(l.index) for l in new_tzxl.lines])
    bis_str_last = ",".join([str(l.index) for l in last.lines])
    if old_c_new:
        print(f"  bi[{bis_str_new}] max={new_tzxl.max:.1f} min={new_tzxl.min:.1f} → OLD⊃NEW merge into [{bis_str_last}]")
        last.lines.append(bi)
        last.done = done
        last.line_bad = False
        last.update_maxmin()
    elif new_c_old:
        print(f"  bi[{bis_str_new}] max={new_tzxl.max:.1f} min={new_tzxl.min:.1f} → NEW⊃OLD (vs [{bis_str_last}] max={last.max:.1f} min={last.min:.1f}), bad=True")
        new_tzxl.line_bad = True
        tzxls_nobh.append(new_tzxl)
    else:
        print(f"  bi[{bis_str_new}] max={new_tzxl.max:.1f} min={new_tzxl.min:.1f} → no containment")
        tzxls_nobh.append(new_tzxl)

print(f"\nno_bh TZXLs ({len(tzxls_nobh)}):")
for i, xl in enumerate(tzxls_nobh):
    bis_str = ",".join([str(l.index) for l in xl.lines])
    print(f"  [{i}] bi[{bis_str}] max={xl.max:.1f} min={xl.min:.1f} bad={xl.line_bad}")

# Build bh
print("\nbh construction:")
tzxls_bh = []
for bi in tzxl_bis:
    pre_line = bis[bi.index - 1] if bi.index > 0 else bi
    done = bi.is_done()
    new_tzxl = TZXL(bh_direction=bh_direction, line=bi, pre_line=pre_line, line_bad=False, done=done)
    if not tzxls_bh:
        tzxls_bh.append(new_tzxl)
        bis_str = ",".join([str(l.index) for l in new_tzxl.lines])
        print(f"  First TZXL: bi[{bis_str}] max={new_tzxl.max:.1f} min={new_tzxl.min:.1f}")
        continue
    last = tzxls_bh[-1]
    old_c_new = last.max >= new_tzxl.max and last.min <= new_tzxl.min
    new_c_old = new_tzxl.max >= last.max and new_tzxl.min <= last.min
    bis_str_new = ",".join([str(l.index) for l in new_tzxl.lines])
    bis_str_last = ",".join([str(l.index) for l in last.lines])
    if old_c_new or new_c_old:
        ctype = "OLD⊃NEW" if old_c_new else "NEW⊃OLD"
        print(f"  bi[{bis_str_new}] max={new_tzxl.max:.1f} min={new_tzxl.min:.1f} → {ctype} merge into [{bis_str_last}]")
        last.lines.append(bi)
        last.done = done
        last.line_bad = False
        last.update_maxmin()
    else:
        print(f"  bi[{bis_str_new}] max={new_tzxl.max:.1f} min={new_tzxl.min:.1f} → no containment")
        tzxls_bh.append(new_tzxl)

print(f"\nbh TZXLs ({len(tzxls_bh)}):")
for i, xl in enumerate(tzxls_bh):
    bis_str = ",".join([str(l.index) for l in xl.lines])
    print(f"  [{i}] bi[{bis_str}] max={xl.max:.1f} min={xl.min:.1f} bad={xl.line_bad}")

# Now check: is bi[39→41] the answer from pyarmor's split logic?
# Pyarmor has down[28→38] then up[39→41] (split=笔破坏)
# So up[39→41] is created by the SPLIT logic, not by _find_xd_end!
print(f"\n\nPYARMOR CONTEXT:")
from chanlun.cl_pyarmor import CL as CL_P
cl_p = CL_P("BTC60", "60m", config)
cl_p.process_klines(df)
xds_p = cl_p.get_xds()
for i, xd in enumerate(xds_p):
    split_str = f" split={xd.is_split}" if xd.is_split else ""
    print(f"  xd[{i}] {xd.type} bi[{xd.start_line.index}→{xd.end_line.index}]{split_str}")
