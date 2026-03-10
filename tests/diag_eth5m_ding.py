"""Trace ETH5m DOWN BI char seq to find why bad ding at bi[10] isn't found"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P
from chanlun.cl_interface import TZXL

CL_CONFIG = {
    "bi_type": "bi_type_old", "fx_qj": "fx_qj_k", "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes", "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1", "xd_bzh": "xd_bzh_no",
}

df = pd.read_parquet("tests/test_data/ETH_USDT_5m_1000.parquet")
cd_o = CL_O("test", "test", config=CL_CONFIG)
cd_o.process_klines(df)
cd_p = CL_P("test", "test", config=CL_CONFIG)
cd_p.process_klines(df)

bis = cd_o.get_bis()
bis_p = cd_p.get_bis()

# Show first 20 BIs
print("=== ETH5m First 20 BIs ===")
for i in range(min(20, len(bis))):
    bi = bis[i]
    print(f"  bi[{i:>2}] {bi.type:>4} h={bi.high:<10.1f} l={bi.low:<10.1f}")

# Build DOWN BI char seq (bh=up) and trace
print("\n=== DOWN BI Char Seq (bh=up) ===")
down_bis = [bi for bi in bis if bi.type == "down"]
tzxls = []
for bi in down_bis:
    new_tzxl = TZXL(bh_direction="up", line=bi,
                    pre_line=bis[bi.index - 1] if bi.index > 0 else bi,
                    line_bad=False, done=bi.is_done())
    if not tzxls:
        tzxls.append(new_tzxl)
        lines_str = ",".join([f"bi[{l.index}]" for l in new_tzxl.lines])
        print(f"  [{len(tzxls)-1}] max={new_tzxl.max:.1f} min={new_tzxl.min:.1f} bad={new_tzxl.line_bad} lines=[{lines_str}]")
        continue
    last = tzxls[-1]
    o_c_n = last.max >= new_tzxl.max and last.min <= new_tzxl.min
    n_c_o = new_tzxl.max >= last.max and new_tzxl.min <= last.min
    if o_c_n:
        last.lines.append(bi)
        last.done = bi.is_done()
        last.line_bad = False
        last.update_maxmin()
        lines_str = ",".join([f"bi[{l.index}]" for l in last.lines])
        print(f"  [{len(tzxls)-1}] MERGE max={last.max:.1f} min={last.min:.1f} lines=[{lines_str}]")
    elif n_c_o:
        new_tzxl.line_bad = True
        tzxls.append(new_tzxl)
        lines_str = ",".join([f"bi[{l.index}]" for l in new_tzxl.lines])
        print(f"  [{len(tzxls)-1}] max={new_tzxl.max:.1f} min={new_tzxl.min:.1f} BAD=True lines=[{lines_str}]")
    else:
        tzxls.append(new_tzxl)
        lines_str = ",".join([f"bi[{l.index}]" for l in new_tzxl.lines])
        print(f"  [{len(tzxls)-1}] max={new_tzxl.max:.1f} min={new_tzxl.min:.1f} bad=False lines=[{lines_str}]")
    if len(tzxls) > 15:
        break

# Check FX
print("\n=== FX Detection ===")
for i in range(1, len(tzxls) - 1):
    curr = tzxls[i]
    prev = tzxls[i-1]
    nxt = tzxls[i+1]
    if curr.max > prev.max and curr.max > nxt.max:
        key_bi = max(curr.lines, key=lambda l: l.high)
        print(f"  DING at [{i}] max={curr.max:.1f} bad={curr.line_bad} key_bi=bi[{key_bi.index}]")

# Also show pyarmor's first segment ding_fx
xds_p = cd_p.get_xds()
if xds_p:
    xd0 = xds_p[0]
    print(f"\n=== Pyarmor first XD: {xd0.type} bi[{xd0.start_line.index}→{xd0.end_line.index}] ===")
    if xd0.ding_fx:
        for j, xl in enumerate(xd0.ding_fx.xls):
            if xl:
                ls = ",".join([f"bi[{l.index}]" for l in xl.lines])
                print(f"  ding_fx.xls[{j}]: max={xl.max:.1f} min={xl.min:.1f} bad={xl.line_bad} lines=[{ls}]")
