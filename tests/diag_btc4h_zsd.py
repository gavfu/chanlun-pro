"""
诊断 BTC4h5k ZSD[0] 差异：
cl_open 在 xd[5] 找到 ZSD 结束，cl_pyarmor 在 xd[29] 才结束。
打印 cl_pyarmor XD 列表的高低点，追踪特征序列构建过程。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P
from chanlun.cl_interface import TZXL

CL_CONFIG = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
}

df = pd.read_parquet('tests/test_data/BTC_USDT_4h_5000.parquet')

cl_p = CL_P("test", "test", CL_CONFIG)
cl_p.process_klines(df)
xds_p = cl_p.get_xds()

print(f"cl_pyarmor has {len(xds_p)} XDs")
print(f"\n--- All pyarmor XDs: idx  type  high         low          bi_start→end ---")
for xd in xds_p:
    print(f"  xd[{xd.index:2}] {xd.type:4}  high={xd.high:12.2f}  low={xd.low:12.2f}  "
          f"bi[{xd.start.index}→{xd.end.index}]  done={xd.done}")

# Now trace _find_all_tzxl_fx for DOWN xds (which form the characteristic seq of UP ZSD)
print(f"\n--- DOWN XDs (characteristic sequence for UP ZSD) ---")
down_xds = [xd for xd in xds_p if xd.type == "down"]
for xd in down_xds:
    print(f"  xd[{xd.index:2}]  high={xd.high:12.2f}  low={xd.low:12.2f}  bi[{xd.start.index}→{xd.end.index}]")

# Manually simulate TZXL building for DOWN XDs starting from index 3 (first UP ZSD starts)
print(f"\n--- Simulate TZXL building for DOWN XDs from xd[4] onward (bh_direction=up) ---")
# For UP ZSD, we use DOWN XDs, bh_direction="up"
bis = list(xds_p)  # re-index
for i, xd in enumerate(bis):
    xd.index = i

tzxls = []
bh_direction = "up"
bi_type = "down"
down_bis = [bi for bi in bis if bi.type == bi_type and bi.index >= 4]  # from xd[4]

for bi in down_bis:
    pre_line = bis[bi.index - 1] if bi.index > 0 else bi
    new_tzxl = TZXL(
        bh_direction=bh_direction, line=bi, pre_line=pre_line,
        line_bad=False, done=bi.index < len(bis) - 1,
    )
    if len(tzxls) == 0:
        tzxls.append(new_tzxl)
        print(f"  Add  xd[{bi.index:2}]  max={new_tzxl.max:12.2f}  min={new_tzxl.min:12.2f}")
        continue
    last = tzxls[-1]
    old_contains_new = last.max >= new_tzxl.max and last.min <= new_tzxl.min
    new_contains_old = new_tzxl.max >= last.max and new_tzxl.min <= last.min
    if old_contains_new:
        last.lines.append(bi)
        last.done = bi.index < len(bis) - 1
        last.line_bad = False
        last.update_maxmin()
        print(f"  Merge xd[{bi.index:2}] into tzxl[{len(tzxls)-1}]  max={last.max:12.2f}  min={last.min:12.2f}  lines={len(last.lines)}")
    elif new_contains_old:
        new_tzxl.line_bad = True
        tzxls.append(new_tzxl)
        print(f"  Add* xd[{bi.index:2}]  max={new_tzxl.max:12.2f}  min={new_tzxl.min:12.2f}  line_bad=True (new contains old)")
    else:
        tzxls.append(new_tzxl)
        print(f"  Add  xd[{bi.index:2}]  max={new_tzxl.max:12.2f}  min={new_tzxl.min:12.2f}")

print(f"\n  Total TZXL elements: {len(tzxls)}")
print(f"\n--- TZXL FX check (looking for DING FX) ---")
for i in range(1, len(tzxls) - 1):
    l = tzxls[i-1]
    m = tzxls[i]
    r = tzxls[i+1]
    # DING: middle has highest max
    if m.max > l.max and m.max > r.max:
        print(f"  DING FX at tzxl[{i}]:  left max={l.max:.2f}  mid max={m.max:.2f}  right max={r.max:.2f}")
    else:
        print(f"  No FX at tzxl[{i}]: left max={l.max:.2f}  mid max={m.max:.2f}  right max={r.max:.2f}  line_bad={m.line_bad}")
