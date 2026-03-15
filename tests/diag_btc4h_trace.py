"""
深入追踪 BTC4h5k ZSD 差异：
1. 打印 _find_xd_end 的 TZXL FX 选择过程
2. 检查 _split_xds 是否导致额外拆分
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P
from chanlun.cl_interface import TZXL, XLFX, XD

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
for i, xd in enumerate(xds_p):
    xd.index = i

cl_o = CL_O("test", "test", CL_CONFIG)
cl_o.process_klines(df)

# Patch _build_xds to skip _split_xds and trace the raw ZSD result
print("=== Raw ZSDs (before _split_xds) ===")
bis = list(xds_p)
xd_type, start_bi_idx = cl_o._find_first_xd_start(bis)
print(f"_find_first_xd_start → type={xd_type}, start_idx={start_bi_idx}")

raw_zsds = []
si = start_bi_idx
while si < len(bis):
    result = cl_o._find_xd_end(bis, si, xd_type)
    if result is None:
        if si < len(bis) - 2:
            end_bi_idx = len(bis) - 1
            if bis[end_bi_idx].type != xd_type:
                end_bi_idx -= 1
            zd = cl_o._create_xd(bis, si, end_bi_idx, xd_type, len(raw_zsds), done=False)
            if zd:
                raw_zsds.append(zd)
        break
    end_bi_idx, ding_fx, di_fx, tzxls = result
    zd = cl_o._create_xd(bis, si, end_bi_idx, xd_type, len(raw_zsds), done=True,
                          ding_fx=ding_fx, di_fx=di_fx, tzxls=tzxls)
    if zd:
        raw_zsds.append(zd)
        si = end_bi_idx + 1
        xd_type = "down" if xd_type == "up" else "up"
    else:
        si += 1

print(f"\nRaw ZSDs (before split): {len(raw_zsds)}")
for i, zsd in enumerate(raw_zsds):
    try:
        print(f"  zsd[{i}] {zsd.type:>4} xd[{zsd.start_line.index}→{zsd.end_line.index}] done={zsd.done}")
    except:
        print(f"  zsd[{i}] error")

print(f"\ncl_pyarmor ZSDs: {len(cl_p.get_zsds())}")
for i, zsd in enumerate(cl_p.get_zsds()):
    try:
        print(f"  zsd[{i}] {zsd.type:>4} xd[{zsd.start_line.index}→{zsd.end_line.index}] done={zsd.done}")
    except:
        print(f"  zsd[{i}] error")

# Now trace the FX detection for UP ZSD starting at xd[3] (from _find_xd_end)
print(f"\n=== Tracing _find_xd_end(bis, {start_bi_idx}, '{xd_type}') → actually trace from start=3 ===")
start = 3  # pyarmor ZSD[0] starts at xd[3] up
ut = "up"
tzxl_bi_type = "down"
bh_direction = "up"

tzxl_bis = [b for b in bis[start:] if b.type == tzxl_bi_type]
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
    if old_c_new:
        last.lines.append(bi)
        last.done = bi.is_done()
        last.line_bad = False
        last.update_maxmin()
    elif new_c_old:
        new_tzxl.line_bad = True
        tzxls.append(new_tzxl)
    else:
        tzxls.append(new_tzxl)

print(f"\nTZXL elements ({len(tzxls)} total):")
for i, t in enumerate(tzxls):
    xd_indices = [xd.index for xd in t.lines]
    print(f"  tzxl[{i:2}] max={t.max:12.2f} min={t.min:12.2f} line_bad={t.line_bad} lines={xd_indices}")

print(f"\nFX scan:")
for i in range(1, len(tzxls)-1):
    p, c, n = tzxls[i-1], tzxls[i], tzxls[i+1]
    is_ding = c.max > p.max and c.max > n.max
    marking = ""
    if is_ding:
        marking = f" ← DING FX  prev_bad={p.line_bad} curr_bad={c.line_bad}"
    print(f"  i={i:2} prev.max={p.max:12.2f} curr.max={c.max:12.2f} next.max={n.max:12.2f}"
          f" curr.line_bad={c.line_bad}{marking}")
