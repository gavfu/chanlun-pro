"""
Check BTC5m up bi[25] - why bh gives end=49 (wrong)
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_open import CL
from chanlun.cl_interface import TZXL

df = pd.read_parquet("tests/test_data/BTC_USDT_5m_1000.parquet")
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

cl = CL("BTC5m", "5m", config)
cl.process_klines(df)
bis = cl.get_bis()

print("=== BTC5m up bi[25] → expected end=27 ===")
for i in range(25, 35):
    if i < len(bis):
        print(f"  bi[{i}]: type={bis[i].type}, high={bis[i].high:.1f}, low={bis[i].low:.1f}")

def build_tzxls_up(bis, start, mode):
    tzxl_bis = [b for b in bis[start:] if b.type == "down"]
    tzxls = []
    for bi in tzxl_bis:
        pre_line = bis[bi.index - 1] if bi.index > 0 else bi
        new_tzxl = TZXL(
            bh_direction="up", line=bi, pre_line=pre_line,
            line_bad=False, done=bi.is_done(),
        )
        if len(tzxls) == 0:
            tzxls.append(new_tzxl)
            continue
        last_tzxl = tzxls[-1]
        o_c_n = last_tzxl.max >= new_tzxl.max and last_tzxl.min <= new_tzxl.min
        n_c_o = new_tzxl.max >= last_tzxl.max and new_tzxl.min <= last_tzxl.min
        if o_c_n:
            last_tzxl.lines.append(bi)
            last_tzxl.done = bi.is_done()
            last_tzxl.line_bad = False
            last_tzxl.update_maxmin()
        elif n_c_o:
            if mode == 'bh':
                last_tzxl.lines.append(bi)
                last_tzxl.done = bi.is_done()
                last_tzxl.line_bad = False
                last_tzxl.update_maxmin()
            else:
                new_tzxl.line_bad = True
                tzxls.append(new_tzxl)
        else:
            tzxls.append(new_tzxl)
    return tzxls

for mode in ['no_bh', 'bh']:
    tzxls = build_tzxls_up(bis, 25, mode)
    print(f"\n  {mode} TZXLs ({len(tzxls)}):")
    for i, t in enumerate(tzxls[:15]):
        lines = [l.index for l in t.lines]
        print(f"    [{i}] max={t.max:.1f} min={t.min:.1f} bad={t.line_bad} lines={lines}")
    
    print(f"\n  DING FX search:")
    for i in range(1, min(len(tzxls) - 1, 12)):
        curr = tzxls[i]
        prev_xl = tzxls[i - 1]
        next_xl = tzxls[i + 1]
        is_fx = curr.max > prev_xl.max and curr.max > next_xl.max
        if is_fx:
            end_bi = max(curr.lines, key=lambda l: l.high)
            end_bi_idx = end_bi.index
            if bis[end_bi_idx].type == "down" and end_bi_idx > 0:
                end_bi_idx -= 1
            lines = [l.index for l in curr.lines]
            print(f"    FX at [{i}] lines={lines} max={curr.max:.1f} bad={curr.line_bad} → end={end_bi_idx}")
