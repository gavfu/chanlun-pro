"""
Compare bh vs no_bh for BTC5m up bi[3] (expected end=9)
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

# Show BIs
print("=== BTC5m up bi[3] ===")
print("UP segment, looking for DING FX in DOWN BIs")
for i in range(3, 15):
    if i < len(bis):
        print(f"  bi[{i}]: type={bis[i].type}, high={bis[i].high:.1f}, low={bis[i].low:.1f}")

# Build no_bh TZXLs
tzxl_bi_type = "down"
bh_direction = "up"

def build_tzxls(bis, start, mode):
    tzxl_bis = [b for b in bis[start:] if b.type == "down"]
    tzxls = []
    for bi in tzxl_bis:
        pre_line = bis[bi.index - 1] if bi.index > 0 else bi
        new_tzxl = TZXL(
            bh_direction="up",
            line=bi,
            pre_line=pre_line,
            line_bad=False,
            done=bi.is_done(),
        )
        if len(tzxls) == 0:
            tzxls.append(new_tzxl)
            continue
        
        last_tzxl = tzxls[-1]
        old_contains_new = last_tzxl.max >= new_tzxl.max and last_tzxl.min <= new_tzxl.min
        new_contains_old = new_tzxl.max >= last_tzxl.max and new_tzxl.min <= last_tzxl.min
        
        if old_contains_new:
            last_tzxl.lines.append(bi)
            last_tzxl.done = bi.is_done()
            last_tzxl.line_bad = False
            last_tzxl.update_maxmin()
        elif new_contains_old:
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
    tzxls = build_tzxls(bis, 3, mode)
    print(f"\n  {mode} TZXLs ({len(tzxls)}):")
    for i, t in enumerate(tzxls[:12]):
        lines = [l.index for l in t.lines]
        print(f"    [{i}] max={t.max:.1f} min={t.min:.1f} bad={t.line_bad} lines={lines}")
    
    # Find DING FX
    print(f"\n  DING FX search in {mode}:")
    for i in range(1, min(len(tzxls) - 1, 10)):
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
            print(f"    FX at TZXL[{i}] lines={lines} max={curr.max:.1f} bad={curr.line_bad} → end={end_bi_idx}")
