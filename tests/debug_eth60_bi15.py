"""Check ETH60 up from bi[15] TZXL/FX to understand the natural segment"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
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

original_find_xd_end = CL_O._find_xd_end

def debug_find_xd_end(self, bis, start_bi_idx, xd_type):
    if start_bi_idx == 15:
        # Trace
        tzxl_bi_type = "down"  # up segment takes down BIs
        bh_direction = "up"
        
        tzxl_bis = [bi for bi in bis[start_bi_idx:] if bi.type == tzxl_bi_type]
        
        print(f"\n=== ETH60: up from bi[{start_bi_idx}] ===")
        
        # Build TZXL
        tzxls = []
        for bi in tzxl_bis:
            pre_line = bis[bi.index - 1] if bi.index > 0 else bi
            new_t = TZXL(bh_direction=bh_direction, line=bi, pre_line=pre_line,
                        line_bad=False, done=bi.is_done())
            if not tzxls:
                tzxls.append(new_t)
                continue
            last = tzxls[-1]
            old_new = last.max >= new_t.max and last.min <= new_t.min
            new_old = new_t.max >= last.max and new_t.min <= last.min
            if old_new:
                last.lines.append(bi)
                last.done = bi.is_done()
                last.line_bad = False
                last.update_maxmin()
            elif new_old:
                new_t.line_bad = True
                tzxls.append(new_t)
            else:
                tzxls.append(new_t)
        
        print(f"TZXL list ({len(tzxls)}):")
        for i, xl in enumerate(tzxls):
            print(f"  [{i}]: max={xl.max:.1f} min={xl.min:.1f} bad={xl.line_bad} lines=[{','.join(str(l.index) for l in xl.lines)}]")
        
        # Check DING FX (for UP segment)
        print(f"\nDING FX candidates:")
        for i in range(1, len(tzxls) - 1):
            xl = tzxls[i]
            if xl.max > tzxls[i-1].max and xl.max > tzxls[i+1].max:
                print(f"  TZXL[{i}]: max={xl.max:.1f} bad={xl.line_bad} lines=[{','.join(str(l.index) for l in xl.lines)}]")
                # What end_bi_idx would this give?
                end_bi = max(xl.lines, key=lambda l: l.high)
                ebi = end_bi.index
                if bis[ebi].type == "down" and ebi > 0:
                    ebi -= 1
                print(f"    → end_bi_idx={ebi}")
    
    return original_find_xd_end(self, bis, start_bi_idx, xd_type)

CL_O._find_xd_end = debug_find_xd_end

df = pd.read_parquet("tests/test_data/ETH_USDT_60m_1000.parquet")
cd = CL_O("test", "test", config=CL_CONFIG)
cd.process_klines(df)

print("\n=== Result ===")
xds = cd.get_xds()
for i, xd in enumerate(xds[:6]):
    si = xd.start_line.index
    ei = xd.end_line.index
    print(f"  xd[{i}] {xd.type:>4s} bi[{si}->{ei}] split=[{xd.is_split}]")
