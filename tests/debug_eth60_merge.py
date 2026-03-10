"""Trace ETH60 TZXL merge for up from bi[31]"""
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
    if start_bi_idx == 31:
        # Manually trace TZXL construction
        tzxl_bi_type = "down" if xd_type == "up" else "up"
        bh_direction = "up" if xd_type == "up" else "down"
        
        tzxl_bis = []
        for i in range(start_bi_idx, len(bis)):
            if bis[i].type == tzxl_bi_type:
                tzxl_bis.append(bis[i])
        
        print(f"\n=== ETH60: _find_xd_end {xd_type} from bi[{start_bi_idx}] ===")
        print(f"Raw DOWN BIs ({len(tzxl_bis)}):")
        for bi in tzxl_bis[:12]:
            print(f"  bi[{bi.index}] high={bi.high:.1f} low={bi.low:.1f}")
        
        print(f"\nMerge trace (bh_direction={bh_direction}):")
        tzxls = []
        for bi in tzxl_bis:
            pre_line = bis[bi.index - 1] if bi.index > 0 else bi
            new_tzxl = TZXL(
                bh_direction=bh_direction, line=bi, pre_line=pre_line,
                line_bad=False, done=bi.is_done()
            )
            if len(tzxls) == 0:
                tzxls.append(new_tzxl)
                print(f"  + bi[{bi.index}] → TZXL[0] max={new_tzxl.max:.1f} min={new_tzxl.min:.1f}")
                continue
            
            last = tzxls[-1]
            old_new = last.max >= new_tzxl.max and last.min <= new_tzxl.min
            new_old = new_tzxl.max >= last.max and new_tzxl.min <= last.min
            
            if old_new:
                print(f"  + bi[{bi.index}] max={new_tzxl.max:.1f} min={new_tzxl.min:.1f} → OLD⊃NEW merge into TZXL[{len(tzxls)-1}] (was bad={last.line_bad})")
                last.lines.append(bi)
                last.done = bi.is_done()
                last.line_bad = False
                last.update_maxmin()
            elif new_old:
                print(f"  + bi[{bi.index}] max={new_tzxl.max:.1f} min={new_tzxl.min:.1f} → NEW⊃OLD, TZXL[{len(tzxls)}] bad=True")
                new_tzxl.line_bad = True
                tzxls.append(new_tzxl)
            else:
                print(f"  + bi[{bi.index}] → TZXL[{len(tzxls)}] max={new_tzxl.max:.1f} min={new_tzxl.min:.1f}")
                tzxls.append(new_tzxl)
        
        print(f"\nFinal TZXL list:")
        for i, xl in enumerate(tzxls):
            print(f"  TZXL[{i}]: max={xl.max:.1f} min={xl.min:.1f} bad={xl.line_bad} lines=[{','.join(str(l.index) for l in xl.lines)}]")
    
    return original_find_xd_end(self, bis, start_bi_idx, xd_type)

CL_O._find_xd_end = debug_find_xd_end

df = pd.read_parquet("tests/test_data/ETH_USDT_60m_1000.parquet")
cd = CL_O("test", "test", config=CL_CONFIG)
cd.process_klines(df)
