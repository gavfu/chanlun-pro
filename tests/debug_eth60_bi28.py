"""Detailed trace of ETH60 _find_xd_end(down, 28) to understand why it ends at 30"""
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

def debug(self, bis, start_bi_idx, xd_type):
    if start_bi_idx == 28 and xd_type == "down":
        # Manually build TZXL
        tzxl_bis = [bi for bi in bis[start_bi_idx:] if bi.type == "up"]  # down takes up BIs
        
        print(f"\n=== ETH60: down from bi[{start_bi_idx}] ===")
        print(f"UP BIs ({len(tzxl_bis)}):")
        for bi in tzxl_bis[:15]:
            print(f"  bi[{bi.index}] high={bi.high:.1f} low={bi.low:.1f}")
        
        # Build TZXL with merge
        tzxls = []
        for bi in tzxl_bis:
            pre_line = bis[bi.index - 1] if bi.index > 0 else bi
            new_t = TZXL(bh_direction="down", line=bi, pre_line=pre_line,
                        line_bad=False, done=bi.is_done())
            if not tzxls:
                tzxls.append(new_t)
                print(f"  + bi[{bi.index}] → TZXL[0] max={new_t.max:.1f} min={new_t.min:.1f}")
                continue
            last = tzxls[-1]
            old_new = last.max >= new_t.max and last.min <= new_t.min
            new_old = new_t.max >= last.max and new_t.min <= last.min
            if old_new:
                print(f"  + bi[{bi.index}] max={new_t.max:.1f} min={new_t.min:.1f} → OLD⊃NEW merge[{len(tzxls)-1}]")
                last.lines.append(bi)
                last.done = bi.is_done()
                last.line_bad = False
                last.update_maxmin()
            elif new_old:
                new_t.line_bad = True
                tzxls.append(new_t)
                print(f"  + bi[{bi.index}] → TZXL[{len(tzxls)-1}] max={new_t.max:.1f} min={new_t.min:.1f} bad=True")
            else:
                tzxls.append(new_t)
                print(f"  + bi[{bi.index}] → TZXL[{len(tzxls)-1}] max={new_t.max:.1f} min={new_t.min:.1f}")
        
        print(f"\nFinal TZXL ({len(tzxls)}):")
        for i, xl in enumerate(tzxls):
            print(f"  [{i}]: max={xl.max:.1f} min={xl.min:.1f} bad={xl.line_bad} lines=[{','.join(str(l.index) for l in xl.lines)}]")
        
        # DI FX check
        print(f"\nDI FX (min < prev.min AND min < next.min):")
        for i in range(1, len(tzxls) - 1):
            xl = tzxls[i]
            if xl.min < tzxls[i-1].min and xl.min < tzxls[i+1].min:
                end_bi = min(xl.lines, key=lambda l: l.low)
                ebi = end_bi.index
                if bis[ebi].type == "up" and ebi > 0:
                    ebi -= 1
                print(f"  TZXL[{i}]: min={xl.min:.1f} bad={xl.line_bad} lines=[{','.join(str(l.index) for l in xl.lines)}] → end_bi={ebi}")
                # bi_pohuai check
                pohuai = self._check_xd_bi_pohuai(bis, start_bi_idx, xl, "down")
                print(f"    bi_pohuai={pohuai}")
    
    return original_find_xd_end(self, bis, start_bi_idx, xd_type)

CL_O._find_xd_end = debug

df = pd.read_parquet("tests/test_data/ETH_USDT_60m_1000.parquet")
cd = CL_O("test", "test", config=CL_CONFIG)
cd.process_klines(df)

CL_O._find_xd_end = original_find_xd_end
