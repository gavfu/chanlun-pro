"""Trace ETH60 down from bi[34] TZXL to understand xd[6]"""
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

# Two segment starts are relevant: where xd[6] starts
# With "more extreme": xd[6] = down bi[34->40]
# With "first FX": xd[6] = down bi[34->36]
# Both after split from up bi[31->33/41]

# In the current code, xd[6] comes from split of the natural segment.
# But what matters is the segment that STARTS at bi[34] as a new DOWN segment.
# After xd[5] ends (at bi[33] after split), the next segment starts at bi[34] as DOWN.

# Trace down from bi[34]
def debug(self, bis, start_bi_idx, xd_type):
    if start_bi_idx == 34:
        tzxl_bi_type = "up"  # down segment takes up BIs
        bh_direction = "down"
        
        tzxl_bis = [bi for bi in bis[start_bi_idx:] if bi.type == tzxl_bi_type]
        
        print(f"\n=== ETH60: down from bi[{start_bi_idx}] ===")
        
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
                was_bad = last.line_bad
                last.lines.append(bi)
                last.done = bi.is_done()
                last.line_bad = False
                last.update_maxmin()
                print(f"  bi[{bi.index}] max={new_t.max:.1f} min={new_t.min:.1f} → OLD⊃NEW merge[{len(tzxls)-1}] (was_bad={was_bad})")
            elif new_old:
                new_t.line_bad = True
                tzxls.append(new_t)
                print(f"  bi[{bi.index}] → TZXL[{len(tzxls)-1}] max={new_t.max:.1f} min={new_t.min:.1f} bad=True (NEW⊃OLD)")
            else:
                tzxls.append(new_t)
                print(f"  bi[{bi.index}] → TZXL[{len(tzxls)-1}] max={new_t.max:.1f} min={new_t.min:.1f}")
        
        print(f"\nFinal TZXL ({len(tzxls)}):")
        for i, xl in enumerate(tzxls):
            print(f"  [{i}]: max={xl.max:.1f} min={xl.min:.1f} bad={xl.line_bad} lines=[{','.join(str(l.index) for l in xl.lines)}]")
        
        # DI FX check
        print(f"\nDI FX candidates (min < prev.min AND min < next.min):")
        for i in range(1, len(tzxls) - 1):
            xl = tzxls[i]
            if xl.min < tzxls[i-1].min and xl.min < tzxls[i+1].min:
                end_bi = min(xl.lines, key=lambda l: l.low)
                ebi = end_bi.index
                if bis[ebi].type == "up" and ebi > 0:
                    ebi -= 1
                print(f"  TZXL[{i}]: min={xl.min:.1f} bad={xl.line_bad} lines=[{','.join(str(l.index) for l in xl.lines)}] → end={ebi}")
                # Check bi_pohuai
                pohuai = self._check_xd_bi_pohuai(bis, start_bi_idx, xl, "down")
                print(f"    bi_pohuai={pohuai}")
    
    return original_find_xd_end(self, bis, start_bi_idx, xd_type)

CL_O._find_xd_end = debug

df = pd.read_parquet("tests/test_data/ETH_USDT_60m_1000.parquet")
cd = CL_O("test", "test", config=CL_CONFIG)
cd.process_klines(df)

# Also check what xd[6] should be with split
xds = cd.get_xds()
print(f"\nActual segments:")
for i, xd in enumerate(xds):
    if i >= 4 and i <= 8:
        print(f"  xd[{i}] {xd.type:>4s} bi[{xd.start_line.index}->{xd.end_line.index}] split=[{xd.is_split}]")
