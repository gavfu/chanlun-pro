"""Trace TZXL merge process for BTC60 (bi[28] down) and BTC5m (bi[46] down) """
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

def trace_tzxl_merge(data_file, target_start_bi, label):
    """Trace the TZXL construction merge ops for a specific segment"""
    original_find_xd_end = CL_O._find_xd_end

    def debug_find_xd_end(self, bis, start_bi_idx, xd_type):
        if start_bi_idx == target_start_bi:
            print(f"\n{'='*70}")
            print(f"  {label}: _find_xd_end {xd_type} from bi[{start_bi_idx}]")
            print(f"{'='*70}")
            
            # Manually trace TZXL construction
            tzxl_bi_type = "down" if xd_type == "up" else "up"
            bh_direction = "up" if xd_type == "up" else "down"
            
            tzxl_bis = []
            for i in range(start_bi_idx, len(bis)):
                if bis[i].type == tzxl_bi_type:
                    tzxl_bis.append(bis[i])
            
            print(f"\n  Raw TZXL BIs (type={tzxl_bi_type}):")
            for bi in tzxl_bis:
                print(f"    bi[{bi.index}] {bi.type} high={bi.high:.1f} low={bi.low:.1f}")
            
            # Replay merge
            print(f"\n  Merge trace (bh_direction={bh_direction}):")
            tzxls = []
            for bi in tzxl_bis:
                pre_line = bis[bi.index - 1] if bi.index > 0 else bi
                new_tzxl = TZXL(
                    bh_direction=bh_direction,
                    line=bi, pre_line=pre_line,
                    line_bad=False, done=bi.is_done()
                )
                
                if len(tzxls) == 0:
                    tzxls.append(new_tzxl)
                    print(f"    + bi[{bi.index}] → TZXL[0] max={new_tzxl.max:.1f} min={new_tzxl.min:.1f}")
                    continue
                
                last_tzxl = tzxls[-1]
                old_contains_new = last_tzxl.max >= new_tzxl.max and last_tzxl.min <= new_tzxl.min
                new_contains_old = new_tzxl.max >= last_tzxl.max and new_tzxl.min <= last_tzxl.min
                
                last_lines = [l.index for l in last_tzxl.lines]
                
                if old_contains_new:
                    print(f"    + bi[{bi.index}] max={new_tzxl.max:.1f} min={new_tzxl.min:.1f}")
                    print(f"      OLD⊃NEW: merge into TZXL[{len(tzxls)-1}] (lines={last_lines})")
                    print(f"        OLD was bad={last_tzxl.line_bad} → reset to False")
                    last_tzxl.lines.append(bi)
                    last_tzxl.done = bi.is_done()
                    last_tzxl.line_bad = False
                    last_tzxl.update_maxmin()
                    print(f"        After merge: max={last_tzxl.max:.1f} min={last_tzxl.min:.1f} lines={[l.index for l in last_tzxl.lines]}")
                elif new_contains_old:
                    print(f"    + bi[{bi.index}] max={new_tzxl.max:.1f} min={new_tzxl.min:.1f}")
                    print(f"      NEW⊃OLD: new TZXL[{len(tzxls)}] bad=True (last was TZXL[{len(tzxls)-1}] lines={last_lines})")
                    new_tzxl.line_bad = True
                    tzxls.append(new_tzxl)
                else:
                    print(f"    + bi[{bi.index}] → TZXL[{len(tzxls)}] max={new_tzxl.max:.1f} min={new_tzxl.min:.1f}")
                    tzxls.append(new_tzxl)
            
            print(f"\n  Final TZXL list:")
            for i, xl in enumerate(tzxls):
                print(f"    TZXL[{i}]: max={xl.max:.1f} min={xl.min:.1f} bad={xl.line_bad} lines=[{','.join(str(l.index) for l in xl.lines)}]")
        
        return original_find_xd_end(self, bis, start_bi_idx, xd_type)

    CL_O._find_xd_end = debug_find_xd_end
    
    df = pd.read_parquet(data_file)
    cd = CL_O("test", "test", config=CL_CONFIG)
    cd.process_klines(df)
    
    CL_O._find_xd_end = original_find_xd_end
    return cd

# BTC60 - down from bi[28]
trace_tzxl_merge("tests/test_data/BTC_USDT_60m_1000.parquet", 28, "BTC60")

# BTC5m - down from bi[46]
trace_tzxl_merge("tests/test_data/BTC_USDT_5m_1000.parquet", 46, "BTC5m")
