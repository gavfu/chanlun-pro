"""Debug BTC60 TZXL for down bi[28->44] vs pyarmor down bi[28->38]"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O

CL_CONFIG = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
}

# Monkey-patch _find_xd_end to add tracing for segments starting at bi[28]
original_find_xd_end = CL_O._find_xd_end

def debug_find_xd_end(self, bis, start_bi_idx, xd_type):
    if start_bi_idx == 28:
        print(f"\n=== _find_xd_end: {xd_type} from bi[{start_bi_idx}] ===")
    result = original_find_xd_end(self, bis, start_bi_idx, xd_type)
    if start_bi_idx == 28:
        if result:
            end_bi_idx, ding_fx, di_fx, tzxls = result
            print(f"  RESULT: end_bi_idx={end_bi_idx}")
            print(f"  TZXL count: {len(tzxls)}")
            for i, xl in enumerate(tzxls):
                print(f"    TZXL[{i}]: max={xl.max:.1f} min={xl.min:.1f} bad={xl.line_bad} lines=[{','.join(str(l.index) for l in xl.lines)}]")
            if di_fx:
                print(f"  DI_FX: type={di_fx.type} bad={di_fx.is_line_bad} xl_min={di_fx.xl.min:.1f}")
        else:
            print(f"  RESULT: None")
    return result

CL_O._find_xd_end = debug_find_xd_end

df = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cd = CL_O("test", "test", config=CL_CONFIG)
cd.process_klines(df)

xds = cd.get_xds()
print("\n=== Relevant XDs ===")
for i, xd in enumerate(xds):
    si = xd.start_line.index
    ei = xd.end_line.index
    if si >= 13 and si <= 45:
        print(f"  xd[{i}] {xd.type:>4s} bi[{si}->{ei}] done={xd.done} split=[{xd.is_split}]")
