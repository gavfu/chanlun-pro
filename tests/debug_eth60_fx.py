"""Check ETH60 TZXL FX for segments where 'always first FX' broke things"""
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

# Monkey-patch to trace ALL FX selections
original_find_xd_end = CL_O._find_xd_end

def debug_find_xd_end(self, bis, start_bi_idx, xd_type):
    result = original_find_xd_end(self, bis, start_bi_idx, xd_type)
    if result:
        end_bi_idx, ding_fx, di_fx, tzxls = result
        target_fx = ding_fx if xd_type == "up" else di_fx
        if target_fx and target_fx.is_line_bad:
            print(f"  _find_xd_end({xd_type} from bi[{start_bi_idx}]) → end={end_bi_idx}, FX_BAD=True")
            # Show bad TZXL and any non-bad alternatives
            target_fx_type = "ding" if xd_type == "up" else "di"
            for i in range(1, len(tzxls) - 1):
                xl = tzxls[i]
                prev_xl = tzxls[i - 1]
                next_xl = tzxls[i + 1]
                if target_fx_type == "ding":
                    is_fx = xl.max > prev_xl.max and xl.max > next_xl.max
                else:
                    is_fx = xl.min < prev_xl.min and xl.min < next_xl.min
                if is_fx:
                    extreme = xl.max if target_fx_type == "ding" else xl.min
                    print(f"    TZXL[{i}]: bad={xl.line_bad} extreme={extreme:.1f} lines=[{','.join(str(l.index) for l in xl.lines)}]")
    return result

CL_O._find_xd_end = debug_find_xd_end

print("=== ETH60 ===")
df = pd.read_parquet("tests/test_data/ETH_USDT_60m_1000.parquet")
cd = CL_O("test", "test", config=CL_CONFIG)
cd.process_klines(df)

xds = cd.get_xds()
print("\nETH60 Segments:")
for i, xd in enumerate(xds):
    si = xd.start_line.index
    ei = xd.end_line.index
    print(f"  xd[{i}] {xd.type:>4s} bi[{si}->{ei}] done={xd.done} split=[{xd.is_split}]")

CL_O._find_xd_end = original_find_xd_end

print("\n=== ETH60 Pyarmor ===")
from chanlun.cl import CL as CL_P
cd_p = CL_P("test", "test", config=CL_CONFIG)
cd_p.process_klines(df)
xds_p = cd_p.get_xds()
for i, xd in enumerate(xds_p):
    si = xd.start_line.index
    ei = xd.end_line.index
    print(f"  xd[{i}] {xd.type:>4s} bi[{si}->{ei}] done={xd.done} split=[{xd.is_split}]")
