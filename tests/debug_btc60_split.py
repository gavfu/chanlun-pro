"""Debug BTC60 split issue - xd[1] regression"""
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

df = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cd = CL_O("test", "test", config=CL_CONFIG)
cd.process_klines(df)

bis = cd.get_bis()
xds = cd.get_xds()

print("=== BTC60 Open Segments ===")
for i, xd in enumerate(xds):
    print(f"  xd[{i}] {xd.type:>4s} bi[{xd.start_line.index}->{xd.end_line.index}] done={xd.done} split=[{xd.is_split}]")

# xd[4] = down bi[28->44], needs to be down bi[28->38]
for xd in xds:
    si = xd.start_line.index
    ei = xd.end_line.index
    if si == 28:
        print(f"\n=== Segment {xd.type} bi[{si}->{ei}] ===")
        bi_count = ei - si + 1
        print(f"  bi_count={bi_count}, done={xd.done}")
        
        # Build ZS
        zs_list = cd._build_zs_in_range(bis, si, ei)
        for zs in zs_list:
            print(f"  ZS: bi[{zs['start_list_idx']}->{zs['end_list_idx']}] type={zs['type']} line_num={zs['line_num']}")
        
        same_dir = [z for z in zs_list if z['type'] == xd.type]
        opp_dir = [z for z in zs_list if z['type'] != xd.type]
        print(f"  same_dir_zs: {len(same_dir)}, opp_dir_zs: {len(opp_dir)}")
        
        # Show BI data
        print(f"\n  BIs in segment:")
        for i in range(si, min(ei+1, len(bis))):
            b = bis[i]
            print(f"    bi[{b.index}] {b.type:>4s} high={b.high:.1f} low={b.low:.1f}")
        
        # Check extreme point
        extreme_down_idx = None
        extreme_down_val = None
        for i in range(si, ei+1):
            b = bis[i]
            if b.type == "down":
                if extreme_down_val is None or b.low < extreme_down_val:
                    extreme_down_val = b.low
                    extreme_down_idx = i
        print(f"\n  Extreme DOWN bi: bi[{extreme_down_idx}] low={extreme_down_val}")
        
        # Check BI pohuai (UP going higher - same as _find_bi_pohuai_split)
        seg_bis = bis[si:ei+1]
        for i in range(2, len(seg_bis)):
            b = seg_bis[i]
            if b.type == "up":
                prev_same = None
                for k in range(i-2, -1, -1):
                    if seg_bis[k].type == "up":
                        prev_same = seg_bis[k]
                        break
                if prev_same:
                    pohuai = b.high > prev_same.high
                    print(f"    bi[{b.index}] UP high={b.high:.1f} vs prev bi[{prev_same.index}] high={prev_same.high:.1f} {'** POHUAI **' if pohuai else 'no'}")

# Also show pyarmor segments for reference
cd_p = CL_P("test", "test", config=CL_CONFIG)
cd_p.process_klines(df)
print("\n=== BTC60 Pyarmor Segments ===")
for xd in cd_p.get_xds():
    si = xd.start_line.index
    ei = xd.end_line.index
    print(f"  xd {xd.type:>4s} bi[{si}->{ei}] done={xd.done} split=[{xd.is_split}]")

bis = cd.get_bis()
xds = cd.get_xds()

print("=== BTC60 Open Segments ===")
for i, xd in enumerate(xds):
    print(f"  xd[{i}] {xd.type:>4s} bi[{xd.start_line.index}->{xd.end_line.index}] done={xd.done} split=[{xd.is_split}]")

# Check the xd that covers bi[13->17] - it should have been split
# Build ZS for that segment
for xd in xds:
    si = xd.start_line.index
    ei = xd.end_line.index
    if si <= 13 <= ei or si <= 17 <= ei:
        print(f"\n=== Segment {xd.type} bi[{si}->{ei}] - ZS analysis ===")
        bi_count = ei - si + 1
        print(f"  bi_count={bi_count}, done={xd.done}")
        
        # Build ZS
        zs_list = cd._build_zs_in_range(bis, si, ei)
        for zs in zs_list:
            print(f"  ZS: bi[{zs['start_list_idx']}->{zs['end_list_idx']}] type={zs['type']} line_num={zs['line_num']}")
        
        # Check split conditions
        same_dir = [z for z in zs_list if z['type'] == xd.type]
        opp_dir = [z for z in zs_list if z['type'] != xd.type]
        print(f"  same_dir_zs: {len(same_dir)}")
        print(f"  opp_dir_zs: {len(opp_dir)}")
        
        for zs in zs_list:
            print(f"  ZS line_num={zs['line_num']} >= threshold 11? {zs['line_num'] >= 11}")
