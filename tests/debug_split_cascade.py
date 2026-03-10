"""
Verify: if BTC60 down[28→38] is fixed, what happens to the cascade?
- up[39→45] should be split by 段内不同向中枢 or 笔破坏
- Verify our split logic handles this correctly
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_interface import TZXL, BI, XD, XLFX

config = {
    "bi_type": "bi_type_old", "fx_qj": "fx_qj_k", "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes", "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1", "xd_bzh": "xd_bzh_no",
    "xd_zs_max_lines_split": 11, "xd_allow_split_no_highlow": 1,
    "xd_allow_split_zs_kz": 0, "xd_allow_split_zs_more_line": 1,
    "xd_allow_split_zs_no_direction": 1,
}

df60 = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cl_o = CL_Open("test", "60m", config)
cl_o.process_klines(df60)
bis_o = cl_o.get_bis()

# Simulate up[39→45]
print("=== Simulating up[39→45] split check ===")
start_idx = 39
end_idx = 45
bi_count = end_idx - start_idx + 1
print(f"  Segment: up[{start_idx}→{end_idx}], {bi_count} BIs")

# Show BIs
for i in range(start_idx, end_idx + 1):
    bi = bis_o[i]
    print(f"    bi[{i}]: {bi.type} h={bi.high:.1f} l={bi.low:.1f}")

# Build ZS
zs_list = cl_o._build_zs_in_range(bis_o, start_idx, end_idx)
print(f"\n  Pivot zones ({len(zs_list)}):")
for zs in zs_list:
    print(f"    {zs['type']} bi[{zs['start_bi']}→{zs['end_bi']}] "
          f"zg={zs['zg']:.1f} zd={zs['zd']:.1f} lines={zs['line_num']}")

# Check same/opp direction
same_dir = [zs for zs in zs_list if zs['type'] == 'up']
opp_dir = [zs for zs in zs_list if zs['type'] != 'up']
print(f"\n  same_dir ZS: {len(same_dir)}, opp_dir ZS: {len(opp_dir)}")

# Check bi-pohuai
has_pohuai = cl_o._has_bi_pohuai_in_range(bis_o, start_idx, end_idx, "up")
print(f"  has_bi_pohuai: {has_pohuai}")

# Check condition 2 for split
print(f"\n  Split condition 2 check:")
print(f"    xd_allow_split_zs_no_direction: {cl_o.xd_allow_split_zs_no_direction}")
print(f"    opp_dir_zs exists: {bool(opp_dir)}")
print(f"    no same_dir_zs: {not same_dir}")
print(f"    bi_count >= 7: {bi_count >= 7}")

if opp_dir:
    for zs in opp_dir:
        covers_start = zs['start_list_idx'] <= start_idx
        covers_end = zs['end_list_idx'] >= end_idx
        not_covering = zs['start_list_idx'] > start_idx or zs['end_list_idx'] < end_idx
        print(f"    ZS covers: start={covers_start} end={covers_end} not_covering={not_covering}")

# Now manually create the XD and test split
print("\n\n=== Test actual _check_xd_split ===")
xd = cl_o._create_split_xd(bis_o, start_idx, end_idx, "up", 0, "test")
if xd:
    xd.done = True
    xd.is_split = None  # Not a split result
    splits = cl_o._check_xd_split(xd, bis_o)
    if splits:
        print(f"  Split result: {len(splits)} segments")
        for s in splits:
            print(f"    {s.type} [{s.start_line.index}→{s.end_line.index}] ({s.is_split})")
    else:
        print(f"  No split triggered!")
        
    # Also test pure bi-pohuai split
    bi_split = cl_o._split_by_bi_pohuai(xd, bis_o)
    if bi_split:
        print(f"\n  Pure bi-pohuai split: {len(bi_split)} segments")
        for s in bi_split:
            print(f"    {s.type} [{s.start_line.index}→{s.end_line.index}]")
    else:
        print(f"\n  No pure bi-pohuai split")
else:
    print("  Failed to create XD!")
