"""
Compare pyarmor's no_bh FX results vs our code's FX results for ETH60.
Key question: does pyarmor's no_bh mode ignore line_bad and just take first FX?
And if so, what makes the output still match ETH60 correctly?
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_pyarmor import CL

df = pd.read_parquet("tests/test_data/ETH_USDT_60m_1000.parquet")
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

cl = CL("ETH60", "60m", config)
cl.process_klines(df)
bis = cl.get_bis()

# Check ETH60 cases where our code uses the "more extreme" rule
# xd[1] pyarmor: up bi[15→19]  
# Our code with "first FX": up bi[15→17] (wrong)
# Our code with "more extreme": up bi[15→19] (correct!)

# For up bi[15], look for DING FX in DOWN BIs
# bh_direction = "up", TZXL uses DOWN BIs

# Let's check what pyarmor's no_bh gives for different ranges
print("=== ETH60 up bi[15]: DING FX search ===")

# With increasing ranges
for end_idx in [20, 22, 24, 28, 34]:
    subset = bis[15:end_idx]
    print(f"\n  lines[15..{end_idx-1}] ({len(subset)} BIs):")
    
    # bh mode
    tzxls_bh, xlfxs_bh = cl._xd_cal_line_xlfx(subset, 'ding', 'bh')
    bh_fx = None
    if xlfxs_bh:
        bh_fx = xlfxs_bh[0]
        print(f"    bh: XLFX type={bh_fx.type} bad={bh_fx.is_line_bad} xl_lines={[l.index for l in bh_fx.xl.lines]}")
    else:
        print(f"    bh: No FX found ({len(tzxls_bh)} TZXLs)")
    
    # no_bh mode
    tzxls_no, xlfxs_no = cl._xd_cal_line_xlfx(subset, 'ding', 'no_bh')
    nobh_fx = None
    if xlfxs_no:
        nobh_fx = xlfxs_no[0]
        print(f"    no_bh: XLFX type={nobh_fx.type} bad={nobh_fx.is_line_bad} xl_lines={[l.index for l in nobh_fx.xl.lines]}")
        # Show middle TZXL bad status
        for i, t in enumerate(tzxls_no):
            if t.lines and any(l.index in [ll.index for ll in nobh_fx.xl.lines] for l in t.lines):
                print(f"      middle TZXL[{i}]: bad={t.line_bad} lines={[l.index for l in t.lines]}")
    else:
        print(f"    no_bh: No FX found ({len(tzxls_no)} TZXLs)")

# Also check ETH60 up bi[31] case
print("\n\n=== ETH60 up bi[31]: DING FX search ===")
for end_idx in [36, 38, 40, 42, 44]:
    subset = bis[31:end_idx]
    print(f"\n  lines[31..{end_idx-1}] ({len(subset)} BIs):")
    
    tzxls_bh, xlfxs_bh = cl._xd_cal_line_xlfx(subset, 'ding', 'bh')
    if xlfxs_bh:
        fx = xlfxs_bh[0]
        print(f"    bh: XLFX type={fx.type} bad={fx.is_line_bad} xl_lines={[l.index for l in fx.xl.lines]}")
    else:
        print(f"    bh: No FX ({len(tzxls_bh)} TZXLs)")
        for t in tzxls_bh:
            print(f"      [{tzxls_bh.index(t)}]: max={t.max} min={t.min} lines={[l.index for l in t.lines]}")
    
    tzxls_no, xlfxs_no = cl._xd_cal_line_xlfx(subset, 'ding', 'no_bh')
    if xlfxs_no:
        fx = xlfxs_no[0]
        print(f"    no_bh: XLFX type={fx.type} bad={fx.is_line_bad} xl_lines={[l.index for l in fx.xl.lines]}")
    else:
        print(f"    no_bh: No FX ({len(tzxls_no)} TZXLs)")

# Check ETH60 down bi[34] case 
print("\n\n=== ETH60 down bi[34]: DI FX search ===")
for end_idx in [40, 42, 44]:
    subset = bis[34:end_idx]
    print(f"\n  lines[34..{end_idx-1}] ({len(subset)} BIs):")
    
    tzxls_bh, xlfxs_bh = cl._xd_cal_line_xlfx(subset, 'di', 'bh')
    if xlfxs_bh:
        fx = xlfxs_bh[0]
        print(f"    bh: XLFX type={fx.type} bad={fx.is_line_bad} xl_lines={[l.index for l in fx.xl.lines]}")
    else:
        print(f"    bh: No FX ({len(tzxls_bh)} TZXLs)")
        for t in tzxls_bh:
            print(f"      [{tzxls_bh.index(t)}]: max={t.max} min={t.min} lines={[l.index for l in t.lines]}")
    
    tzxls_no, xlfxs_no = cl._xd_cal_line_xlfx(subset, 'di', 'no_bh')
    if xlfxs_no:
        fx = xlfxs_no[0]
        print(f"    no_bh: XLFX type={fx.type} bad={fx.is_line_bad} xl_lines={[l.index for l in fx.xl.lines]}")
        # find middle TZXL bad status
        for t in tzxls_no:
            if any(l.index in [ll.index for ll in fx.xl.lines] for l in t.lines):
                print(f"      middle TZXL: bad={t.line_bad} lines={[l.index for l in t.lines]}")
    else:
        print(f"    no_bh: No FX ({len(tzxls_no)} TZXLs)")
