"""
Deep investigation: what property of a TZXL determines is_line_bad in pyarmor?

We know:
- BTC60 down FX@bi[39]: xl.line_bad=True, is_line_bad=False
- BTC60 up FX@bi[42]: xl.line_bad=True, is_line_bad=True
- BTC5m up FX@bi[6]: xl.line_bad=True, is_line_bad=True  
- BTC5m up FX@bi[10]: xl.line_bad=False, is_line_bad=True

Let's dump the FULL TZXL construction details: which BIs are merged, 
containment direction (OLD⊃NEW vs NEW⊃OLD), and the actual values.

For the "bad" TZXLs, the bad flag comes from NEW⊃OLD containment.
Let's check if the containment direction matters for is_line_bad.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_pyarmor import CL as CL_Pyarmor

config = {
    "bi_type": "bi_type_old", "fx_qj": "fx_qj_k", "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes", "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1", "xd_bzh": "xd_bzh_no",
    "xd_zs_max_lines_split": 11, "xd_allow_split_no_highlow": 1,
    "xd_allow_split_zs_kz": 0, "xd_allow_split_zs_more_line": 1,
    "xd_allow_split_zs_no_direction": 1,
}

def analyze_tzxls(cl, bis, lines, fx_type, label):
    """Analyze TZXL construction and XLFX is_line_bad for given lines."""
    result = cl._xd_cal_line_xlfx(lines, fx_type=fx_type, bh_type='no_bh')
    tzxls, xlfxs = result
    
    print(f"\n--- {label}: {fx_type} mode, {len(lines)} lines starting at bi[{lines[0].index}] ---")
    
    # Show ALL TZXLs
    print(f"  TZXLs ({len(tzxls)}):")
    for i, tz in enumerate(tzxls):
        bi_indices = [l.index for l in tz.lines]
        print(f"    [{i}] bis={bi_indices} bad={tz.line_bad} max={tz.max:.1f} min={tz.min:.1f} "
              f"done={tz.done} nlines={len(tz.lines)}")
        # Show individual BI details
        for l in tz.lines:
            print(f"         bi[{l.index}] {l.type} high={l.high:.1f} low={l.low:.1f}")
    
    # Show ALL XLFXs
    print(f"  XLFXs ({len(xlfxs)}):")
    for i, fx in enumerate(xlfxs):
        xl_bis = [l.index for l in fx.xl.lines]
        xls_bis = [[l.index for l in x.lines] for x in fx.xls]
        print(f"    [{i}] type={fx.type} FX_xl_bis={xl_bis} is_line_bad={fx.is_line_bad} "
              f"xl.line_bad={fx.xl.line_bad}")
        print(f"         xls_bis={xls_bis}")
        print(f"         fx_high={fx.fx_high:.1f} fx_low={fx.fx_low:.1f} done={fx.done} qk={fx.qk}")
        for j, x in enumerate(fx.xls):
            x_bis = [l.index for l in x.lines]
            print(f"         xls[{j}]: bis={x_bis} bad={x.line_bad} max={x.max:.1f} min={x.min:.1f}")

# ===== BTC60 =====
df60 = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cl60 = CL_Pyarmor("test", "60m", config)
cl60.process_klines(df60)
bis60 = cl60.get_bis()

print("="*80)
print("BTC60")
print("="*80)

# Case 1: BTC60 down[28], FX@bi[39] → is_line_bad=False
# down segment: look for di FX among down BIs starting from bi[28]
analyze_tzxls(cl60, bis60, bis60[28:], 'di', 'BTC60 down from bi[28]')

# Case 2: BTC60 up[39], FX@bi[42] → is_line_bad=True 
analyze_tzxls(cl60, bis60, bis60[39:], 'ding', 'BTC60 up from bi[39]')

# Now try with fewer lines - what happens incrementally?
print("\n\n" + "="*80)
print("BTC60: INCREMENTAL di from bi[28] (adding lines)")
print("="*80)
for n in range(3, min(20, len(bis60) - 28)):
    lines = bis60[28:28+n]
    result = cl60._xd_cal_line_xlfx(lines, fx_type='di', bh_type='no_bh')
    tzxls, xlfxs = result
    bad_fxs = [(fx, [l.index for l in fx.xl.lines]) for fx in xlfxs if fx.xl.line_bad]
    if bad_fxs:
        for fx, xl_bis in bad_fxs:
            print(f"  n={n} lines[28..{27+n}]: FX@bi[{xl_bis}] is_line_bad={fx.is_line_bad} "
                  f"xl_bad={fx.xl.line_bad} nlines_in_xl={len(fx.xl.lines)}")


# ===== BTC5m =====
df5m = pd.read_parquet("tests/test_data/BTC_USDT_5m_1000.parquet") 
cl5m = CL_Pyarmor("test", "5m", config)
cl5m.process_klines(df5m)
bis5m = cl5m.get_bis()

print("\n\n" + "="*80)
print("BTC5m")
print("="*80)

# Case 3: BTC5m up[3], FX@bi[6] → is_line_bad=True, xl.line_bad=True
# Case 4: BTC5m up[3], FX@bi[10] → is_line_bad=True, xl.line_bad=False
analyze_tzxls(cl5m, bis5m, bis5m[3:], 'ding', 'BTC5m up from bi[3]')

# ===== ETH60 =====
dfeth = pd.read_parquet("tests/test_data/ETH_USDT_60m_1000.parquet")
cleth = CL_Pyarmor("test", "60m", config)
cleth.process_klines(dfeth)
biseth = cleth.get_bis()

print("\n\n" + "="*80)
print("ETH60")
print("="*80)

# ETH60 up[15], FX@bi[20] → is_line_bad=True
analyze_tzxls(cleth, biseth, biseth[15:], 'ding', 'ETH60 up from bi[15]')

# Now let's look at cases where is_line_bad=False more broadly
print("\n\n" + "="*80)
print("ALL is_line_bad=False cases with xl.line_bad=True (BTC60)")
print("="*80)

for start in range(0, len(bis60) - 3):
    for fx_type in ['ding', 'di']:
        lines = bis60[start:]
        result = cl60._xd_cal_line_xlfx(lines, fx_type=fx_type, bh_type='no_bh')
        _, xlfxs = result
        for fx in xlfxs:
            if fx.xl.line_bad and not fx.is_line_bad:
                xl_bis = [l.index for l in fx.xl.lines]
                xls_info = []
                for x in fx.xls:
                    x_bis = [l.index for l in x.lines]
                    xls_info.append(f"bis={x_bis} bad={x.line_bad}")
                
                # Check: is the bad TZXL the LAST in xls?
                # Check: is there a non-bad TZXL after it?
                xl_pos = None
                for j, x in enumerate(fx.xls):
                    if any(l.index == fx.xl.lines[0].index for l in x.lines):
                        xl_pos = j
                        break
                
                print(f"  start={start} {fx_type}: FX@bi[{xl_bis}] is_line_bad=FALSE "
                      f"xl_pos_in_xls={xl_pos}/{len(fx.xls)}")
                print(f"    xls: {xls_info}")
                break  # first match per start/type combo
