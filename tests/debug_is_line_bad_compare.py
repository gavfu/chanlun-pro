"""
Compare TRUE vs FALSE is_line_bad cases: what's ACTUALLY different?

We have pairs of identical-looking XLFX structures (both neighbors not-bad, 
middle is bad) but one gets is_line_bad=True and the other False.

Let's compare EVERYTHING about these paired cases.
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

def dump_fx_detail(cl, bis, start_idx, fx_type, label):
    """Dump full XLFX detail for EVERY FX returned."""
    lines = bis[start_idx:]
    result = cl._xd_cal_line_xlfx(lines, fx_type=fx_type, bh_type='no_bh')
    tzxls, xlfxs = result
    
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"  start={start_idx}, fx_type={fx_type}, {len(tzxls)} TZXLs, {len(xlfxs)} XLFXs")
    print(f"{'='*60}")
    
    # Show TZXLs with containment direction info
    print(f"\n  TZXLs:")
    for i, tz in enumerate(tzxls):
        bi_indices = [l.index for l in tz.lines]
        # Determine containment: 
        # - nlines=1, bad=False: standalone
        # - nlines>1, bad=False: OLD⊃NEW merge
        # - nlines=1, bad=True: NEW⊃OLD (result kept separate)
        if tz.line_bad:
            ctype = "NEW⊃OLD"
        elif len(tz.lines) > 1:
            ctype = "OLD⊃NEW_merged"
        else:
            ctype = "standalone"
            
        print(f"    [{i}] bis={bi_indices} bad={tz.line_bad} max={tz.max:.1f} min={tz.min:.1f} "
              f"type={ctype}")
        for l in tz.lines:
            print(f"          bi[{l.index}] {l.type} h={l.high:.1f} l={l.low:.1f}")
    
    # Show XLFXs
    print(f"\n  XLFXs:")
    for i, fx in enumerate(xlfxs):
        xl_bis = [l.index for l in fx.xl.lines]
        
        # Key: what's the relationship between the FX element and its neighbors?
        # For ding FX: xl.max > left.max AND xl.max > right.max
        # For di FX: xl.min < left.min AND xl.min < right.min
        left = fx.xls[0]
        mid = fx.xls[1]  # = fx.xl
        right = fx.xls[2]
        
        if fx.type == 'ding':
            # ding: max is key
            left_rel = f"mid.max({mid.max:.1f}) > left.max({left.max:.1f})" if mid.max > left.max else f"mid.max({mid.max:.1f}) <= left.max({left.max:.1f})"
            right_rel = f"mid.max({mid.max:.1f}) > right.max({right.max:.1f})" if mid.max > right.max else f"mid.max({mid.max:.1f}) <= right.max({right.max:.1f})"
        else:
            # di: min is key
            left_rel = f"mid.min({mid.min:.1f}) < left.min({left.min:.1f})" if mid.min < left.min else f"mid.min({mid.min:.1f}) >= left.min({left.min:.1f})"
            right_rel = f"mid.min({mid.min:.1f}) < right.min({right.min:.1f})" if mid.min < right.min else f"mid.min({mid.min:.1f}) >= right.min({right.min:.1f})"
        
        marker = " *** FALSE ***" if (fx.xl.line_bad and not fx.is_line_bad) else ""
        marker2 = " *** TRUE(xl_not_bad) ***" if (not fx.xl.line_bad and fx.is_line_bad) else ""
        
        print(f"    [{i}] type={fx.type} FX@bi[{xl_bis}] is_line_bad={fx.is_line_bad} "
              f"xl.line_bad={fx.xl.line_bad}{marker}{marker2}")
        print(f"         {left_rel}")
        print(f"         {right_rel}")
        
        for j, x in enumerate(fx.xls):
            x_bis = [l.index for l in x.lines]
            print(f"         xls[{j}]: bis={x_bis} bad={x.line_bad} "
                  f"max={x.max:.1f} min={x.min:.1f}")

# ===== BTC60 =====
df60 = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cl60 = CL_Pyarmor("test", "60m", config)
cl60.process_klines(df60)
bis60 = cl60.get_bis()

# FALSE case: BTC60 FX@bi[39] di from start=28
dump_fx_detail(cl60, bis60, 28, 'di', 'BTC60 di from bi[28] (FX@bi[39] is_line_bad=FALSE)')

# TRUE case: BTC60 FX@bi[42] ding from start=39  
dump_fx_detail(cl60, bis60, 39, 'ding', 'BTC60 ding from bi[39] (FX@bi[42] is_line_bad=TRUE)')

# FALSE case: BTC60 FX@bi[25] di from start=0
dump_fx_detail(cl60, bis60, 0, 'di', 'BTC60 di from bi[0] (FX@bi[25] is_line_bad=FALSE)')

# Let's also check: same FX type and direction but different start
# FX@bi[42] ding but starting earlier
dump_fx_detail(cl60, bis60, 13, 'ding', 'BTC60 ding from bi[13] (FX@bi[42] is_line_bad=?)')

# ===== BTC5m =====
df5m = pd.read_parquet("tests/test_data/BTC_USDT_5m_1000.parquet")
cl5m = CL_Pyarmor("test", "5m", config)
cl5m.process_klines(df5m)
bis5m = cl5m.get_bis()

# TRUE case: BTC5m FX@bi[6] ding from start=3
dump_fx_detail(cl5m, bis5m, 3, 'ding', 'BTC5m ding from bi[3] (FX@bi[6] is_line_bad=TRUE)')

# What about BTC5m di starting early where FX@bi[6] would be relevant too?
# Actually bi[6] is a down BI, so for ding (looking at down BIs' maxes)...

# Let me check: what's the relationship between is_line_bad and 
# whether the bad TZXL is more extreme than both neighbors?
print("\n\n" + "="*80)
print("PATTERN SEARCH: is_line_bad vs containment-extremity relationship")
print("="*80)

# For EVERY start and fx_type, check:
# - For each FX with xl.line_bad=True:
#   - Is the bad TZXL more extreme than BOTH its neighbors? (genuine FX)
#   - Or is it only more extreme than ONE neighbor? (artificial FX from containment)
for ds_name, df, cl, bis in [
    ("BTC60", df60, cl60, bis60),
    ("BTC5m", df5m, cl5m, bis5m),
]:
    print(f"\n--- {ds_name} ---")
    seen = set()
    for start in range(0, len(bis) - 3):
        for fx_type in ['ding', 'di']:
            lines = bis[start:]
            result = cl._xd_cal_line_xlfx(lines, fx_type=fx_type, bh_type='no_bh')
            _, xlfxs = result
            for fx in xlfxs:
                if not fx.xl.line_bad:
                    continue
                xl_bis = tuple(l.index for l in fx.xl.lines)
                key = (fx_type, xl_bis)
                if key in seen:
                    continue
                seen.add(key)
                
                left = fx.xls[0]
                mid = fx.xls[1]
                right = fx.xls[2]
                
                if fx_type == 'ding':
                    more_than_left = mid.max > left.max
                    more_than_right = mid.max > right.max
                else:
                    more_than_left = mid.min < left.min
                    more_than_right = mid.min < right.min
                    
                genuine = more_than_left and more_than_right
                
                print(f"  {fx_type} FX@bi[{list(xl_bis)}] "
                      f"is_line_bad={fx.is_line_bad} "
                      f"genuine_fx={genuine} "
                      f"left={more_than_left} right={more_than_right}")
