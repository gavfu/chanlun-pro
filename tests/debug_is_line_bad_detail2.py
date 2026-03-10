"""
Deep comparison of is_line_bad TRUE vs FALSE cases.

All bad TZXL cases are "genuine" FXs. What differentiates TRUE from FALSE?

Let's compare the actual BI prices to understand the "bad" containment:
- For a bad TZXL: NEW⊃OLD (new BI contains old BI)
- This means: new_bi.high >= old_bi.high AND new_bi.low <= old_bi.low

Hypothesis: maybe is_line_bad depends on whether the bad TZXL's 
containment direction matches or opposes the FX direction.

Another hypothesis: is_line_bad is about whether the TZXL's line
"generates" a new extreme or "maintains" the previous TZXL's extreme.
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

datasets = [
    ("BTC60", "tests/test_data/BTC_USDT_60m_1000.parquet"),
    ("BTC5m", "tests/test_data/BTC_USDT_5m_1000.parquet"),
    ("ETH60", "tests/test_data/ETH_USDT_60m_1000.parquet"),
    ("ETH5m", "tests/test_data/ETH_USDT_5m_1000.parquet"),
    ("BTCd", "tests/test_data/BTC_USDT_d_500.parquet"),
]

for ds_name, ds_path in datasets:
    df = pd.read_parquet(ds_path)
    cl = CL_Pyarmor("test", "60m", config)
    cl.process_klines(df)
    bis = cl.get_bis()
    
    print(f"\n{'='*80}")
    print(f"=== {ds_name}: ALL bad TZXL FX cases (unique) ===")
    print(f"{'='*80}")
    
    seen = set()
    for start in range(0, len(bis) - 3):
        for fx_type in ['ding', 'di']:
            lines = bis[start:]
            result = cl._xd_cal_line_xlfx(lines, fx_type=fx_type, bh_type='no_bh')
            tzxls, xlfxs = result
            for fx in xlfxs:
                if not fx.xl.line_bad:
                    continue
                xl_bis = tuple(l.index for l in fx.xl.lines)
                key = (fx_type, xl_bis)
                if key in seen:
                    continue
                seen.add(key)
                
                # The bad TZXL has 1 line (NEW⊃OLD: new stands alone, old was consumed)
                # But wait - in the TZXL construction:
                #   OLD⊃NEW: OLD is kept, NEW is merged into OLD (nlines grows)
                #   NEW⊃OLD: OLD is kept, NEW stands alone with bad=True
                # So a bad TZXL is the NEW line that contains the OLD
                
                bad_bi = fx.xl.lines[0]  # The BI that caused containment
                
                # What was the PREVIOUS TZXL? That's fx.xls[0]
                prev_tzxl = fx.xls[0]
                prev_bi_last = prev_tzxl.lines[-1]  # Last BI of previous TZXL
                
                # The "old" line that was contained is prev_bi_last
                # The "new" line that contains is bad_bi
                # NEW⊃OLD means: bad_bi.high >= prev_bi_last.high AND bad_bi.low <= prev_bi_last.low
                # (but this is checked in the same-direction filtered BIs)
                
                # Actually, for bh_direction="down" (ding type, looking at down BIs):
                # TZXL max = max(highs), min = max(lows) -- NO WAIT
                # For ding: we look at down BIs. TZXL(bh_direction="down"):
                #   max = min(highs across lines), min = min(lows across lines)
                # Actually, let me trace the exact TZXL construction...
                
                # The key question: what makes is_line_bad different?
                # Let's look at ALL attributes we can find
                
                # Check neighboring TZXLs
                left = fx.xls[0]
                mid = fx.xls[1]  # = fx.xl
                right = fx.xls[2]
                
                # For ding (looking at down BIs): 
                #   FX exists if mid.max > left.max AND mid.max > right.max
                # For di (looking at up BIs):
                #   FX exists if mid.min < left.min AND mid.min < right.min
                
                # Check: does the bad TZXL have higher max than its predecessor?
                # This would indicate trend continuation vs reversal
                
                # Find bad TZXL position in full tzxls list
                bad_pos = None
                for ti, tz in enumerate(tzxls):
                    if any(l.index == fx.xl.lines[0].index for l in tz.lines):
                        bad_pos = ti
                        break
                
                # Check preceding TZXL (NOT the FX neighbor, but the actual predecessor)
                if bad_pos is not None and bad_pos > 0:
                    prev_tz = tzxls[bad_pos - 1]
                    prev_bis = [l.index for l in prev_tz.lines]
                    
                    # Compare prices
                    if fx_type == 'ding':
                        # For ding (down BIs): max = min(highs), min = min(lows)
                        # A "bad" down BI contains the previous: 
                        #   bad_bi.high >= prev_last.high AND bad_bi.low <= prev_last.low
                        # After TZXL construction with bh_direction="down":
                        #   bad_tzxl.max = min(highs) = bad_bi.high (single line)
                        #   prev_tzxl.max = min(highs_of_lines)
                        # The containment: bad > prev means NEW is HIGHER than OLD
                        direction_info = f"mid.max({mid.max:.1f}) vs prev.max({prev_tz.max:.1f})"
                        mid_exceeds = mid.max > prev_tz.max
                    else:
                        direction_info = f"mid.min({mid.min:.1f}) vs prev.min({prev_tz.min:.1f})"
                        mid_exceeds = mid.min < prev_tz.min
                else:
                    direction_info = "no_predecessor"
                    mid_exceeds = None
                
                marker = "FALSE" if not fx.is_line_bad else "TRUE"
                
                print(f"\n  {fx_type} FX@bi[{list(xl_bis)}] is_line_bad={marker} "
                      f"mid_exceeds_prev={mid_exceeds}")
                print(f"    {direction_info}")
                print(f"    left: bis={[l.index for l in left.lines]} bad={left.line_bad} "
                      f"max={left.max:.1f} min={left.min:.1f}")
                print(f"    mid:  bis={[l.index for l in mid.lines]} bad={mid.line_bad} "
                      f"max={mid.max:.1f} min={mid.min:.1f}")
                print(f"    right:bis={[l.index for l in right.lines]} bad={right.line_bad} "
                      f"max={right.max:.1f} min={right.min:.1f}")
                
                # Show raw BIs in the bad TZXL and its predecessor
                print(f"    Bad BI: bi[{bad_bi.index}] {bad_bi.type} h={bad_bi.high:.1f} l={bad_bi.low:.1f}")
                print(f"    Prev BI(last): bi[{prev_bi_last.index}] {prev_bi_last.type} "
                      f"h={prev_bi_last.high:.1f} l={prev_bi_last.low:.1f}")
                
                # The containment check: NEW(bad_bi) ⊃ OLD(prev_bi_last)
                contains = (bad_bi.high >= prev_bi_last.high and bad_bi.low <= prev_bi_last.low)
                print(f"    Containment: bad_bi⊃prev_last = {contains}")
