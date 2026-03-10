"""
Trace pyarmor's TZXL construction in detail.
We need to understand WHY pyarmor merges bi[37] and bi[39] while our code doesn't.

Our code: bi[39] NEW⊃OLD bi[37] (using direction-adjusted max/min for bh_direction="down")
Pyarmor: MERGED into one TZXL with lines=[37, 39]

Key question: What containment criterion does pyarmor use?
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_interface import ICL, TZXL
from chanlun.cl import CL

# Load BTC60 data
df = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
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

cl_pyarmor = CL("BTC60", "60m", config)
cl_pyarmor.process_klines(df)

# Get BI list
bis = cl_pyarmor.get_bis()

# Focus on BTC60 down from bi[28]: looking at UP BIs (odd indices for down segment starting at even)
# bi[28] is DOWN, so UP BIs are 29, 31, 33, 35, 37, 39, 41, ...
print("=== BTC60 down from bi[28]: UP BIs (TZXL candidates) ===")
print("bh_direction = 'down'")
print()

# For bh_direction="down", TZXL uses: max = min(highs of lines), min = min(lows of lines)
# But what if pyarmor uses different values for containment check?
up_bis = [bi for bi in bis if bi.index >= 29 and bi.index <= 45 and bi.type == 'up']
for bi in up_bis:
    print(f"  bi[{bi.index}]: type={bi.type}, high={bi.high}, low={bi.low}")
    # For single-BI TZXL with bh_direction="down":
    # TZXL.max = min(highs) = bi.high (only one BI)
    # TZXL.min = min(lows) = bi.low (only one BI)
    print(f"    TZXL(down): max={bi.high}, min={bi.low}")
    # Check against previous
    if bi.index > 29:
        prev = [b for b in up_bis if b.index < bi.index][-1]
        # Direction-adjusted containment (our code)
        our_max, our_min = bi.high, bi.low
        prev_max, prev_min = prev.high, prev.low
        
        o_c_n = prev_max >= our_max and prev_min <= our_min  # OLD⊃NEW
        n_c_o = our_max >= prev_max and our_min <= prev_min  # NEW⊃OLD
        
        # Raw BI high/low containment (alternative)
        raw_o_c_n = prev.high >= bi.high and prev.low <= bi.low
        raw_n_c_o = bi.high >= prev.high and bi.low <= prev.low
        
        print(f"    vs prev bi[{prev.index}] (max={prev_max}, min={prev_min}):")
        print(f"      Direction-adj: OLD⊃NEW={o_c_n}, NEW⊃OLD={n_c_o}")
        print(f"      Raw BI h/l:   OLD⊃NEW={raw_o_c_n}, NEW⊃OLD={raw_n_c_o}")
    print()

# Now let's also check what happens if we have an accumulated TZXL from previous merges
# After processing bi[29] through bi[37], let's see the TZXL state
print("=== Simulating TZXL construction with our rules ===")
bh_direction = "down"
tzxls = []
for bi in up_bis:
    new_tzxl = TZXL(
        bh_direction=bh_direction,
        line=bi,
        pre_line=bis[bi.index - 1],
        line_bad=False,
        done=bi.is_done(),
    )
    
    if len(tzxls) == 0:
        tzxls.append(new_tzxl)
        print(f"  Add TZXL[0]: bi[{bi.index}] max={new_tzxl.max} min={new_tzxl.min}")
        continue
    
    last = tzxls[-1]
    o_c_n = last.max >= new_tzxl.max and last.min <= new_tzxl.min
    n_c_o = new_tzxl.max >= last.max and new_tzxl.min <= last.min
    
    print(f"  bi[{bi.index}] (max={new_tzxl.max}, min={new_tzxl.min}) vs TZXL[-1] (max={last.max}, min={last.min}, lines={[l.index for l in last.lines]}):")
    if o_c_n:
        print(f"    → OLD⊃NEW: MERGE into TZXL[-1]")
        last.lines.append(bi)
        last.done = bi.is_done()
        last.line_bad = False
        last.update_maxmin()
        print(f"    → After merge: max={last.max}, min={last.min}, lines={[l.index for l in last.lines]}")
    elif n_c_o:
        print(f"    → NEW⊃OLD: NEW element, line_bad=True")
        new_tzxl.line_bad = True
        tzxls.append(new_tzxl)
    else:
        print(f"    → No containment: NEW element, line_bad=False")
        tzxls.append(new_tzxl)

print()
print("=== Our final TZXLs ===")
for i, t in enumerate(tzxls):
    print(f"  [{i}]: max={t.max} min={t.min} bad={t.line_bad} lines={[l.index for l in t.lines]}")

print()
print("=== Pyarmor's TZXLs (from trace) ===")
print("  [0]: max=70514.6 min=69212.2 bad=False lines=[29]")
print("  [1]: max=70110.9 min=68029.6 bad=False lines=[31]")
print("  [2]: max=69033.0 min=67785.4 bad=False lines=[33]")
print("  [3]: max=68438.0 min=66588.0 bad=False lines=[35]")
print("  [4]: max=67299.4 min=65595.7 bad=False lines=[37,39]")
print("  [5]: max=68115.9 min=66915.0 bad=False lines=[41]")

# KEY INSIGHT: pyarmor's TZXL[4] has max=67299.4 min=65595.7
# If we look at bi[37]: high=67299.4, low=65826.1
# And bi[39]: high=68283.7, low=65595.7
# Then max=67299.4 = min(67299.4, 68283.7) and min=65595.7 = min(65826.1, 65595.7)
# This is consistent with bh_direction="down" update_maxmin
# But this means pyarmor TREATED it as OLD⊃NEW (merged into old)
# even though in our containment check it's NEW⊃OLD!
#
# So pyarmor's containment criterion must be DIFFERENT.
# Let's check: what if pyarmor checks containment using RAW BI values
# (the CURRENT BI only, not the accumulated TZXL max/min)?

print()
print("=== Hypothesis: Pyarmor uses raw BI high/low for ALL containment checks ===")
print(f"  bi[37] raw: high={up_bis[4].high}, low={up_bis[4].low}")
print(f"  bi[39] raw: high={up_bis[5].high}, low={up_bis[5].low}")
print(f"  OLD(bi[37])⊃NEW(bi[39])? high: {up_bis[4].high} >= {up_bis[5].high} = {up_bis[4].high >= up_bis[5].high}")
print(f"  NEW(bi[39])⊃OLD(bi[37])? high: {up_bis[5].high} >= {up_bis[4].high} = {up_bis[5].high >= up_bis[4].high}")

print()
print("=== Hypothesis: Pyarmor checks containment differently for bh_type='bh' ===")
print("  Maybe for bh_direction='down' (looking at UP BIs), containment uses:")
print("    OLD⊃NEW if old.low <= new.low (only comparing the 'direction-relevant' value)")
print("    or some other single-value comparison")
print(f"  bi[37].low={up_bis[4].low}, bi[39].low={up_bis[5].low}")
print(f"  bi[37].low <= bi[39].low? {up_bis[4].low <= up_bis[5].low}")
print(f"  bi[37].low >= bi[39].low? {up_bis[4].low >= up_bis[5].low}")

print()
print("=== Hypothesis: Pyarmor uses OPPOSITE direction for containment ===")
print("  bh_direction='down' but containment uses 'up' logic?")
print("  For 'up': max=max(highs), min=max(lows)")
print(f"  bi[37] as up-TZXL: max={up_bis[4].high}, min={up_bis[4].low}")
print(f"  bi[39] as up-TZXL: max={up_bis[5].high}, min={up_bis[5].low}")
print(f"  OLD⊃NEW (up): {up_bis[4].high} >= {up_bis[5].high} AND {up_bis[4].low} <= {up_bis[5].low}")
print(f"    = {up_bis[4].high >= up_bis[5].high} AND {up_bis[4].low <= up_bis[5].low}")
print(f"    = {up_bis[4].high >= up_bis[5].high and up_bis[4].low <= up_bis[5].low}")

print()
print("=== Hypothesis: Pyarmor uses K-line style containment (high/low) ===")
print("  Just like K-line merging: compare high and low directly")
print("  For a DOWN trend merge direction:")  
print("  If overlap, merge taking MIN of highs and MIN of lows (down-direction merge)")
print(f"  bi[37]: h={up_bis[4].high}, l={up_bis[4].low}")
print(f"  bi[39]: h={up_bis[5].high}, l={up_bis[5].low}")
print(f"  Does bi[37].high >= bi[39].low AND bi[39].high >= bi[37].low? (overlap check)")
print(f"    {up_bis[4].high >= up_bis[5].low} AND {up_bis[5].high >= up_bis[4].low}")

# Let's also check what pyarmor does with bi[35] → bi[37] pair (no merge in pyarmor)
print()
print("=== Check bi[35] → bi[37] (NOT merged in pyarmor) ===")
bi35 = up_bis[3]  # index 35
bi37 = up_bis[4]  # index 37
print(f"  bi[35]: high={bi35.high}, low={bi35.low}")
print(f"  bi[37]: high={bi37.high}, low={bi37.low}")
print(f"  Direction-adj containment (down):")
print(f"    OLD(35)⊃NEW(37): {bi35.high} >= {bi37.high} AND {bi35.low} <= {bi37.low}")
print(f"      = {bi35.high >= bi37.high} AND {bi35.low <= bi37.low} = {bi35.high >= bi37.high and bi35.low <= bi37.low}")
print(f"    NEW(37)⊃OLD(35): {bi37.high} >= {bi35.high} AND {bi37.low} <= {bi35.low}")
print(f"      = {bi37.high >= bi35.high} AND {bi37.low <= bi35.low} = {bi37.high >= bi35.high and bi37.low <= bi35.low}")

# Check what the pre_line values are
print()
print("=== Check pre_line (DOWN BIs before each UP BI) ===")
for bi in up_bis:
    pre = bis[bi.index - 1]
    print(f"  bi[{bi.index}] pre_line = bi[{pre.index}]: high={pre.high}, low={pre.low}")

# Maybe pyarmor uses pre_line for containment?
print()
print("=== Hypothesis: Pyarmor uses pre_line for containment check ===")
print("  (pre_line is the DOWN BI before each UP BI)")
bi37_pre = bis[up_bis[4].index - 1]  # bi[36]
bi39_pre = bis[up_bis[5].index - 1]  # bi[38]
print(f"  bi[37].pre = bi[36]: high={bi37_pre.high}, low={bi37_pre.low}")
print(f"  bi[39].pre = bi[38]: high={bi39_pre.high}, low={bi39_pre.low}")
print(f"  OLD(pre36)⊃NEW(pre38): {bi37_pre.high} >= {bi39_pre.high} AND {bi37_pre.low} <= {bi39_pre.low}")
print(f"    = {bi37_pre.high >= bi39_pre.high} AND {bi37_pre.low <= bi39_pre.low}")
print(f"    = {bi37_pre.high >= bi39_pre.high and bi37_pre.low <= bi39_pre.low}")
