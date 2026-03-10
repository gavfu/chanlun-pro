"""
Key finding so far:
- ALL is_line_bad=FALSE with xl.line_bad=True are fx_type='di' 
- ALL is_line_bad=TRUE with xl.line_bad=True are fx_type='ding' (except BTC60 di FX@bi[49])

The ONE exception is BTC60 di FX@bi[49]. Let's investigate this exception.

Wait - maybe it's about the BI TYPE (direction of the actual BI), not the fx_type:
- di FX type looks at UP BIs (taking min of each up BI's lows)
- ding FX type looks at DOWN BIs (taking min of each down BI's highs)

For di type: the bad TZXL is an UP BI
For ding type: the bad TZXL is a DOWN BI

So maybe is_line_bad depends on:
- For ding (looking at down BIs): always True when bad
- For di (looking at up BIs): depends on something specific

Let's investigate BTC60 di FX@bi[49] specifically.
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

df60 = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cl60 = CL_Pyarmor("test", "60m", config)
cl60.process_klines(df60)
bis60 = cl60.get_bis()

# BTC60 di FX@bi[49]: is_line_bad=TRUE, the ONE exception
# Let's check: bi[49] is up
bi49 = bis60[49]
print(f"bi[49]: type={bi49.type} high={bi49.high} low={bi49.low}")

# The prev TZXL should be bi[47]
bi47 = bis60[47]
print(f"bi[47]: type={bi47.type} high={bi47.high} low={bi47.low}")

# bi[45] is also relevant
bi45 = bis60[45]
print(f"bi[45]: type={bi45.type} high={bi45.high} low={bi45.low}")

# In the TZXL list from start=28 (di mode), bi[45,47] are merged (OLD⊃NEW):
# bis=[45, 47] bad=False max=68188.8 min=62401.7 done=True nlines=2
# bi[45] up high=69999.0 low=62401.7
# bi[47] up high=68188.8 low=66462.0
# Then bi[49] is separate with bad=True:
# bis=[49] bad=True max=68189.0 min=62979.5

# Wait - but the TZXL for bi[49] shows max=68189.0 min=62979.5
# vs the merged [45,47] TZXL: max=68188.8 min=62401.7
# Containment in TZXL terms: is 49 ⊃ [45,47]?
# TZXL max comparison: 68189.0 vs 68188.8 - 49 has slightly higher max
# TZXL min comparison: 62979.5 vs 62401.7 - 49 has higher min
# So bi[49] does NOT contain TZXL[45,47] in TZXL terms!

# But what about at the raw BI level?
# bi[49]: h=68189.0, l=62979.5
# bi[47]: h=68188.8, l=66462.0
# Contains? 68189.0 >= 68188.8 ✓ AND 62979.5 <= 66462.0 ✓ → YES contains!
# But TZXL max for [45,47] = min(highs) = min(69999.0, 68188.8) = 68188.8
# TZXL min for [45,47] = min(lows) = min(62401.7, 66462.0) = 62401.7

# Hmm wait, for bh_direction="down" (used for di FX of up BIs):
# No wait - for di FX, we're looking at UP BIs
# bh_direction for "di" of up BIs should be "up"? Let me check...

# Let me trace what _xd_cal_line_xlfx does with the bh_direction
# From code: for "di" searching min, up BIs...

# Actually, the key difference might be: bi[49] contains bi[47] but NOT bi[45]
# while bi[39] contains bi[37] at the raw BI level
# Let me check all containments

print("\n\n=== Detailed containment analysis ===")

# Check FALSE case: BTC60 di FX@bi[39]
print("\n--- FALSE case: di FX@bi[39] ---")
bi37 = bis60[37]
bi39 = bis60[39]
bi41 = bis60[41]
print(f"bi[37]: h={bi37.high}, l={bi37.low}")
print(f"bi[39]: h={bi39.high}, l={bi39.low}")
print(f"bi[41]: h={bi41.high}, l={bi41.low}")
print(f"bi[39]⊃bi[37]? h:{bi39.high >= bi37.high}({bi39.high} >= {bi37.high}) "
      f"l:{bi39.low <= bi37.low}({bi39.low} <= {bi37.low}) → {bi39.high >= bi37.high and bi39.low <= bi37.low}")

# Check TRUE case: BTC60 di FX@bi[49]
print("\n--- TRUE case: di FX@bi[49] ---")
print(f"bi[45]: h={bi45.high}, l={bi45.low}")
print(f"bi[47]: h={bi47.high}, l={bi47.low}")
print(f"bi[49]: h={bi49.high}, l={bi49.low}")
print(f"bi[49]⊃bi[47]? h:{bi49.high >= bi47.high}({bi49.high} >= {bi47.high}) "
      f"l:{bi49.low <= bi47.low}({bi49.low} <= {bi47.low}) → {bi49.high >= bi47.high and bi49.low <= bi47.low}")
print(f"bi[49]⊃bi[45]? h:{bi49.high >= bi45.high}({bi49.high} >= {bi45.high}) "
      f"l:{bi49.low <= bi45.low}({bi49.low} <= {bi45.low}) → {bi49.high >= bi45.high and bi49.low <= bi45.low}")

# Check FALSE case: BTC60 di FX@bi[25]
print("\n--- FALSE case: di FX@bi[25] ---")
bi23 = bis60[23]
bi25 = bis60[25]
bi27 = bis60[27]
print(f"bi[23]: h={bi23.high}, l={bi23.low}")
print(f"bi[25]: h={bi25.high}, l={bi25.low}")
print(f"bi[27]: h={bi27.high}, l={bi27.low}")
print(f"bi[25]⊃bi[23]? h:{bi25.high >= bi23.high}({bi25.high} >= {bi23.high}) "
      f"l:{bi25.low <= bi23.low}({bi25.low} <= {bi23.low}) → {bi25.high >= bi23.high and bi25.low <= bi23.low}")

# Now let me check: what is the TZXL's bh_direction for these?
# _xd_cal_line_xlfx takes fx_type and bh_type
# Let me check the TZXL.bh_direction attribute
print("\n\n=== TZXL direction check ===")

for start, ft in [(28, 'di'), (39, 'ding'), (0, 'di')]:
    lines = bis60[start:]
    result = cl60._xd_cal_line_xlfx(lines, fx_type=ft, bh_type='no_bh')
    tzxls, xlfxs = result
    for tz in tzxls:
        if tz.line_bad:
            bi_indices = [l.index for l in tz.lines]
            # Check if TZXL has bh_direction attribute
            attrs = [a for a in dir(tz) if not a.startswith('_')]
            if hasattr(tz, 'bh_direction'):
                print(f"  start={start} {ft}: TZXL@bis={bi_indices} bad=True "
                      f"bh_direction={tz.bh_direction}")
            else:
                print(f"  start={start} {ft}: TZXL@bis={bi_indices} bad=True "
                      f"NO bh_direction attr. attrs: {attrs}")

# Now let me check: is it the TZXL construction that differs? 
# For di FX (up BIs), what bh_direction is used?
# From cl_open.py: TZXL construction uses bh_direction based on the line type
print("\n\n=== TZXL construction comparison ===")

# Attempt: compare the actual TZXL directly before and after the bad one
# to see what containment relationship exists

# For di FX@bi[39] (FALSE):
# TZXL sequence: ...[35](not_bad) [37](not_bad) [39](bad) [41](not_bad)...
# For di FX@bi[49] (TRUE):  
# TZXL sequence: ...[43](not_bad) [45,47](not_bad) [49](bad) [51](not_bad)...

# KEY DIFFERENCE: bi[49]'s predecessor is a MERGED TZXL [45,47]
# while bi[39]'s predecessor [37] is a single-line TZXL

# Let me check if the containment happens at the TZXL level vs BI level
print("\n\n=== TZXL-level containment ===")

# FALSE case bi[39]:
# TZXL[37]: max=67299.4 min=65826.1 (single bi[37])
# TZXL[39]: max=68283.7 min=65595.7 (single bi[39], bad)
# TZXL[39] ⊃ TZXL[37]? max:68283.7>67299.4 ✓ min:65595.7<65826.1 ✓ → YES

# TRUE case bi[49]:
# TZXL[45,47]: max=68188.8 min=62401.7 (merged [45,47])
# TZXL[49]: max=68189.0 min=62979.5 (single bi[49], bad)
# TZXL[49] ⊃ TZXL[45,47]? max:68189.0>68188.8 ✓ min:62979.5<62401.7 ✗ → NO!

print("FALSE case bi[39]: TZXL[39] ⊃ TZXL[37]?")
print(f"  max: 68283.7 > 67299.4 = {68283.7 > 67299.4}")
print(f"  min: 65595.7 < 65826.1 = {65595.7 < 65826.1}")
print(f"  Contains: {68283.7 > 67299.4 and 65595.7 < 65826.1}")

print("\nTRUE case bi[49]: TZXL[49] ⊃ TZXL[45,47]?")
print(f"  max: 68189.0 > 68188.8 = {68189.0 > 68188.8}")
print(f"  min: 62979.5 < 62401.7 = {62979.5 < 62401.7}")
print(f"  Contains: {68189.0 > 68188.8 and 62979.5 < 62401.7}")

print("\n  *** bi[49] does NOT contain TZXL[45,47] at TZXL level (min fails)! ***")
print("  *** But bi[49] DOES contain bi[47] at BI level ***")
print("  *** So is_line_bad might be about TZXL-level containment, not BI-level! ***")

# Let me verify this hypothesis for ALL cases
print("\n\n=== HYPOTHESIS: is_line_bad = NOT (TZXL-level containment) ===")
print("  i.e., is_line_bad=False means the bad TZXL truly contains its predecessor at TZXL level")
print("  and is_line_bad=True means it does NOT (containment was only at BI level)")

for ds_name, ds_path in [
    ("BTC60", "tests/test_data/BTC_USDT_60m_1000.parquet"),
    ("BTC5m", "tests/test_data/BTC_USDT_5m_1000.parquet"),
    ("ETH60", "tests/test_data/ETH_USDT_60m_1000.parquet"),
    ("ETH5m", "tests/test_data/ETH_USDT_5m_1000.parquet"),
    ("BTCd", "tests/test_data/BTC_USDT_d_500.parquet"),
]:
    df = pd.read_parquet(ds_path)
    cl = CL_Pyarmor("test", "60m", config)
    cl.process_klines(df)
    bis = cl.get_bis()
    
    print(f"\n  {ds_name}:")
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
                
                # Find this TZXL and its predecessor in the tzxls list
                bad_pos = None
                for ti, tz in enumerate(tzxls):
                    if id(tz) == id(fx.xl) or (len(tz.lines) == len(fx.xl.lines) and 
                        all(l1.index == l2.index for l1, l2 in zip(tz.lines, fx.xl.lines))):
                        bad_pos = ti
                        break
                
                if bad_pos is not None and bad_pos > 0:
                    prev_tz = tzxls[bad_pos - 1]
                    # Check TZXL-level containment
                    # For both ding and di: the "containment" means
                    # bad.max >= prev.max AND bad.min <= prev.min
                    tzxl_contains = (fx.xl.max >= prev_tz.max and fx.xl.min <= prev_tz.min)
                    
                    predicted = not tzxl_contains  # is_line_bad = NOT contains
                    actual = fx.is_line_bad
                    match = "✓" if predicted == actual else "✗ MISMATCH"
                    
                    print(f"    {fx_type} FX@bi[{list(xl_bis)}] "
                          f"is_line_bad={actual} "
                          f"tzxl_contains={tzxl_contains} "
                          f"predicted(not_contains)={predicted} "
                          f"{match}")
                    print(f"      bad: max={fx.xl.max:.1f} min={fx.xl.min:.1f}")
                    print(f"      prev: max={prev_tz.max:.1f} min={prev_tz.min:.1f} "
                          f"bis={[l.index for l in prev_tz.lines]}")
