"""Check which TZXLs form DI FX for BTC5m down bi[46] under different line_bad rules"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# BTC5m TZXL list (from merge trace):
# TZXL[0]: max=68111.2 min=67764.1 bad=False lines=[47]
# TZXL[1]: max=68042.3 min=67411.0 bad=??? lines=[49,51]    (bad=False if reset, bad=True if preserved)
# TZXL[2]: max=67850.6 min=67568.4 bad=False lines=[53,55]
# TZXL[3]: max=67329.7 min=66850.1 bad=False lines=[57,59,61]
# TZXL[4]: max=67399.8 min=67073.5 bad=False lines=[63]
# TZXL[5]: max=67202.5 min=66508.0 bad=False lines=[65]
# TZXL[6]: max=67400.0 min=67002.4 bad=False lines=[67]
# TZXL[7]: max=68171.6 min=67121.1 bad=False lines=[69]

# For DOWN segment, target_fx_type = "di"
# DI FX: curr.min < prev.min AND curr.min < next.min

mins = [67764.1, 67411.0, 67568.4, 66850.1, 67073.5, 66508.0, 67002.4, 67121.1]
bads_reset = [False, False, False, False, False, False, False, False]  # current behavior
bads_preserved = [False, True, False, False, False, False, False, False]  # alternative

print("DI FX check (target: min < prev.min AND min < next.min):")
for i in range(1, len(mins) - 1):
    is_di = mins[i] < mins[i-1] and mins[i] < mins[i+1]
    if is_di:
        print(f"  TZXL[{i}]: min={mins[i]:.1f} (prev={mins[i-1]:.1f}, next={mins[i+1]:.1f}) → DI FX!")
        print(f"    With reset: bad={bads_reset[i]}")
        print(f"    With preserved: bad={bads_preserved[i]}")

# BTC60 TZXL list:
# TZXL[0]: max=70514.6 min=69212.2 bad=False
# TZXL[1]: max=70110.9 min=68029.6 bad=False
# TZXL[2]: max=69033.0 min=67785.4 bad=False
# TZXL[3]: max=68438.0 min=66588.0 bad=False
# TZXL[4]: max=67299.4 min=65826.1 bad=False
# TZXL[5]: max=68283.7 min=65595.7 bad=True
# TZXL[6]: max=68687.0 min=66915.0 bad=False
# TZXL[7]: max=66240.4 min=64232.8 bad=False
# TZXL[8]: max=68188.8 min=62401.7 bad=False/True?
# TZXL[9]: max=68189.0 min=62979.5 bad=False
# TZXL[10]: max=70100.0 min=65011.0 bad=False
# TZXL[11]: max=68171.6 min=66080.0 bad=False

print("\nBTC60 DI FX check:")
mins60 = [69212.2, 68029.6, 67785.4, 66588.0, 65826.1, 65595.7, 66915.0, 64232.8, 62401.7, 62979.5, 65011.0, 66080.0]
bads60_reset = [False, False, False, False, False, True, False, False, False, False, False, False]
bads60_preserved = [False, False, False, False, False, True, False, False, True, False, False, False]

for i in range(1, len(mins60) - 1):
    is_di = mins60[i] < mins60[i-1] and mins60[i] < mins60[i+1]
    if is_di:
        print(f"  TZXL[{i}]: min={mins60[i]:.1f} (prev={mins60[i-1]:.1f}, next={mins60[i+1]:.1f}) → DI FX!")
        print(f"    With reset: bad={bads60_reset[i]}")
        print(f"    With preserved: bad={bads60_preserved[i]}")
