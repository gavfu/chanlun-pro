"""Simulate the TZXL inclusion processing step by step to understand
the exact algorithm. Process UP strokes one at a time for DOWN XD.

Key question: Does the containment check compare against the LAST 
TZXL element (even if bad), or the last NON-BAD element?

For bi[15]: prev elements are ...bi[11](good), bi[13](good)
- If checking vs bi[13]: bi[15] h=69999>66240 AND l=62401<64232 → NEW⊃OLD → bad=True ✓
- bi[15] bad=True at this point

For bi[17]: prev element is bi[15](bad)
- If checking vs bi[15]: bi[15] h=69999>68188 AND l=62401<66462 → OLD⊃NEW → MERGE ✓
- Merged bi[15,17]: bad=False (RESET)

So the check is against LAST element regardless of bad status. Let me verify with bi[9]:

For bi[9]: prev elements are ...bi[5], bi[7]  
- If checking vs bi[7]: bi[9] h=68283>67299 AND l=65595<65826 → NEW⊃OLD → bad=True ✓

For bi[11]: prev element is bi[9](bad)
- bi[11] h=68687,l=66915 vs bi[9] h=68283,l=65595
- h: 68687>68283 ✓, l: 66915>65595 ✗ → NOT contained → NEW element (not merged, not bad) ✓

Let me also check: does bi[11] get compared against bi[9] or bi[7]?
If against bi[9]: 68687>68283 AND 66915>65595 → no containment → separate, good ✓
If against bi[7]: 68687>67299 AND 66915>65826 → no containment (not bi[7]⊃bi[11] nor bi[11]⊃bi[7])
  Actually: 68687>=67299 and 66915>=65826 → NEW⊃OLD! Would make line_bad=True.
  But bi[11] has line_bad=False. So the check must be against bi[9] (LAST), not bi[7] (last non-bad).
"""
import sys; sys.path.insert(0, 'src')
import pandas as pd
from chanlun.cl_open import CL as CL_O

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')
cd_o = CL_O('BTC/USDT', '60m')
cd_o.process_klines(df)
bis = cd_o.bis

# Simulate step-by-step DOWN XD TZXL processing
# In DOWN XD, characteristic sequence uses UP strokes
# bh_direction = down, so max = min(highs), min = min(lows) for merged

up_bis = [bi for bi in bis if bi.type == "up"]
print("=== UP strokes (characteristic sequence for DOWN XD) ===")
for bi in up_bis:
    print(f"  bi[{bi.index}] h={bi.high:.1f} l={bi.low:.1f}")

print("\n=== Step-by-step TZXL inclusion processing ===")
print("(Comparing against LAST element regardless of bad status)")
print()

# Each TZXL element has: lines[], max, min, line_bad
tzxls = []

for bi in up_bis:
    new_max = bi.high
    new_min = bi.low
    new_lines = [bi]
    new_bad = False
    
    if len(tzxls) == 0:
        tzxls.append({
            'lines': new_lines, 'max': new_max, 'min': new_min, 'bad': False
        })
        print(f"  Add bi[{bi.index}] → tzxl[0] max={new_max:.1f} min={new_min:.1f}")
        continue
    
    last = tzxls[-1]
    last_max, last_min = last['max'], last['min']
    
    # Containment check
    old_contains_new = last_max >= new_max and last_min <= new_min
    new_contains_old = new_max >= last_max and new_min <= last_min
    
    if old_contains_new:
        # OLD ⊃ NEW → MERGE, reset bad=False
        # For DOWN direction: max = min(old_max, new_max), min = min(old_min, new_min)
        merged_max = min(last_max, new_max)
        merged_min = min(last_min, new_min)
        last['lines'].extend(new_lines)
        last['max'] = merged_max
        last['min'] = merged_min  
        last['bad'] = False  # RESET to False when merging
        lines_str = ','.join(str(l.index) for l in last['lines'])
        print(f"  bi[{bi.index}] OLD⊃NEW with last → MERGE → bi[{lines_str}] max={merged_max:.1f} min={merged_min:.1f} bad=False")
    elif new_contains_old:
        # NEW ⊃ OLD → SEPARATE, mark bad=True
        tzxls.append({
            'lines': new_lines, 'max': new_max, 'min': new_min, 'bad': True
        })
        idx = len(tzxls) - 1
        last_lines_str = ','.join(str(l.index) for l in last['lines'])
        print(f"  bi[{bi.index}] NEW⊃OLD with last bi[{last_lines_str}] → SEPARATE → tzxl[{idx}] bad=True")
    else:
        # No containment → normal new element
        tzxls.append({
            'lines': new_lines, 'max': new_max, 'min': new_min, 'bad': False
        })
        idx = len(tzxls) - 1
        print(f"  bi[{bi.index}] no containment → tzxl[{idx}] bad=False")

print("\n=== Final TZXL list ===")
for i, t in enumerate(tzxls):
    lines_str = ','.join(str(l.index) for l in t['lines'])
    print(f"  [{i}] bi[{lines_str}] max={t['max']:.1f} min={t['min']:.1f} bad={t['bad']}")

# Expected pyarmor TZXL for XD[0]:
print("\n=== Expected (pyarmor) ===")
expected = [
    ([1], 70110.9, 68112.2, False),
    ([3], 69033.0, 67785.4, False),
    ([5], 68438.0, 66588.0, False),
    ([7], 67299.4, 65826.1, False),
    ([9], 68283.7, 65595.7, True),
    ([11], 68687.0, 66915.0, False),
    ([13], 66240.4, 64232.8, False),
    ([15,17], 68188.8, 62401.7, False),
    ([19], 68189.0, 62979.5, False),
]
for i, (lines, mx, mn, bad) in enumerate(expected):
    print(f"  [{i}] bi[{','.join(str(l) for l in lines)}] max={mx} min={mn} bad={bad}")
