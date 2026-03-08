# -*- coding: utf-8 -*-
"""
Decode lines 3865/3866: what is this check?
Theory: It's a SECOND strict check on DOWN stroke confirmation.
For DOWN stroke (ding→di): after passing low check (3850/3851),
check confirm.high vs end.high (3865/3866).

Also investigate: for UP stroke, does a symmetric second check exist?
Lines 3860/3861: end.low vs start.low for UP stroke setting end_fx.

Let's extract ALL 3865/3866 pairs and analyze outcomes.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from chanlun.cl_interface import FX

original_high = FX.high
original_low = FX.low

call_log = []
logging_enabled = False

def patched_high(self, qj_type, qy_type):
    result = original_high(self, qj_type, qy_type)
    if logging_enabled:
        frame = sys._getframe(1)
        line = frame.f_lineno
        call_log.append(('high', self.k.index, self.type, self.val, result, line))
    return result

def patched_low(self, qj_type, qy_type):
    result = original_low(self, qj_type, qy_type)
    if logging_enabled:
        frame = sys._getframe(1)
        line = frame.f_lineno
        call_log.append(('low', self.k.index, self.type, self.val, result, line))
    return result

FX.high = patched_high
FX.low = patched_low

from chanlun.cl_pyarmor import CL

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "test_data", "BTC_USDT_60m_500.parquet"))
cd = CL("BTC/USDT", "60m")

logging_enabled = True
cd.process_klines(df)
logging_enabled = False

# Extract 3865/3866 pairs
# These always come as 3865 followed by 3866
pairs_3865 = []
for i in range(len(call_log) - 1):
    if call_log[i][5] == 3865 and call_log[i+1][5] == 3866:
        m1, ck1, ft1, fv1, r1, l1 = call_log[i]
        m2, ck2, ft2, fv2, r2, l2 = call_log[i+1]
        pairs_3865.append((ck1, ft1, fv1, r1, ck2, ft2, fv2, r2))

print(f"Total 3865/3866 pairs: {len(pairs_3865)}")
print(f"\nAll unique 3865/3866 pairs:")
seen = set()
for p in pairs_3865:
    ck1, ft1, fv1, r1, ck2, ft2, fv2, r2 = p
    key = (ck1, ck2)
    if key not in seen:
        seen.add(key)
        # r1 = confirm.high, r2 = end.high  (based on trace: 3865=confirm, 3866=end)
        comparison = "confirm.high > end.high" if r1 > r2 else "confirm.high <= end.high"
        print(f"  {ft1}@{ck1}.high={r1:.1f} vs {ft2}@{ck2}.high={r2:.1f}  => {comparison}  (r1>r2: {r1>r2})")

# Also extract 3860/3861 pairs (UP stroke second check)
pairs_3860 = []
for i in range(len(call_log) - 1):
    if call_log[i][5] == 3860 and call_log[i+1][5] == 3861:
        m1, ck1, ft1, fv1, r1, l1 = call_log[i]
        m2, ck2, ft2, fv2, r2, l2 = call_log[i+1]
        pairs_3860.append((ck1, ft1, fv1, r1, ck2, ft2, fv2, r2))

print(f"\n\nTotal 3860/3861 pairs: {len(pairs_3860)}")
print(f"\nAll unique 3860/3861 pairs:")
seen2 = set()
for p in pairs_3860:
    ck1, ft1, fv1, r1, ck2, ft2, fv2, r2 = p
    key = (ck1, ck2)
    if key not in seen2:
        seen2.add(key)
        # r1 = end.low, r2 = start.low (UP stroke: end=ding, start=di)
        comparison = "end.low < start.low" if r1 < r2 else "end.low >= start.low"
        print(f"  {ft1}@{ck1}.low={r1:.1f} vs {ft2}@{ck2}.low={r2:.1f}  => {comparison}  (r1<r2: {r1<r2})")

# Now the key: what happens AFTER 3865/3866? 
# If confirm.high > end.high blocks the confirmation, then the next calls
# should NOT be about creating a BI (no further progression).
# If it doesn't block, the next calls should show progression.
print("\n\n=== Context after each unique 3865/3866 pair ===")
seen3 = set()
for i in range(len(call_log) - 1):
    if call_log[i][5] == 3865 and call_log[i+1][5] == 3866:
        m1, ck1, ft1, fv1, r1, l1 = call_log[i]
        m2, ck2, ft2, fv2, r2, l2 = call_log[i+1]
        key = (ck1, ck2, round(r1, 1))
        if key in seen3:
            continue
        seen3.add(key)
        
        comparison = r1 > r2
        # Look at next 4 calls after this pair
        next_calls = []
        for j in range(i+2, min(i+6, len(call_log))):
            m, ck, ft, fv, r, l = call_log[j]
            next_calls.append(f"FX[{ck}].{m}={r:.1f} line={l}")
        
        marker = "CONFIRM_HIGH_EXCEEDS" if comparison else "confirm_low_or_equal"
        print(f"\n  [{marker}] {ft1}@{ck1}.high({r1:.1f}) vs {ft2}@{ck2}.high({r2:.1f})")
        print(f"    Next: {' | '.join(next_calls)}")
