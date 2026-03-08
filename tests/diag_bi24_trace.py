# -*- coding: utf-8 -*-
"""
Check pyarmor's confirmation behavior around 347-370.
Does pyarmor apply strict check to confirmation?
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

BI_LINES = {3850, 3851, 3855, 3856, 3860, 3861, 3865, 3866}

# Focus on area 340-375 (around the 347->355 stroke)
print("=== BI construction calls around ck 340-375 ===")
for i, entry in enumerate(call_log):
    method, ck, ftype, fval, result, line = entry
    if 340 <= ck <= 375 and line in BI_LINES:
        print(f"  [{i:5d}] FX[{ck:3d}].{method:4s} = {result:9.1f}  [{ftype}@{ck} val={fval:.1f}] line={line}")

# Specifically trace the confirmation from di@347:
# For UP stroke from di@347, the end candidates are dings: 348, 351, 353, 355
# Check what calls pyarmor makes with each
print("\n\n=== ALL BI calls involving FX[347] ===")
for i, entry in enumerate(call_log):
    method, ck, ftype, fval, result, line = entry
    if ck == 347 and line in BI_LINES:
        # Show this and next 3 lines
        for j in range(max(0, i-1), min(len(call_log), i+4)):
            m, c, ft, fv, r, l = call_log[j]
            marker = ">>>" if j == i else "   "
            if l in BI_LINES:
                print(f"  {marker}[{j:5d}] FX[{c:3d}].{m:4s} = {r:9.1f}  [{ft}@{c} val={fv:.1f}] line={l}")
        print("  ---")

# Also check: does pyarmor ever do a strict check on the confirmation from 347?
# UP stroke check (3855/3856) on 347->355:
print("\n\n=== Looking for UP check 347->355 (3855 on 347 followed by 3856 on 355) ===")
for i in range(len(call_log) - 1):
    if (call_log[i][5] == 3855 and call_log[i][1] == 347 and 
        call_log[i+1][5] == 3856 and call_log[i+1][1] == 355):
        m1, c1, ft1, fv1, r1, l1 = call_log[i]
        m2, c2, ft2, fv2, r2, l2 = call_log[i+1]
        print(f"  FOUND: FX[{c1}].high={r1:.1f} (line={l1}) vs FX[{c2}].high={r2:.1f} (line={l2})")
        # Show block
        blocked = r1 > r2
        print(f"    start.high({r1:.1f}) > end.high({r2:.1f}) = {blocked}")

print("(if none found, pyarmor doesn't check 347->355 UP strict)")

# Check also DOWN strict check on 315->347 confirmation
print("\n\n=== DOWN checks involving ding@315 as start and di@347 as end (3850/3851) ===")
for i in range(len(call_log) - 1):
    if (call_log[i][5] == 3850 and call_log[i][1] == 315 and 
        call_log[i+1][5] == 3851):
        m1, c1, ft1, fv1, r1, l1 = call_log[i]
        m2, c2, ft2, fv2, r2, l2 = call_log[i+1]
        print(f"  ding@315.low={r1:.1f} vs {ft2}@{c2}.low={r2:.1f} blocked={r1 < r2}")
