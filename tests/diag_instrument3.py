# -*- coding: utf-8 -*-
"""
Instrumentation to trace pyarmor bi construction calls.
Focus on areas where strokes SUCCEED to see the full pattern.
Also trace the 262-276 area to understand blocking.
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
        line = str(frame.f_lineno)
        call_log.append(('high', self.k.index, self.type, self.val, result, line))
    return result

def patched_low(self, qj_type, qy_type):
    result = original_low(self, qj_type, qy_type)
    if logging_enabled:
        frame = sys._getframe(1)
        line = str(frame.f_lineno)
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

BI_LINES = {'3855','3856','3860','3861','3850','3851','3865','3866'}

# Print the LAST iteration of each check in the 262-276 area
# (final state of incremental processing)
print("=" * 80)
print("BI CONSTRUCTION CALLS (only bi-related lines 3850-3866)")
print("=" * 80)

# Part 1: Look at the LAST complete cycle for stroke up 233->243 (successful)
# Focus on calls where BOTH FXs are in range and line is bi-related
print("\n--- Area: ck 230-265 (successful strokes + divergence area) ---")
for i, entry in enumerate(call_log):
    method, ck, ftype, fval, result, line = entry
    if 230 <= ck <= 280 and line in BI_LINES:
        print(f"  [{i:5d}] FX[{ck:3d}].{method:4s} = {result:9.1f}  [{ftype}@{ck} val={fval:.1f}] line={line}")

# Part 2: Look at the confirmation check pattern for 3865/3866
# Find ALL occurrences of 3865/3866 and what happens around them
print("\n\n--- ALL 3865/3866 calls (post-confirmation check) ---")
for i, entry in enumerate(call_log):
    method, ck, ftype, fval, result, line = entry
    if line in ('3865', '3866'):
        # Show context: 2 lines before and 2 after
        for j in range(max(0, i-4), min(len(call_log), i+3)):
            m, c, ft, fv, r, l = call_log[j]
            marker = " >>>" if j == i else "    "
            if l in BI_LINES:
                print(f"{marker}[{j:5d}] FX[{c:3d}].{m:4s} = {r:9.1f}  [{ft}@{c} val={fv:.1f}] line={l}")
        print("  ---")

# Part 3: Show the FIRST successful confirmation (earliest bi)
# Find where bi[0] gets confirmed
print("\n\n--- Area: ck 0-50 (first strokes) ---")
for i, entry in enumerate(call_log):
    method, ck, ftype, fval, result, line = entry
    if 0 <= ck <= 50 and line in BI_LINES:
        print(f"  [{i:5d}] FX[{ck:3d}].{method:4s} = {result:9.1f}  [{ftype}@{ck} val={fval:.1f}] line={line}")
