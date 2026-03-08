# -*- coding: utf-8 -*-
"""
Better instrumentation: capture call PAIRS with call stack context
to understand exactly what comparisons pyarmor makes.
"""
import os, sys, traceback
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
        # Get 1 frame of caller info
        frame = sys._getframe(1)
        caller = f"{frame.f_code.co_filename}:{frame.f_lineno}"
        call_log.append(('high', self.k.index, self.type, self.val, qj_type, qy_type, result, caller))
    return result

def patched_low(self, qj_type, qy_type):
    result = original_low(self, qj_type, qy_type)
    if logging_enabled:
        frame = sys._getframe(1)
        caller = f"{frame.f_code.co_filename}:{frame.f_lineno}"
        call_log.append(('low', self.k.index, self.type, self.val, qj_type, qy_type, result, caller))
    return result

FX.high = patched_high
FX.low = patched_low

from chanlun.cl_pyarmor import CL

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "test_data", "BTC_USDT_60m_500.parquet"))
cd = CL("BTC/USDT", "60m")

logging_enabled = True
cd.process_klines(df)
logging_enabled = False

# The calls happen during bi construction. 
# Group them by consecutive pairs from different FXs
print(f"Total calls: {len(call_log)}")

# Find unique callers
callers = set()
for entry in call_log:
    callers.add(entry[7])
print(f"\nUnique callers (line numbers in pyarmor):")
for c in sorted(callers):
    count = sum(1 for e in call_log if e[7] == c)
    print(f"  {c}: {count} calls")

# Now detect COMPARISON patterns: consecutive calls from DIFFERENT FXs
# that likely form a comparison  
print(f"\n=== Call sequence around ck 262-276 ===")
for i, entry in enumerate(call_log):
    method, ck, ftype, fval, qj, qy, result, caller = entry
    if 260 <= ck <= 280:
        # Show this call and its neighbors
        print(f"  [{i:5d}] FX[{ck:3d}].{method:4s}({qy}) = {result:8.1f}  [{ftype}@{ck} val={fval:.1f}] from {caller.split('/')[-1]}")

# Show around ck 96-113
print(f"\n=== Call sequence around ck 96-113 ===")
for i, entry in enumerate(call_log):
    method, ck, ftype, fval, qj, qy, result, caller = entry
    if 94 <= ck <= 115:
        print(f"  [{i:5d}] FX[{ck:3d}].{method:4s}({qy}) = {result:8.1f}  [{ftype}@{ck} val={fval:.1f}] from {caller.split('/')[-1]}")

# Show around ck 40-56
print(f"\n=== Call sequence around ck 40-56 ===")
for i, entry in enumerate(call_log):
    method, ck, ftype, fval, qj, qy, result, caller = entry
    if 38 <= ck <= 58:
        print(f"  [{i:5d}] FX[{ck:3d}].{method:4s}({qy}) = {result:8.1f}  [{ftype}@{ck} val={fval:.1f}] from {caller.split('/')[-1]}")

# Show around ck 25-29
print(f"\n=== Call sequence around ck 25-29 ===")
for i, entry in enumerate(call_log):
    method, ck, ftype, fval, qj, qy, result, caller = entry
    if 23 <= ck <= 31:
        print(f"  [{i:5d}] FX[{ck:3d}].{method:4s}({qy}) = {result:8.1f}  [{ftype}@{ck} val={fval:.1f}] from {caller.split('/')[-1]}")
