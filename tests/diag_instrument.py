# -*- coding: utf-8 -*-
"""
Instrument pyarmor CL to trace its internal _bi_fx_valid calls.
Since pyarmor is encrypted, we can't read its source, but we CAN
monkey-patch the FX class to log when its high()/low() methods are called
during bi construction.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from chanlun.cl_interface import FX, Config

# Save original methods
original_high = FX.high
original_low = FX.low

call_log = []
logging_enabled = False

def patched_high(self, qj_type, qy_type):
    result = original_high(self, qj_type, qy_type)
    if logging_enabled:
        call_log.append(('high', self.k.index, self.type, qj_type, qy_type, result))
    return result

def patched_low(self, qj_type, qy_type):
    result = original_low(self, qj_type, qy_type)
    if logging_enabled:
        call_log.append(('low', self.k.index, self.type, qj_type, qy_type, result))
    return result

FX.high = patched_high
FX.low = patched_low

from chanlun.cl_pyarmor import CL as CL_Pyarmor

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "test_data", "BTC_USDT_60m_500.parquet"))

# First run without logging to initialize
cd = CL_Pyarmor("BTC/USDT", "60m")

# Now enable logging and process
logging_enabled = True
cd.process_klines(df)
logging_enabled = False

# Analyze the log
print(f"Total high/low calls: {len(call_log)}")

# Filter for calls involving our key fractals
key_cks = {40, 44, 96, 101, 106, 262, 265, 267, 270, 274, 276, 347, 355, 25, 29}
print(f"\n=== Calls involving key fractals ===")
for method, ck, ftype, qj, qy, result in call_log:
    if ck in key_cks:
        print(f"  FX[{ck:3d}].{method}({qj}, {qy}) = {result:.1f}  [{ftype}]")

# Show ALL high/low pairs (grouped by consecutive calls)
print(f"\n=== All high/low call pairs ===")
i = 0
pair_count = 0
while i < len(call_log) - 1:
    c1 = call_log[i]
    c2 = call_log[i+1]
    
    # Look for patterns: high called on one FX, then high/low on another
    # Or: low called on one FX, then low on another
    m1, ck1, t1, qj1, qy1, r1 = c1
    m2, ck2, t2, qj2, qy2, r2 = c2
    
    if ck1 != ck2:  # Different fractals - this is a comparison
        pair_count += 1
        op = ">" if r1 > r2 else ("<" if r1 < r2 else "==")
        result_bool = r1 > r2
        
        # Only show pairs involving key fractals  
        if ck1 in key_cks or ck2 in key_cks:
            print(f"  PAIR[{pair_count:3d}]: FX[{ck1:3d}].{m1}={r1:8.1f} vs FX[{ck2:3d}].{m2}={r2:8.1f} "
                  f"({r1:.1f} {op} {r2:.1f}) [{t1}@{ck1} vs {t2}@{ck2}]")
        i += 2
    else:
        i += 1

# Show the unique qj/qy used
qj_qy_used = set()
for method, ck, ftype, qj, qy, result in call_log:
    qj_qy_used.add((qj, qy))
print(f"\n=== qj/qy combinations used: {qj_qy_used} ===")
