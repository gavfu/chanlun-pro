# -*- coding: utf-8 -*-
"""
Comprehensive instrumentation: capture ALL FX.high/low calls related to 
FX indices 347-370. Show caller line numbers and values.
Focus on finding WHERE end_fx setting for ding@355 happens.
"""
import os, sys, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from chanlun.cl_interface import FX

# Build a lookup of fx objects by id after they're created
_fx_index_map = {}

_orig_high = FX.high
_orig_low = FX.low

def _patched_high(self, *args, **kwargs):
    result = _orig_high(self, *args, **kwargs)
    idx = self.k.index if hasattr(self, 'k') and self.k else -1
    if 345 <= idx <= 375:
        frame = sys._getframe(1)
        lineno = frame.f_lineno
        fname = frame.f_code.co_filename
        short_fname = os.path.basename(fname)
        # Get the calling function name
        func = frame.f_code.co_name
        print(f"HIGH fx@{idx}({self.type}) = {result:.1f}  caller: {short_fname}:{lineno} in {func}")
    return result

def _patched_low(self, *args, **kwargs):
    result = _orig_low(self, *args, **kwargs)
    idx = self.k.index if hasattr(self, 'k') and self.k else -1
    if 345 <= idx <= 375:
        frame = sys._getframe(1)
        lineno = frame.f_lineno
        fname = frame.f_code.co_filename
        short_fname = os.path.basename(fname)
        func = frame.f_code.co_name
        print(f"LOW  fx@{idx}({self.type}) = {result:.1f}  caller: {short_fname}:{lineno} in {func}")
    return result

FX.high = _patched_high
FX.low = _patched_low

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "test_data", "BTC_USDT_60m_500.parquet"))

from chanlun.cl_pyarmor import CL as CL_P

print("=== Capturing all FX.high/low calls for indices 345-375 ===\n")
cd_p = CL_P("BTC/USDT", "60m")
cd_p.process_klines(df)

print(f"\n=== Result: {len(cd_p.bis)} strokes ===")
for i, bi in enumerate(cd_p.bis):
    if bi.start.k.index >= 310 or bi.end.k.index >= 310:
        print(f"bi[{i}] {bi.type} {bi.start.k.index} -> {bi.end.k.index}")
