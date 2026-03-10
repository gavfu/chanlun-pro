"""
Ultra-targeted tracer: capture cross-counting variables for ETH5m BI(191→213).
Uses call-level filtering for performance.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_interface import *
from chanlun.cl_pyarmor import CL as CL_Pyarmor

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), 'test_data', 'ETH_USDT_5m_1000.parquet'))

results = []
call_count = [0]

def line_tracer(frame, event, arg):
    if event != 'line':
        return line_tracer
    
    loc = frame.f_locals
    lineno = frame.f_lineno
    
    if '_var_var_160' not in loc:
        return line_tracer
    
    v160 = loc.get('_var_var_160')
    v159 = loc.get('_var_var_159')
    v5 = loc.get('_var_var_5')
    
    # Only record transitions
    if results and results[-1].get('_160') == v160 and results[-1].get('_159') == v159 and results[-1].get('_5') == v5:
        return line_tracer
    
    entry = {'line': lineno, '_5': v5, '_159': v159, '_160': v160}
    for vn in [149, 155, 156, 158, 161]:
        key = f'_var_var_{vn}'
        val = loc.get(key)
        if val is not None:
            if isinstance(val, list):
                entry[f'_{vn}'] = [round(x, 2) if isinstance(x, float) else x for x in val]
            elif isinstance(val, (int, float)):
                entry[f'_{vn}'] = val
    results.append(entry)
    return line_tracer

def call_tracer(frame, event, arg):
    if event == 'call':
        loc = frame.f_locals
        for key, val in loc.items():
            if isinstance(val, BI):
                try:
                    if val.start.k.index == 191 and val.end.k.index == 213:
                        call_count[0] += 1
                        return line_tracer
                except:
                    pass
    return call_tracer

sys.settrace(call_tracer)
cl_p = CL_Pyarmor('ETH/USDT', '5m')
cl_p.process_klines(df)
sys.settrace(None)

print(f"Matched {call_count[0]} calls, captured {len(results)} transitions")

for i, p in enumerate(results):
    row = f"[{i:3d}] L{p['line']}: "
    for k in ['_5', '_149', '_155', '_156', '_158', '_159', '_160', '_161']:
        if k in p:
            row += f"{k}={p[k]} "
    print(row)
