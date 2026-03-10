"""
Trace the split SELECTION phase (L2534+) to understand the validation conditions
for choosing (di_fx, ding_fx) split points.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_interface import *
from chanlun.cl_pyarmor import CL as CL_Pyarmor

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), 'test_data', 'ETH_USDT_5m_1000.parquet'))

trace_data = []

def line_tracer(frame, event, arg):
    if event != 'line':
        return line_tracer
    
    loc = frame.f_locals
    lineno = frame.f_lineno
    
    # Only capture the split selection phase (L2534+)
    if lineno >= 2530 and '_var_var_162' in loc:
        v162 = loc.get('_var_var_162')
        v163 = loc.get('_var_var_163')
        v168 = loc.get('_var_var_168')
        
        entry = {
            'line': lineno,
        }
        
        for vn in [162, 163, 164, 165, 166, 167, 168, 71, 72]:
            key = f'_var_var_{vn}'
            val = loc.get(key)
            if val is not None:
                if isinstance(val, (int, float)):
                    entry[f'_{vn}'] = val
                elif isinstance(val, FX):
                    entry[f'_{vn}'] = f"FX({val.k.index},{val.type},{val.val:.2f},ki={val.k.k_index})"
                elif isinstance(val, list):
                    parts = []
                    for v in val:
                        if isinstance(v, FX):
                            parts.append(f"FX({v.k.index},{v.type},{v.val:.2f})")
                        else:
                            parts.append(str(v))
                    entry[f'_{vn}'] = parts
        
        trace_data.append(entry)
    
    return line_tracer

def call_tracer(frame, event, arg):
    if event == 'call':
        loc = frame.f_locals
        for key, val in loc.items():
            if isinstance(val, BI):
                try:
                    if val.start.k.index == 191 and val.end.k.index == 213:
                        return line_tracer
                except:
                    pass
    return call_tracer

sys.settrace(call_tracer)
cl_p = CL_Pyarmor('ETH/USDT', '5m')
cl_p.process_klines(df)
sys.settrace(None)

print(f"Captured {len(trace_data)} trace points for split selection")

# Show only when lines change or key values change
prev_line = None
prev_162 = None
prev_163 = None
for i, p in enumerate(trace_data):
    line = p['line']
    v162 = p.get('_162', '')
    v163 = p.get('_163', '')
    
    if line != prev_line or v162 != prev_162 or v163 != prev_163:
        row = f"[{i:3d}] L{line}: "
        for k in sorted(p.keys()):
            if k != 'line':
                row += f"{k}={p[k]}  "
        print(row)
        prev_line = line
        prev_162 = v162
        prev_163 = v163
