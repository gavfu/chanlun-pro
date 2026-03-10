#!/usr/bin/env python3
"""Parse the condensed pyarmor trace to understand the algorithm."""
import sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pandas as pd
from chanlun.cl_interface import BI, FX
from chanlun.cl_pyarmor import CL as CLPyarmor

df = pd.read_parquet('tests/test_data/ETH_USDT_5m_1000.parquet')
original = CLPyarmor._bi_special_bi_split

condensed = []
def summarize_val(v):
    if isinstance(v, FX):
        return f"FX({v.k.index},{v.type},{v.val:.2f},ki={v.k.k_index})"
    if isinstance(v, BI):
        return f"BI({v.start.k.index}->{v.end.k.index},{v.type})"
    if isinstance(v, list):
        if len(v) <= 5:
            return f"[{', '.join(summarize_val(x) for x in v)}]"
        return f"[len={len(v)}]"
    if isinstance(v, (int, float, str, bool)) or v is None:
        return repr(v)
    return type(v).__name__

last = {}
def tracer(frame, event, arg):
    if frame.f_code is not original.__code__:
        return tracer
    bi = frame.f_locals.get('bi')
    if bi is None or (bi.start.k.index, bi.end.k.index) != (191, 213):
        return tracer
    if event == 'line':
        ln = frame.f_lineno
        v160 = frame.f_locals.get('_var_var_160')
        v159 = frame.f_locals.get('_var_var_159')
        v5 = frame.f_locals.get('_var_var_5')

        if ln >= 2530:
            # Split selection phase - capture key variables
            v162 = frame.f_locals.get('_var_var_162')
            v163 = frame.f_locals.get('_var_var_163')
            v164 = frame.f_locals.get('_var_var_164')
            v165 = frame.f_locals.get('_var_var_165')
            v166 = frame.f_locals.get('_var_var_166')
            v167 = frame.f_locals.get('_var_var_167')
            v168 = frame.f_locals.get('_var_var_168')
            s = f"L{ln}: _160={v160}, fx162={summarize_val(v162) if v162 else None}, fx163={summarize_val(v163) if v163 else None}"
            s += f", _164={v164}, _165={v165}, _166={v166}, _167={v167}"
            if v168:
                s += f", fx168={summarize_val(v168)}"
            condensed.append(s)
        else:
            cur = (v5, v159, v160)
            if cur != last.get('counter'):
                condensed.append(f"L{ln}: _5={v5}, _159={v159}, _160={v160}")
                last['counter'] = cur
    return tracer

sys.settrace(tracer)
try:
    cd = CLPyarmor("ETH/USDT", "5m")
    cd.process_klines(df)
finally:
    sys.settrace(None)

for line in condensed:
    print(line)
