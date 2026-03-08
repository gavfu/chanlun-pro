"""Instrument pyarmor to trace XD construction."""
import sys; sys.path.insert(0, 'src')
import pandas as pd
from chanlun.cl_pyarmor import CL as CL_P
from chanlun.cl_interface import TZXL, XLFX, XD

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')

# Monkey-patch TZXL __init__ to track creation
original_tzxl_init = TZXL.__init__
tzxl_creations = []
def traced_tzxl_init(self, bh_direction, line, pre_line, line_bad, done):
    original_tzxl_init(self, bh_direction, line, pre_line, line_bad, done)
    if line is not None:
        tzxl_creations.append({
            'bh_direction': bh_direction,
            'line_index': line.index,
            'line_type': line.type,
            'line_start': line.start.k.index,
            'line_end': line.end.k.index,
            'max': self.max,
            'min': self.min,
            'line_bad': line_bad,
        })

TZXL.__init__ = traced_tzxl_init

# Monkey-patch XLFX __init__
original_xlfx_init = XLFX.__init__
xlfx_creations = []
def traced_xlfx_init(self, _type, xl, xls, done=True):
    original_xlfx_init(self, _type, xl, xls, done)
    xlfx_creations.append({
        'type': _type,
        'xl_max': xl.max,
        'xl_min': xl.min,
        'xl_lines': [l.index for l in xl.lines] if xl.lines else [],
        'done': done,
        'high': self.high,
        'low': self.low,
    })

XLFX.__init__ = traced_xlfx_init

# Monkey-patch XD __init__
original_xd_init = XD.__init__
xd_creations = []
def traced_xd_init(self, start, end, start_line, end_line=None, _type=None,
                   ding_fx=None, di_fx=None, index=0, default_zs_type=None):
    original_xd_init(self, start, end, start_line, end_line, _type,
                     ding_fx, di_fx, index, default_zs_type)
    xd_creations.append({
        'type': _type,
        'start_line_type': start_line.type if start_line else None,
        'start_line_idx': start_line.index if start_line else None,
        'end_line_type': end_line.type if end_line else None,
        'end_line_idx': end_line.index if end_line else None,
        'index': index,
    })

XD.__init__ = traced_xd_init

cd_p = CL_P('BTC/USDT', '60m')
cd_p.process_klines(df)

print("=== TZXL Creations ===")
for i, t in enumerate(tzxl_creations):
    print(f"  [{i}] dir={t['bh_direction']} line=bi[{t['line_index']}]({t['line_type']}) "
          f"{t['line_start']}→{t['line_end']} max={t['max']:.1f} min={t['min']:.1f} "
          f"bad={t['line_bad']}")

print(f"\n=== XLFX Creations ({len(xlfx_creations)}) ===")
for i, x in enumerate(xlfx_creations):
    print(f"  [{i}] {x['type']} high={x['high']:.1f} low={x['low']:.1f} "
          f"xl_max={x['xl_max']:.1f} xl_min={x['xl_min']:.1f} "
          f"xl_lines=bi{x['xl_lines']} done={x['done']}")

print(f"\n=== XD Creations ({len(xd_creations)}) ===")
for i, x in enumerate(xd_creations):
    print(f"  [{i}] {x['type']} start_line=bi[{x['start_line_idx']}]({x['start_line_type']}) "
          f"end_line=bi[{x['end_line_idx']}]({x['end_line_type']}) idx={x['index']}")
