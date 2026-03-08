"""Instrument TZXL to capture line_bad assignment from pyarmor."""
import sys; sys.path.insert(0, 'src')
import pandas as pd
import traceback
from chanlun.cl_interface import TZXL

# Store ALL TZXL init calls with their call stacks
orig_init = TZXL.__init__
tzxl_log = []

def traced_init(self, bh_direction, line, pre_line, line_bad, done):
    orig_init(self, bh_direction, line, pre_line, line_bad, done)
    if line is not None and line_bad:
        # Capture where line_bad=True comes from
        stack = traceback.extract_stack()
        # Get the immediate caller
        caller = stack[-2] if len(stack) >= 2 else None
        entry = {
            'line_index': line.index,
            'line_type': line.type,
            'bh_direction': bh_direction,
            'line_bad': line_bad,
            'caller_file': caller.filename if caller else 'unknown',
            'caller_line': caller.lineno if caller else 0,
            'caller_name': caller.name if caller else 'unknown',
        }
        tzxl_log.append(entry)

TZXL.__init__ = traced_init

from chanlun.cl_pyarmor import CL as CL_P
df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')
cd_p = CL_P('BTC/USDT', '60m')
cd_p.process_klines(df)

print(f"=== TZXL creations with line_bad=True: {len(tzxl_log)} ===")
for entry in tzxl_log:
    print(f"  bi[{entry['line_index']}] {entry['line_type']} dir={entry['bh_direction']} "
          f"caller={entry['caller_name']}:{entry['caller_line']}")
