"""Track line_bad attribute changes on TZXL objects."""
import sys; sys.path.insert(0, 'src')
import pandas as pd
import traceback
from chanlun.cl_interface import TZXL

# Override __setattr__ to track line_bad changes
orig_init = TZXL.__init__

all_tzxl_instances = []

def traced_init(self, bh_direction, line, pre_line, line_bad, done):
    # Allow normal init
    object.__setattr__(self, '_tracking', False)
    orig_init(self, bh_direction, line, pre_line, line_bad, done)
    object.__setattr__(self, '_tracking', True)
    all_tzxl_instances.append(self)

orig_setattr = TZXL.__setattr__ if hasattr(TZXL, '__setattr__') else object.__setattr__

def traced_setattr(self, name, value):
    object.__setattr__(self, name, value)
    if name == 'line_bad' and value == True and getattr(self, '_tracking', False):
        line_idx = self.line.index if self.line else '?'
        stack = traceback.extract_stack()
        # Find the caller from pyarmor
        for frame in reversed(stack):
            if 'cl_pyarmor' in frame.filename or 'cl_analyse' in frame.filename:
                print(f"line_bad=True set on bi[{line_idx}] at {frame.filename}:{frame.lineno} in {frame.name}")
                break
        else:
            caller = stack[-2] if len(stack) >= 2 else None
            if caller:
                print(f"line_bad=True set on bi[{line_idx}] at {caller.filename}:{caller.lineno} in {caller.name}")

TZXL.__init__ = traced_init
TZXL.__setattr__ = traced_setattr

from chanlun.cl_pyarmor import CL as CL_P
df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')
cd_p = CL_P('BTC/USDT', '60m')
cd_p.process_klines(df)

# Check final state
print(f"\n=== Final TZXL states with line_bad=True ===")
for xl in all_tzxl_instances:
    if xl.line_bad and xl.line is not None:
        lines = [l.index for l in xl.lines]
        print(f"  bi{lines} dir={xl.bh_direction} max={xl.max:.1f} min={xl.min:.1f}")
