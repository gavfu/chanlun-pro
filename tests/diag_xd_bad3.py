"""Detailed trace of line_bad=True in pyarmor, with context."""
import sys; sys.path.insert(0, 'src')
import pandas as pd
import traceback
from chanlun.cl_interface import TZXL

# Track containment decisions
orig_init = TZXL.__init__
containment_log = []
call_count = [0]

def traced_init(self, bh_direction, line, pre_line, line_bad, done):
    object.__setattr__(self, '_tracking', False)
    orig_init(self, bh_direction, line, pre_line, line_bad, done)
    object.__setattr__(self, '_tracking', True)
    object.__setattr__(self, '_id', call_count[0])
    call_count[0] += 1

def traced_setattr(self, name, value):
    tracking = getattr(self, '_tracking', False)
    old_val = getattr(self, name, None) if tracking else None
    object.__setattr__(self, name, value)
    
    if name == 'line_bad' and value == True and tracking:
        if old_val != True:  # Only log first time
            line_idx = self.line.index if self.line else '?'
            # Get full stack  
            stack = traceback.extract_stack()
            # Find the pyarmor frame
            for i, frame in enumerate(stack):
                if 'cl_pyarmor' in frame.filename or 'frozen cl' in frame.filename.lower():
                    containment_log.append({
                        'line_index': line_idx,
                        'pyarmor_line': frame.lineno,
                        'func': frame.name,
                        'max': self.max,
                        'min': self.min,
                        'lines': [l.index for l in self.lines],
                        'id': getattr(self, '_id', '?'),
                    })
                    break

TZXL.__init__ = traced_init
TZXL.__setattr__ = traced_setattr

from chanlun.cl_pyarmor import CL as CL_P
df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')
cd_p = CL_P('BTC/USDT', '60m')
cd_p.process_klines(df)

# Deduplicate: group by (line_index, id) and keep unique
seen = set()
unique_log = []
for entry in containment_log:
    key = (entry['line_index'], entry['id'])
    if key not in seen:
        seen.add(key)
        unique_log.append(entry)

print(f"=== Unique line_bad=True events: {len(unique_log)} ===")
for entry in unique_log:
    print(f"  bi[{entry['line_index']}] id={entry['id']} at line {entry['pyarmor_line']} "
          f"max={entry['max']:.1f} min={entry['min']:.1f} lines=bi{entry['lines']}")

# Now let's understand the containment rule
# For XD[0]'s TZXL sequence, bi[9] has line_bad=True
# Let me check: what this means for the sequence fractal detection
print("\n=== Key question: Does line_bad affect sequence fractal detection? ===")
xd0 = cd_p.xds[0]
print("XD[0] di_fx (the fractal that ends the first XD):")
dfx = xd0.di_fx
print(f"  type={dfx.type} is_line_bad={dfx.is_line_bad}")
print(f"  xls:")
for j, xl in enumerate(dfx.xls):
    if xl:
        print(f"    [{j}] max={xl.max:.1f} min={xl.min:.1f} bad={xl.line_bad}")
