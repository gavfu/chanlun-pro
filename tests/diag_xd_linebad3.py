"""Track ALL line_bad changes (both True and False) on bi[15] TZXL elements."""
import sys; sys.path.insert(0, 'src')
import pandas as pd
from chanlun.cl_pyarmor import CL as CL_P
from chanlun.cl_interface import TZXL
import traceback

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')

original_setattr = TZXL.__setattr__

def tracked_setattr(self, name, value):
    if name == 'line_bad':
        # Track for bi[15] specifically
        bi_idx = None
        if hasattr(self, 'line') and self.line is not None:
            bi_idx = self.line.index
        if hasattr(self, 'lines') and self.lines:
            bi_indices = [l.index for l in self.lines]
            if 15 in bi_indices:
                tb = traceback.format_stack()
                caller = ""
                for line in tb:
                    if 'frozen cl' in line:
                        caller = line.strip()
                print(f"  bi[{','.join(str(i) for i in bi_indices)}] line_bad = {value} at: {caller}")
    original_setattr(self, name, value)

TZXL.__setattr__ = tracked_setattr

cd_p = CL_P('BTC/USDT', '60m')
cd_p.process_klines(df)
xds = cd_p.xds

print(f"\n=== FINAL XD[0] ===")
xd0 = xds[0]
for i, t in enumerate(xd0.tzxls):
    lines_str = ','.join(str(l.index) for l in t.lines)
    print(f"  [{i}] bi[{lines_str}] bad={t.line_bad}")

TZXL.__setattr__ = original_setattr
