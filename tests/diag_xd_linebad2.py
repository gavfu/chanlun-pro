"""Intercept _xd_cal_line_xlfx to understand how line_bad is set.
 
Key hypothesis: line_bad is set based on comparison with the MOST RECENT 
non-bad element, not the immediately preceding element.
"""
import sys; sys.path.insert(0, 'src')
import pandas as pd
from chanlun.cl_pyarmor import CL as CL_P
from chanlun.cl_interface import TZXL, XLFX

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')

# Monkey-patch TZXL to track line_bad changes
original_setattr = TZXL.__setattr__
_tzxl_counter = [0]

def tracked_setattr(self, name, value):
    if name == 'line_bad' and value == True:
        import traceback
        lines = traceback.format_stack()
        # Find the interesting frame
        for line in lines:
            if 'frozen cl' in line or '_xd_cal' in line:
                bi_info = f"bi[{self.line.index}]" if hasattr(self, 'line') and self.line else "?"
                lines_info = ""
                if hasattr(self, 'lines') and self.lines:
                    lines_info = f" lines=[{','.join(str(l.index) for l in self.lines)}]"
                print(f"  SET line_bad=True on {bi_info}{lines_info} at: {line.strip()}")
                break
    original_setattr(self, name, value)

TZXL.__setattr__ = tracked_setattr

# Now run pyarmor and look at the state JUST BEFORE and AFTER inclusion processing
# for XD[0] (DOWN from bi[2])
cd_p = CL_P('BTC/USDT', '60m')
cd_p.process_klines(df)
xds = cd_p.xds

print("\n=== PYARMOR XDs ===")
for x in xds:
    print(f"  {x.type} bi[{x.start_line.index}]→bi[{x.end_line.index}]")

print("\n=== XD[0] TZXL detail ===")
xd0 = xds[0]
for i, t in enumerate(xd0.tzxls):
    lines_str = ','.join(str(l.index) for l in t.lines)
    pre = f"pre={t.pre_line.index}" if t.pre_line else "pre=None"
    print(f"  [{i}] bi[{lines_str}] max={t.max:.1f} min={t.min:.1f} bad={t.line_bad} {pre}")
    # Check containment with previous non-bad
    if i > 0:
        for j in range(i-1, -1, -1):
            prev_t = xd0.tzxls[j]
            if not prev_t.line_bad:
                # Check containment
                old_h, old_l = prev_t.max, prev_t.min
                new_h, new_l = t.max, t.min
                o_c_n = old_h >= new_h and old_l <= new_l
                n_c_o = new_h >= old_h and new_l <= old_l
                if o_c_n or n_c_o:
                    direction = "OLD⊃NEW" if o_c_n else "NEW⊃OLD"
                    print(f"       -> contained by non-bad [{j}] ({direction})")
                else:
                    print(f"       -> NOT contained by non-bad [{j}]")
                break

TZXL.__setattr__ = original_setattr
