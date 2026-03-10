"""
Investigate pyarmor's _xd_cal_line_xlfx by tracing TZXL construction.
Fix: handle both positional and keyword arguments for TZXL.__init__
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl import CL
from chanlun.cl_interface import TZXL

df = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
config = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
    "xd_zs_max_lines_split": 11,
    "xd_allow_split_no_highlow": 1,
    "xd_allow_split_zs_kz": 0,
    "xd_allow_split_zs_more_line": 1,
    "xd_allow_split_zs_no_direction": 1,
}

cl = CL("BTC60", "60m", config)
cl.process_klines(df)
bis = cl.get_bis()

# Trace by patching TZXL init with *args and **kwargs
original_init = TZXL.__init__
original_update = TZXL.update_maxmin

init_log = []
update_log = []

def traced_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    # Log all TZXL creations
    if hasattr(self, 'line') and hasattr(self.line, 'index'):
        init_log.append({
            'line_idx': self.line.index,
            'bh_direction': self.bh_direction,
            'max': self.max,
            'min': self.min,
            'line_bad': self.line_bad,
            'lines': [l.index for l in self.lines],
        })

def traced_update(self):
    old_max, old_min = self.max, self.min
    original_update(self)
    update_log.append({
        'lines': [l.index for l in self.lines],
        'old_max': old_max, 'old_min': old_min,
        'new_max': self.max, 'new_min': self.min,
        'bh_direction': self.bh_direction,
    })

TZXL.__init__ = traced_init
TZXL.update_maxmin = traced_update

# Call with just bi[37:40] to isolate the merge
print("=== Tracing _xd_cal_line_xlfx with bi[37:40], 'di', 'bh' ===")
init_log.clear()
update_log.clear()
subset = bis[37:40]
tzxls, xlfxs = cl._xd_cal_line_xlfx(subset, 'di', 'bh')

print(f"\nTZXL __init__ calls ({len(init_log)}):")
for log in init_log:
    print(f"  line[{log['line_idx']}]: bh_dir={log['bh_direction']} max={log['max']} min={log['min']} bad={log['line_bad']} lines={log['lines']}")

print(f"\nTZXL update_maxmin calls ({len(update_log)}):")
for log in update_log:
    print(f"  lines={log['lines']}: bh_dir={log['bh_direction']} {log['old_max']},{log['old_min']} → {log['new_max']},{log['new_min']}")

print(f"\nResult TZXLs ({len(tzxls)}):")
for i, t in enumerate(tzxls):
    print(f"  [{i}]: max={t.max} min={t.min} bad={t.line_bad} lines={[l.index for l in t.lines]} bh_dir={t.bh_direction}")

# Restore
TZXL.__init__ = original_init
TZXL.update_maxmin = original_update

# Also try calling with bh_type='no_bh' to see the difference
print()
print("=== Compare with bh_type='no_bh' ===")
subset = bis[37:40]
tzxls_nobh, xlfxs_nobh = cl._xd_cal_line_xlfx(subset, 'di', 'no_bh')
print(f"Result TZXLs ({len(tzxls_nobh)}):")
for i, t in enumerate(tzxls_nobh):
    print(f"  [{i}]: max={t.max} min={t.min} bad={t.line_bad} lines={[l.index for l in t.lines]}")

# And bh_type='bh' with 'ding'
print()
print("=== With fx_type='ding', bh_type='bh' ===")
subset = bis[37:40]
tzxls_ding, xlfxs_ding = cl._xd_cal_line_xlfx(subset, 'ding', 'bh')
print(f"Result TZXLs ({len(tzxls_ding)}):")
for i, t in enumerate(tzxls_ding):
    print(f"  [{i}]: max={t.max} min={t.min} bad={t.line_bad} lines={[l.index for l in t.lines]} bh_dir={t.bh_direction}")
