"""
Check if pyarmor's _xd_cal_line_xlfx creates fresh XLFX objects each time,
or if it returns references to objects that get mutated later.
Test by calling the method, recording the id, and checking is_line_bad
before and after subsequent calls.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_pyarmor import CL as CL_Pyarmor

config = {
    "bi_type": "bi_type_old", "fx_qj": "fx_qj_k", "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes", "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1", "xd_bzh": "xd_bzh_no",
    "xd_zs_max_lines_split": 11, "xd_allow_split_no_highlow": 1,
    "xd_allow_split_zs_kz": 0, "xd_allow_split_zs_more_line": 1,
    "xd_allow_split_zs_no_direction": 1,
}

# BTC60 down[28] - the key divergent case
df60 = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cl60 = CL_Pyarmor("test", "60m", config)
cl60.process_klines(df60)
bis60 = cl60.get_bis()

up_bis = [b for b in bis60[28:] if b.type == "up"]

# Call with n=7 (first time FX appears)
lines7 = up_bis[:7]
result7 = cl60._xd_cal_line_xlfx(lines7, "di", "no_bh")
tzxls7, xlfxs7 = result7
fx7 = xlfxs7[0]
print(f"First call n=7:")
print(f"  FX id={id(fx7)} bi[{','.join(str(l.index) for l in fx7.xl.lines)}] "
      f"is_line_bad={fx7.is_line_bad} xl.line_bad={fx7.xl.line_bad}")

# Call again with n=8
lines8 = up_bis[:8]
result8 = cl60._xd_cal_line_xlfx(lines8, "di", "no_bh")
tzxls8, xlfxs8 = result8
fx8 = xlfxs8[0]
print(f"\nSecond call n=8:")
print(f"  FX id={id(fx8)} bi[{','.join(str(l.index) for l in fx8.xl.lines)}] "
      f"is_line_bad={fx8.is_line_bad} xl.line_bad={fx8.xl.line_bad}")

# Check if first FX was mutated
print(f"\nFirst FX after second call:")
print(f"  FX id={id(fx7)} is_line_bad={fx7.is_line_bad}")
print(f"  Same object? {id(fx7) == id(fx8)}")

# Now do BTC5m for comparison
df5m = pd.read_parquet("tests/test_data/BTC_USDT_5m_1000.parquet")
cl5m = CL_Pyarmor("test", "5m", config)
cl5m.process_klines(df5m)
bis5m = cl5m.get_bis()

down_bis_5m = [b for b in bis5m[3:] if b.type == "down"]

# Call with n=3 (first time FX appears)
lines3_5m = down_bis_5m[:3]
result3_5m = cl5m._xd_cal_line_xlfx(lines3_5m, "ding", "no_bh")
tzxls3_5m, xlfxs3_5m = result3_5m
fx3_5m = xlfxs3_5m[0]
print(f"\nBTC5m First call n=3:")
print(f"  FX id={id(fx3_5m)} bi[{','.join(str(l.index) for l in fx3_5m.xl.lines)}] "
      f"is_line_bad={fx3_5m.is_line_bad} xl.line_bad={fx3_5m.xl.line_bad}")

# Now let's do the definitive test: call _xd_cal_line_xlfx on a FRESH CL instance
# to rule out any state pollution
print("\n\n=== FRESH CL INSTANCES ===")
cl60_fresh = CL_Pyarmor("test2", "60m", config)
cl60_fresh.process_klines(df60)
bis60_fresh = cl60_fresh.get_bis()

# BTC60 down[28] - call with up BIs
up_bis_fresh = [b for b in bis60_fresh[28:] if b.type == "up"]
for n in [7, 8, 9]:
    lines = up_bis_fresh[:n]
    result = cl60_fresh._xd_cal_line_xlfx(lines, "di", "no_bh")
    if result and result[1]:
        for fx in result[1]:
            fx_bis = ",".join(str(l.index) for l in fx.xl.lines)
            print(f"  FRESH n={n} FX@bi[{fx_bis}] is_line_bad={fx.is_line_bad} xl.line_bad={fx.xl.line_bad}")
