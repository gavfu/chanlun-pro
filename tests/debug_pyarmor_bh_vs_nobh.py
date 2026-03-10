"""
Deep-dive into pyarmor's _xd_get_up_line_tzxl_info result structure.
Focus on the di/bh_di sub-dicts to understand what FX data is returned.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_pyarmor import CL

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
xds = cl.get_xds()

# Call _xd_get_up_line_tzxl_info directly with appropriate arguments
# We need to figure out what base_lines and up_lines look like
# From the trace, base_lines are BIs, up_lines are XDs

# For the segment that starts at bi[28], the up_lines would be XDs above this level
# Let's call it in a simulated manner

# First, let's look at what result structure a sub-dict contains
# by calling _xd_cal_line_xlfx directly and examining its output
print("=== Calling _xd_cal_line_xlfx with bi{28..42} ===")

# For "down" segment starting at bi[28], looking for DI FX in UP BIs
# lines passed should be the BIs starting from bi[28]
subset_bh = bis[28:42]
tzxls_bh, xlfxs_bh = cl._xd_cal_line_xlfx(subset_bh, 'di', 'bh')
print(f"\nbh_type='bh' → TZXLs ({len(tzxls_bh)}):")
for i, t in enumerate(tzxls_bh):
    print(f"  [{i}]: max={t.max} min={t.min} bad={t.line_bad} lines={[l.index for l in t.lines]}")
print(f"XLFXs ({len(xlfxs_bh)}):")
for i, f in enumerate(xlfxs_bh):
    print(f"  [{i}]: type={f.type} bad={f.is_line_bad} done={f.done} xl_lines={[l.index for l in f.xl.lines]}")
    print(f"       xl.max={f.xl.max} xl.min={f.xl.min}")
    print(f"       fx_high={f.fx_high} fx_low={f.fx_low}")
    print(f"       qk={f.qk}")
    if hasattr(f, 'xls') and f.xls:
        for j, x in enumerate(f.xls):
            print(f"       xls[{j}]: lines={[l.index for l in x.lines]} max={x.max} min={x.min} bad={x.line_bad}")

tzxls_no, xlfxs_no = cl._xd_cal_line_xlfx(subset_bh, 'di', 'no_bh')
print(f"\nbh_type='no_bh' → TZXLs ({len(tzxls_no)}):")
for i, t in enumerate(tzxls_no):
    print(f"  [{i}]: max={t.max} min={t.min} bad={t.line_bad} lines={[l.index for l in t.lines]}")
print(f"XLFXs ({len(xlfxs_no)}):")
for i, f in enumerate(xlfxs_no):
    print(f"  [{i}]: type={f.type} bad={f.is_line_bad} done={f.done} xl_lines={[l.index for l in f.xl.lines]}")
    print(f"       xl.max={f.xl.max} xl.min={f.xl.min}")
    if hasattr(f, 'xls') and f.xls:
        for j, x in enumerate(f.xls):
            print(f"       xls[{j}]: lines={[l.index for l in x.lines]} max={x.max} min={x.min} bad={x.line_bad}")

# Also check for DING FX
print("\n=== DING FX ===")
tzxls_bh_ding, xlfxs_bh_ding = cl._xd_cal_line_xlfx(subset_bh, 'ding', 'bh')
print(f"\nbh_type='bh' DING → XLFXs ({len(xlfxs_bh_ding)}):")
for i, f in enumerate(xlfxs_bh_ding):
    print(f"  [{i}]: type={f.type} bad={f.is_line_bad} done={f.done} xl_lines={[l.index for l in f.xl.lines]}")

tzxls_no_ding, xlfxs_no_ding = cl._xd_cal_line_xlfx(subset_bh, 'ding', 'no_bh')
print(f"\nbh_type='no_bh' DING → XLFXs ({len(xlfxs_no_ding)}):")
for i, f in enumerate(xlfxs_no_ding):
    print(f"  [{i}]: type={f.type} bad={f.is_line_bad} done={f.done} xl_lines={[l.index for l in f.xl.lines]}")


# Now let's also check ETH60 cases
print("\n\n=== ETH60 up from bi[15] (for comparison) ===")
df2 = pd.read_parquet("tests/test_data/ETH_USDT_60m_1000.parquet")
cl2 = CL("ETH60", "60m", config)
cl2.process_klines(df2)
bis2 = cl2.get_bis()

# For "up" segment starting at bi[15], looking for DING FX in DOWN BIs
subset_eth = bis2[15:25]
print(f"BIs: {[b.index for b in subset_eth]}")

tzxls_eth_bh, xlfxs_eth_bh = cl2._xd_cal_line_xlfx(subset_eth, 'ding', 'bh')
print(f"\nETH60 bh_type='bh' DING → TZXLs ({len(tzxls_eth_bh)}):")
for i, t in enumerate(tzxls_eth_bh):
    print(f"  [{i}]: max={t.max} min={t.min} bad={t.line_bad} lines={[l.index for l in t.lines]}")
print(f"XLFXs ({len(xlfxs_eth_bh)}):")
for i, f in enumerate(xlfxs_eth_bh):
    print(f"  [{i}]: type={f.type} bad={f.is_line_bad} done={f.done} xl_lines={[l.index for l in f.xl.lines]}")

tzxls_eth_no, xlfxs_eth_no = cl2._xd_cal_line_xlfx(subset_eth, 'ding', 'no_bh')
print(f"\nETH60 bh_type='no_bh' DING → TZXLs ({len(tzxls_eth_no)}):")
for i, t in enumerate(tzxls_eth_no):
    print(f"  [{i}]: max={t.max} min={t.min} bad={t.line_bad} lines={[l.index for l in t.lines]}")
print(f"XLFXs ({len(xlfxs_eth_no)}):")
for i, f in enumerate(xlfxs_eth_no):
    print(f"  [{i}]: type={f.type} bad={f.is_line_bad} done={f.done} xl_lines={[l.index for l in f.xl.lines]}")
