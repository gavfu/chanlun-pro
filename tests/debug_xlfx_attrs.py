"""
Explore XLFX object from pyarmor to understand its attributes.
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

df = pd.read_parquet("tests/test_data/BTC_USDT_5m_1000.parquet")
cl = CL_Pyarmor("test", "5m", config)
cl.process_klines(df)
bis = cl.get_bis()

# Get relevant BIs for up from bi[3]
rel_bis = [b for b in bis[3:] if b.type == "down"][:4]
print(f"Lines: {[b.index for b in rel_bis]}")

result = cl._xd_cal_line_xlfx(rel_bis, "ding", "no_bh")
print(f"\nResult type: {type(result)}")
if result is not None:
    print(f"Result attrs: {[a for a in dir(result) if not a.startswith('__')]}")
    # Check common attributes
    for attr in ['type', '_type', 'xl', 'xls', 'done', 'is_line_bad', 'high', 'low', 'val', 'index', 'lines']:
        if hasattr(result, attr):
            val = getattr(result, attr)
            if attr == 'xl':
                print(f"  xl type: {type(val)}")
                if val is not None:
                    xl_attrs = [a for a in dir(val) if not a.startswith('__')]
                    print(f"  xl attrs: {xl_attrs}")
                    for xa in ['lines', 'max', 'min', 'line_bad', 'done']:
                        if hasattr(val, xa):
                            xv = getattr(val, xa)
                            if xa == 'lines':
                                print(f"  xl.lines: {[l.index for l in xv] if xv else None}")
                            else:
                                print(f"  xl.{xa}: {xv}")
            elif attr == 'xls':
                print(f"  xls({len(val)}): types={[type(x).__name__ for x in val]}")
                for j, x in enumerate(val):
                    if x is not None and hasattr(x, 'lines'):
                        print(f"    xls[{j}].lines: {[l.index for l in x.lines]}")
            else:
                print(f"  {attr}: {val}")
else:
    print("Result is None")

# Also try with 3 lines
result3 = cl._xd_cal_line_xlfx(rel_bis[:3], "ding", "no_bh")
print(f"\nWith 3 lines: result={result3 is not None}")
if result3 is not None:
    print(f"  xl.lines: {[l.index for l in result3.xl.lines]}")
    print(f"  is_line_bad: {result3.is_line_bad}")
