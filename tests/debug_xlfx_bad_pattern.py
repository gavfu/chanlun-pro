"""
Check more 'di' type FX cases across all datasets to see if is_line_bad 
patterns differ between ding and di.
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

original_xlfx = CL_Pyarmor._xd_cal_line_xlfx

all_fx_results = []

def traced_xlfx(self, lines, fx_type='ding', bh_type='no_bh', *args, **kwargs):
    result = original_xlfx(self, lines, fx_type, bh_type, *args, **kwargs)
    
    if result and result[1] and bh_type == 'no_bh':
        for fx in result[1]:
            if fx.xl.line_bad:  # Only interested in cases where TZXL is bad
                fx_bis = ",".join(str(l.index) for l in fx.xl.lines)
                all_fx_results.append({
                    'fx_type': fx_type,
                    'is_line_bad': fx.is_line_bad,
                    'xl_line_bad': fx.xl.line_bad,
                    'fx_bis': fx_bis,
                    'n_lines': len(lines),
                })
    
    return result

CL_Pyarmor._xd_cal_line_xlfx = traced_xlfx

datasets = [
    ("BTCd", "tests/test_data/BTC_USDT_d_500.parquet", "d"),
    ("ETH60", "tests/test_data/ETH_USDT_60m_1000.parquet", "60m"),
    ("BTC60", "tests/test_data/BTC_USDT_60m_1000.parquet", "60m"),
    ("BTC5m", "tests/test_data/BTC_USDT_5m_1000.parquet", "5m"),
    ("ETH5m", "tests/test_data/ETH_USDT_5m_1000.parquet", "5m"),
]

for name, path, freq in datasets:
    all_fx_results.clear()
    df = pd.read_parquet(path)
    cl = CL_Pyarmor("test", freq, config)
    cl.process_klines(df)
    
    # Deduplicate by (fx_type, fx_bis)
    seen = set()
    unique_results = []
    for r in all_fx_results:
        key = (r['fx_type'], r['fx_bis'], r['is_line_bad'])
        if key not in seen:
            seen.add(key)
            unique_results.append(r)
    
    # Count
    ding_bad_true = sum(1 for r in unique_results if r['fx_type'] == 'ding' and r['is_line_bad'])
    ding_bad_false = sum(1 for r in unique_results if r['fx_type'] == 'ding' and not r['is_line_bad'])
    di_bad_true = sum(1 for r in unique_results if r['fx_type'] == 'di' and r['is_line_bad'])
    di_bad_false = sum(1 for r in unique_results if r['fx_type'] == 'di' and not r['is_line_bad'])
    
    print(f"\n{name}: {len(unique_results)} unique FXs with xl.line_bad=True")
    print(f"  ding: is_line_bad=True: {ding_bad_true}, is_line_bad=False: {ding_bad_false}")
    print(f"  di:   is_line_bad=True: {di_bad_true}, is_line_bad=False: {di_bad_false}")
    
    # Show the False cases
    for r in unique_results:
        if not r['is_line_bad']:
            print(f"    → {r['fx_type']} FX@bi[{r['fx_bis']}] is_line_bad=False (xl.line_bad=True)")
