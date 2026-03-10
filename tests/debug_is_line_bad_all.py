"""
Comprehensive investigation of is_line_bad logic in pyarmor's _xd_cal_line_xlfx.

For EVERY XLFX returned by _xd_cal_line_xlfx (no_bh mode), dump:
- XLFX: type, is_line_bad, fx_high, fx_low, done
- TZXL (xl): index, line_bad, max, min, direction, lines (which BIs), line count
- xls (all TZXLs in the FX): their line_bad, max, min
- qk (gap) info
- The ACTUAL BIs involved: their type/high/low

Goal: find what distinguishes is_line_bad=False cases from is_line_bad=True cases 
      when xl.line_bad=True for both.
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

datasets = [
    ("BTCd", "tests/test_data/BTC_USDT_d_500.parquet"),
    ("ETH60", "tests/test_data/ETH_USDT_60m_1000.parquet"),
    ("BTC60", "tests/test_data/BTC_USDT_60m_1000.parquet"),
    ("BTC5m", "tests/test_data/BTC_USDT_5m_1000.parquet"),
    ("ETH5m", "tests/test_data/ETH_USDT_5m_1000.parquet"),
]

for ds_name, ds_path in datasets:
    df = pd.read_parquet(ds_path)
    cl = CL_Pyarmor("test", "60m", config)
    cl.process_klines(df)
    bis = cl.get_bis()
    
    print(f"\n{'='*80}")
    print(f"=== {ds_name}: {len(bis)} BIs ===")
    print(f"{'='*80}")
    
    # For various starting positions, compute xlfx for both ding and di
    # Use ALL possible starting positions from bi[0] to bi[-3]
    for start_bi_idx in range(0, len(bis) - 2):
        for fx_type in ['ding', 'di']:
            # Build lines from start_bi_idx to end
            lines = bis[start_bi_idx:]
            if len(lines) < 3:
                continue
            
            result = cl._xd_cal_line_xlfx(lines, fx_type=fx_type, bh_type='no_bh')
            tzxls, xlfxs = result
            
            for fx in xlfxs:
                if not fx.xl.line_bad:
                    continue  # We only care about cases where xl.line_bad=True
                
                # Get the BI indices
                xl_bi_indices = [l.index for l in fx.xl.lines]
                xls_info = []
                for x in fx.xls:
                    x_bis = [l.index for l in x.lines]
                    xls_info.append(f"TZXL(bis={x_bis}, bad={x.line_bad}, max={x.max:.1f}, min={x.min:.1f})")
                
                # Only print the FIRST FX found per (start_bi_idx, fx_type) to limit output
                marker = "*** FALSE ***" if not fx.is_line_bad else ""
                print(f"\n  start={start_bi_idx} {fx_type}: FX@bi[{xl_bi_indices}] "
                      f"is_line_bad={fx.is_line_bad} {marker}")
                print(f"    xl: bad={fx.xl.line_bad} max={fx.xl.max:.1f} min={fx.xl.min:.1f} "
                      f"nlines={len(fx.xl.lines)} done={fx.xl.done}")
                print(f"    FX: high={fx.fx_high:.1f} low={fx.fx_low:.1f} done={fx.done} qk={fx.qk}")
                for i, xi in enumerate(xls_info):
                    print(f"    xls[{i}]: {xi}")
                break  # Only first FX with xl.line_bad=True
