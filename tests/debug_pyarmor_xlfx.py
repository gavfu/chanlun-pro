"""
Call pyarmor's _xd_cal_line_xlfx directly for BTC5m up bi[3] and BTC60 up bi[39]
with various line counts to trace the incremental behavior.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor

config = {
    "bi_type": "bi_type_old", "fx_qj": "fx_qj_k", "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes", "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1", "xd_bzh": "xd_bzh_no",
    "xd_zs_max_lines_split": 11, "xd_allow_split_no_highlow": 1,
    "xd_allow_split_zs_kz": 0, "xd_allow_split_zs_more_line": 1,
    "xd_allow_split_zs_no_direction": 1,
}

def test_xd_cal_xlfx(name, file, freq, start_bi, xd_type, max_lines=12):
    df = pd.read_parquet(f"tests/test_data/{file}")
    
    cl_p = CL_Pyarmor(name, freq, config)
    cl_p.process_klines(df)
    bis_p = cl_p.get_bis()
    
    tzxl_bi_type = "down" if xd_type == "up" else "up"
    target_fx = "ding" if xd_type == "up" else "di"
    
    # Collect the relevant BIs (same direction)
    rel_bis = [b for b in bis_p[start_bi:] if b.type == tzxl_bi_type]
    
    print(f"\n{'='*80}")
    print(f"  {name} {xd_type} bi[{start_bi}]")
    print(f"  Relevant bis: {[b.index for b in rel_bis[:max_lines]]}")
    
    # Call _xd_cal_line_xlfx with increasing numbers of lines
    for n in range(3, min(max_lines + 1, len(rel_bis) + 1)):
        lines = rel_bis[:n]
        
        # no_bh mode
        result_nobh = cl_p._xd_cal_line_xlfx(lines, target_fx, 'no_bh')
        # bh mode
        result_bh = cl_p._xd_cal_line_xlfx(lines, target_fx, 'bh')
        
        nobh_str = "None"
        if result_nobh is not None:
            # result is XLFX object
            fx = result_nobh
            # Get the key bi from the FX
            bis_in_xl = fx.xl.lines if hasattr(fx, 'xl') and fx.xl else []
            bis_str = ",".join([str(l.index) for l in bis_in_xl]) if bis_in_xl else "?"
            bad = fx.is_line_bad if hasattr(fx, 'is_line_bad') else "?"
            nobh_str = f"bi[{bis_str}] bad={bad}"
        
        bh_str = "None"
        if result_bh is not None:
            fx = result_bh
            bis_in_xl = fx.xl.lines if hasattr(fx, 'xl') and fx.xl else []
            bis_str = ",".join([str(l.index) for l in bis_in_xl]) if bis_in_xl else "?"
            bad = fx.is_line_bad if hasattr(fx, 'is_line_bad') else "?"
            bh_str = f"bi[{bis_str}] bad={bad}"
        
        print(f"  lines={n:2d} ({[b.index for b in lines]})  no_bh={nobh_str:30s}  bh={bh_str}")

test_xd_cal_xlfx("BTC5m", "BTC_USDT_5m_1000.parquet", "5m", 3, "up", 8)
test_xd_cal_xlfx("BTC60", "BTC_USDT_60m_1000.parquet", "60m", 39, "up", 8)
test_xd_cal_xlfx("BTC60", "BTC_USDT_60m_1000.parquet", "60m", 28, "down", 10)
