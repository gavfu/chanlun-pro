"""
Explore the tuple returned by pyarmor's _xd_cal_line_xlfx.
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
rel_bis = [b for b in bis[3:] if b.type == "down"]

print("BTC5m up bi[3], ding FX:")
for n in range(3, min(10, len(rel_bis) + 1)):
    lines = rel_bis[:n]
    
    result_nobh = cl._xd_cal_line_xlfx(lines, "ding", "no_bh")
    result_bh = cl._xd_cal_line_xlfx(lines, "ding", "bh")
    
    nobh_str = "None"
    if result_nobh is not None:
        # It's a tuple - let's see what's in it
        nobh_str = f"tuple({len(result_nobh)}): {result_nobh}"
    
    bh_str = "None"
    if result_bh is not None:
        bh_str = f"tuple({len(result_bh)}): {result_bh}"
    
    print(f"  n={n:2d} lines=[{','.join(str(l.index) for l in lines)}]")
    print(f"       no_bh: {nobh_str}")
    print(f"       bh:    {bh_str}")
    print()
