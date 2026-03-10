"""
Compare XLFX is_line_bad vs TZXL line_bad for BTC5m and BTC60.
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

def analyze(cl, bis, start_idx, direction, label):
    if direction == "up":
        fx_type = "ding"
        rel_bis = [b for b in bis[start_idx:] if b.type == "down"]
    else:
        fx_type = "di"
        rel_bis = [b for b in bis[start_idx:] if b.type == "up"]
    
    print(f"\n=== {label}: {direction} from bi[{start_idx}], {fx_type} FX ===")
    
    for n in range(3, min(12, len(rel_bis) + 1)):
        lines = rel_bis[:n]
        
        result = cl._xd_cal_line_xlfx(lines, fx_type, "no_bh")
        if result and result[1]:  # has XLFXs
            tzxls, xlfxs = result
            for fx in xlfxs:
                fx_bis = ",".join(str(l.index) for l in fx.xl.lines)
                # Check ALL three xls for line_bad
                xls_bad = [f"TZ[{j}]bad={xl.line_bad}" for j, xl in enumerate(fx.xls) if xl is not None]
                print(f"  n={n:2d} FX@bi[{fx_bis}] is_line_bad={fx.is_line_bad} xl.line_bad={fx.xl.line_bad} xls:{' '.join(xls_bad)}")

# BTC5m up bi[3]
df5m = pd.read_parquet("tests/test_data/BTC_USDT_5m_1000.parquet")
cl5m = CL_Pyarmor("test", "5m", config)
cl5m.process_klines(df5m)
bis5m = cl5m.get_bis()
analyze(cl5m, bis5m, 3, "up", "BTC5m")

# BTC60 down bi[28]
df60 = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cl60 = CL_Pyarmor("test", "60m", config)
cl60.process_klines(df60)
bis60 = cl60.get_bis()
analyze(cl60, bis60, 28, "down", "BTC60")

# BTC60 up bi[39]
analyze(cl60, bis60, 39, "up", "BTC60")

# ETH60 up bi[15]  
df_eth = pd.read_parquet("tests/test_data/ETH_USDT_60m_1000.parquet")
cl_eth = CL_Pyarmor("test", "60m", config)
cl_eth.process_klines(df_eth)
bis_eth = cl_eth.get_bis()
analyze(cl_eth, bis_eth, 15, "up", "ETH60")

# ETH60 up bi[31]
analyze(cl_eth, bis_eth, 31, "up", "ETH60")

# BTC5m up bi[25]
analyze(cl5m, bis5m, 25, "up", "BTC5m")
