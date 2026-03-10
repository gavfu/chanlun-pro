"""
Investigate: does pyarmor's is_line_bad correspond to a bi-pohuai (笔破坏) check?

For each FX where xl.line_bad=True, check if bi-pohuai is present.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_pyarmor import CL as CL_Pyarmor
from chanlun.cl_open import CL as CL_Open

config = {
    "bi_type": "bi_type_old", "fx_qj": "fx_qj_k", "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes", "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1", "xd_bzh": "xd_bzh_no",
    "xd_zs_max_lines_split": 11, "xd_allow_split_no_highlow": 1,
    "xd_allow_split_zs_kz": 0, "xd_allow_split_zs_more_line": 1,
    "xd_allow_split_zs_no_direction": 1,
}

df60 = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cl_p = CL_Pyarmor("test", "60m", config)
cl_p.process_klines(df60)
bis_p = cl_p.get_bis()

cl_o = CL_Open("test", "60m", config)
cl_o.process_klines(df60)
bis_o = cl_o.get_bis()

# Look at XD objects and their FXes
xds = cl_p.get_xds()

print("=== BTC60 XD FX analysis ===\n")
for xd in xds:
    for name, fx in [("ding", xd.ding_fx), ("di", xd.di_fx)]:
        if fx is None:
            continue
        if not fx.xl.line_bad:
            continue
        
        fx_bis = ",".join(str(l.index) for l in fx.xl.lines)
        
        # Check bi-pohuai using cl_open's method
        # Need TZXL and matching context
        # For bi-pohuai: check if the BI after the FX's last line breaks back
        last_line = fx.xl.lines[-1]
        last_line_idx = last_line.index
        
        # Check the next BI after the last line of the FX
        if last_line_idx + 1 < len(bis_o):
            next_bi = bis_o[last_line_idx + 1]
            
            # For ding FX with down segment: check if next bi's high > FX's max
            # For di FX with up segment: check if next bi's low < FX's min
            bi_pohuai_info = ""
            if name == "ding":
                # up segment's end ding: next bi (down) should break below start
                start_bi = bis_o[xd.start_line.index]
                if next_bi.low < start_bi.low:
                    bi_pohuai_info = "BI_POHUAI (next.low < start.low)"
                else:
                    bi_pohuai_info = f"no_pohuai (next.low={next_bi.low:.1f} start.low={start_bi.low:.1f})"
            else:
                # down segment's end di: next bi (up) should break above start
                start_bi = bis_o[xd.start_line.index]
                if next_bi.high > start_bi.high:
                    bi_pohuai_info = "BI_POHUAI (next.high > start.high)"
                else:
                    bi_pohuai_info = f"no_pohuai (next.high={next_bi.high:.1f} start.high={start_bi.high:.1f})"
        else:
            bi_pohuai_info = "no next bi"
        
        print(f"  xd[{xd.index}] {xd.type} bi[{xd.start_line.index}→{xd.end_line.index}]")
        print(f"    {name} FX@bi[{fx_bis}] is_line_bad={fx.is_line_bad} xl_bad={fx.xl.line_bad}")
        print(f"    {bi_pohuai_info}")
        print()

# Now let's specifically look at the _check_xd_bi_pohuai for key cases
print("\n=== Check bi-pohuai using cl_open's _check_xd_bi_pohuai ===")
from chanlun.cl_interface import TZXL 

# BTC60 down[28]: FX@bi[39], is_line_bad=False
# Need to build a TZXL for bi[39]
bi39 = bis_o[39]
pre_line39 = bis_o[38]
tz39 = TZXL(bh_direction="down", line=bi39, pre_line=pre_line39, line_bad=True, done=True)
pohuai39 = cl_o._check_xd_bi_pohuai(bis_o, 28, tz39, "down")
print(f"  down[28] FX@bi[39]: _check_xd_bi_pohuai = {pohuai39}")

# BTC60 up[39]: FX@bi[42], is_line_bad=True  
bi42 = bis_o[42]
pre_line42 = bis_o[41]
tz42 = TZXL(bh_direction="up", line=bi42, pre_line=pre_line42, line_bad=True, done=True)
pohuai42 = cl_o._check_xd_bi_pohuai(bis_o, 39, tz42, "up")
print(f"  up[39] FX@bi[42]: _check_xd_bi_pohuai = {pohuai42}")

# BTC5m up[3]: FX@bi[6], is_line_bad=True
df5m = pd.read_parquet("tests/test_data/BTC_USDT_5m_1000.parquet")
cl5m_o = CL_Open("test", "5m", config)
cl5m_o.process_klines(df5m)
bis5m_o = cl5m_o.get_bis()

bi6 = bis5m_o[6]
pre_line6 = bis5m_o[5]
tz6 = TZXL(bh_direction="up", line=bi6, pre_line=pre_line6, line_bad=True, done=True)
pohuai6 = cl5m_o._check_xd_bi_pohuai(bis5m_o, 3, tz6, "up")
print(f"  up[3] FX@bi[6]: _check_xd_bi_pohuai = {pohuai6}")

# BTC5m up[3]: FX@bi[10], is_line_bad=True (not bad TZXL)
bi10 = bis5m_o[10]
pre_line10 = bis5m_o[9]
tz10 = TZXL(bh_direction="up", line=bi10, pre_line=pre_line10, line_bad=False, done=True)
pohuai10 = cl5m_o._check_xd_bi_pohuai(bis5m_o, 3, tz10, "up")
print(f"  up[3] FX@bi[10]: _check_xd_bi_pohuai = {pohuai10}")
