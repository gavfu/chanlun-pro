"""
Dump detailed XLFX attributes from pyarmor's _xd_cal_line_xlfx tuple.
The tuple is (list[TZXL], list[XLFX]).
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

def dump_xlfx(fx, prefix=""):
    """Dump an XLFX object details."""
    lines_str = ",".join(str(l.index) for l in fx.xl.lines)
    xls_lines = []
    for i, xl in enumerate(fx.xls):
        if xl is not None:
            xl_bis = ",".join(str(l.index) for l in xl.lines)
            xls_lines.append(f"TZ[{i}]bi[{xl_bis}]max={xl.max:.1f}min={xl.min:.1f}")
        else:
            xls_lines.append(f"TZ[{i}]=None")
    print(f"{prefix}XLFX type={fx.type} done={fx.done} qk={fx.qk} is_line_bad={fx.is_line_bad}")
    print(f"{prefix}  xl: bi[{lines_str}] max={fx.xl.max:.1f} min={fx.xl.min:.1f}")
    print(f"{prefix}  xls: {' | '.join(xls_lines)}")
    print(f"{prefix}  fx_high={fx.fx_high:.1f} fx_low={fx.fx_low:.1f} bh_type={fx.bh_type}")

def dump_tzxl(tz, idx, prefix=""):
    """Dump a TZXL object details."""
    lines_str = ",".join(str(l.index) for l in tz.lines)
    print(f"{prefix}TZ[{idx}] bi[{lines_str}] max={tz.max:.1f} min={tz.min:.1f}")

def analyze_case(cl, bis, start_bi_idx, direction, label):
    """Analyze a single case."""
    print(f"\n{'='*80}")
    print(f"{label}: {direction} from bi[{start_bi_idx}]")
    print(f"{'='*80}")
    
    if direction == "up":
        fx_type = "ding"
        rel_bis = [b for b in bis[start_bi_idx:] if b.type == "down"]
    else:
        fx_type = "di"
        rel_bis = [b for b in bis[start_bi_idx:] if b.type == "up"]
    
    for n in range(3, min(12, len(rel_bis) + 1)):
        lines = rel_bis[:n]
        lines_idx = ",".join(str(l.index) for l in lines)
        
        result_nobh = cl._xd_cal_line_xlfx(lines, fx_type, "no_bh")
        result_bh = cl._xd_cal_line_xlfx(lines, fx_type, "bh")
        
        print(f"\n  n={n} lines=[{lines_idx}]")
        
        if result_nobh is not None:
            tzxls, xlfxs = result_nobh
            print(f"    no_bh: {len(tzxls)} TZXLs, {len(xlfxs)} XLFXs")
            for i, tz in enumerate(tzxls):
                dump_tzxl(tz, i, "      ")
            for i, fx in enumerate(xlfxs):
                dump_xlfx(fx, f"      FX[{i}] ")
        else:
            print(f"    no_bh: None")
        
        if result_bh is not None:
            tzxls, xlfxs = result_bh
            print(f"    bh:    {len(tzxls)} TZXLs, {len(xlfxs)} XLFXs")
            for i, tz in enumerate(tzxls):
                dump_tzxl(tz, i, "      ")
            for i, fx in enumerate(xlfxs):
                dump_xlfx(fx, f"      FX[{i}] ")
        else:
            print(f"    bh:    None")


# === BTC5m up bi[3] ===
df5m = pd.read_parquet("tests/test_data/BTC_USDT_5m_1000.parquet")
cl5m = CL_Pyarmor("test", "5m", config)
cl5m.process_klines(df5m)
bis5m = cl5m.get_bis()
analyze_case(cl5m, bis5m, 3, "up", "BTC5m")

# === BTC60 down bi[28] ===
df60 = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cl60 = CL_Pyarmor("test", "60m", config)
cl60.process_klines(df60)
bis60 = cl60.get_bis()
analyze_case(cl60, bis60, 28, "down", "BTC60")

# === BTC60 up bi[39] ===
analyze_case(cl60, bis60, 39, "up", "BTC60")
