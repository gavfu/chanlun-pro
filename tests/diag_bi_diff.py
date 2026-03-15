"""诊断笔差异 - 详细分析 bi[20] 附近的差异"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

# 先看小的数据集，有没有笔差异
DATA_FILES = {
    "BTC60": "tests/test_data/BTC_USDT_60m_1000.parquet",
    "ETH60": "tests/test_data/ETH_USDT_60m_1000.parquet",
    "BTC5m": "tests/test_data/BTC_USDT_5m_1000.parquet",
    "ETH5m": "tests/test_data/ETH_USDT_5m_1000.parquet",
    "BTCd":  "tests/test_data/BTC_USDT_d_500.parquet",
    "BTC4h5k": "tests/test_data/BTC_USDT_4h_5000.parquet",
    "ETH4h5k": "tests/test_data/ETH_USDT_4h_5000.parquet",
    "BTCd3k":  "tests/test_data/BTC_USDT_d_3000.parquet",
}

CL_CONFIG = {
    'bi_type': 'bi_type_old', 'fx_qj': 'fx_qj_k', 'fx_qy': 'fx_qy_three',
    'bi_fx_cgd': 'bi_fx_cgd_yes', 'fx_check_k_nums': 13,
    'bi_split_k_cross_nums': '20,1', 'xd_bzh': 'xd_bzh_no',
}


def check_bi(name, path):
    df = pd.read_parquet(path)
    co = CL_O("test", "test", config=CL_CONFIG)
    co.process_klines(df)
    cp = CL_P("test", "test", config=CL_CONFIG)
    cp.process_klines(df)

    bis_o = co.get_bis()
    bis_p = cp.get_bis()

    # Count boundary differences
    mismatches = 0
    first_diff = None
    for i in range(min(len(bis_o), len(bis_p))):
        bo, bp = bis_o[i], bis_p[i]
        if bo.start.k.k_index != bp.start.k.k_index or bo.end.k.k_index != bp.end.k.k_index:
            mismatches += 1
            if first_diff is None:
                first_diff = i

    status = "✅" if len(bis_o) == len(bis_p) and mismatches == 0 else "❌"
    print(f"  {name:10s} bis: open={len(bis_o)} pyarmor={len(bis_p)} boundary_diff={mismatches} {status}")
    if first_diff is not None:
        bo = bis_o[first_diff]
        bp = bis_p[first_diff]
        print(f"    首个差异 bi[{first_diff}]: open({bo.type} k={bo.start.k.k_index}→{bo.end.k.k_index}) vs pyarmor({bp.type} k={bp.start.k.k_index}→{bp.end.k.k_index})")
        # Show the surrounding bis
        for j in range(max(0, first_diff - 2), min(len(bis_o), first_diff + 5)):
            bo2 = bis_o[j] if j < len(bis_o) else None
            bp2 = bis_p[j] if j < len(bis_p) else None
            o_s = f"{bo2.type:>4} k={bo2.start.k.k_index}→{bo2.end.k.k_index} split=[{bo2.is_split}]" if bo2 else "---"
            p_s = f"{bp2.type:>4} k={bp2.start.k.k_index}→{bp2.end.k.k_index} split=[{bp2.is_split}]" if bp2 else "---"
            mark = "←DIFF" if j == first_diff else ""
            print(f"      bi[{j:>3}] open: {o_s:50} pyarmor: {p_s} {mark}")


for name, path in DATA_FILES.items():
    check_bi(name, path)
