"""
快速对比大数据集的 XD 差异点，定位 XD 分歧的第一个不匹配项。
用于分析 ZSD 差异的根本原因。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

CL_CONFIG = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
}

DATASETS = {
    "BTC4h5k": "tests/test_data/BTC_USDT_4h_5000.parquet",
    "ETH4h5k": "tests/test_data/ETH_USDT_4h_5000.parquet",
    "BTCd3k":  "tests/test_data/BTC_USDT_d_3000.parquet",
}


def xd_key(xd):
    try:
        return (xd.type, xd.start.k.index, xd.end.k.index)
    except Exception:
        return (xd.type, -1, -1)


for name, path in DATASETS.items():
    df = pd.read_parquet(path)
    cl_o = CL_O("test", "test", CL_CONFIG)
    cl_p = CL_P("test", "test", CL_CONFIG)
    cl_o.process_klines(df)
    cl_p.process_klines(df)

    xds_o = cl_o.get_xds()
    xds_p = cl_p.get_xds()
    bis_o = cl_o.get_bis()
    bis_p = cl_p.get_bis()

    print(f"\n{'='*70}")
    print(f"{name}: bis={len(bis_o)}/{len(bis_p)}  xds={len(xds_o)}/{len(xds_p)}")

    # Find first XD divergence
    first_diff = None
    for i in range(min(len(xds_o), len(xds_p))):
        ko = xd_key(xds_o[i])
        kp = xd_key(xds_p[i])
        if ko != kp:
            first_diff = i
            break

    if first_diff is None and len(xds_o) == len(xds_p):
        print("  XD: 完全一致 ✅")
        continue

    if first_diff is None:
        first_diff = min(len(xds_o), len(xds_p))

    print(f"  XD 首个差异: index={first_diff}  (open={len(xds_o)}, pyarmor={len(xds_p)})")

    # Print context around first diff
    start = max(0, first_diff - 2)
    end = min(max(len(xds_o), len(xds_p)), first_diff + 5)
    for i in range(start, end):
        o = xds_o[i] if i < len(xds_o) else None
        p = xds_p[i] if i < len(xds_p) else None
        if o and p:
            ko = xd_key(o)
            kp = xd_key(p)
            match = "✅" if ko == kp else "❌"
        else:
            match = "❌"
        o_str = f"{o.type:>4} bi[{o.start.index}→{o.end.index}]  k[{o.start.k.index}→{o.end.k.index}]" if o else "---"
        p_str = f"{p.type:>4} bi[{p.start.index}→{p.end.index}]  k[{p.start.k.index}→{p.end.k.index}]" if p else "---"
        print(f"  xd[{i:>2}] {match}  open: {o_str:45} pyarmor: {p_str}")

    # Also show BI diff around the first diverging XD
    if first_diff < len(xds_o):
        div_xd = xds_o[first_diff]
        bi_start = div_xd.start.index
        print(f"\n  XD[{first_diff}] starts at bi[{bi_start}] in cl_open")
        if first_diff < len(xds_p):
            pya_xd = xds_p[first_diff]
            bi_start_p = pya_xd.start.index
            print(f"  XD[{first_diff}] starts at bi[{bi_start_p}] in cl_pyarmor")
        print(f"  BI differences: open={len(bis_o)}, pyarmor={len(bis_p)}")
