"""
验证 ZSD 算法的独立正确性：
直接将 cl_pyarmor 计算出的 XD 列表注入 cl_open 的 _build_xds，
对比 cl_open 和 cl_pyarmor 的 ZSD 输出。
消除 XD 差异对 ZSD 比较的干扰。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import copy
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


def fmt_zsd(zsd):
    if zsd is None:
        return "---"
    try:
        si = zsd.start_line.index
        ei = zsd.end_line.index
        sk = zsd.start_line.start.k.index
        ek = zsd.end_line.end.k.index
        return f"{zsd.type:>4} xd[{si}→{ei}] k[{sk}→{ek}] done={zsd.done}"
    except Exception:
        return f"{zsd.type:>4} done={zsd.done}"


def fmt_zss(zs):
    return f"zg={zs.zg:.2f} zd={zs.zd:.2f} type={zs.zs_type}"


for name, path in DATASETS.items():
    df = pd.read_parquet(path)

    # 计算 cl_pyarmor 完整结果（XD 和 ZSD）
    cl_p = CL_P("test", "test", CL_CONFIG)
    cl_p.process_klines(df)
    xds_p = cl_p.get_xds()
    zsds_p = cl_p.get_zsds()
    zsd_zss_p = cl_p.get_zsd_zss()

    # 用 cl_open 引擎，但强制注入 cl_pyarmor 的 XD 列表，仅测试 _build_zsds 算法
    cl_o = CL_O("test", "test", CL_CONFIG)
    cl_o.process_klines(df)
    # 注入 pyarmor XDs（给 xd.index 重新编号，避免 ZSD 算法依赖绝对 index）
    for i, xd in enumerate(xds_p):
        xd.index = i
    cl_o.xds = list(xds_p)
    # 重新运行 ZSD
    cl_o._build_zsds()
    cl_o._build_zsd_zss()
    zsds_o = cl_o.zsds
    zsd_zss_o = cl_o.zsd_zss

    print(f"\n{'='*75}")
    print(f"{name}: injected XD={len(xds_p)} (from pyarmor)")
    print(f"  ZSD: open(injected)={len(zsds_o)}  pyarmor={len(zsds_p)}  "
          f"{'✅' if len(zsds_o) == len(zsds_p) else '❌'}")

    max_n = max(len(zsds_o), len(zsds_p))
    for i in range(max_n):
        o = zsds_o[i] if i < len(zsds_o) else None
        p = zsds_p[i] if i < len(zsds_p) else None
        if o and p:
            try:
                so, to = o.start_line.index, o.type
                sp, tp = p.start_line.index, p.type
                eo, ep = o.end_line.index, p.end_line.index
                match = "✅" if (so == sp and eo == ep and to == tp) else "❌"
            except Exception:
                match = "?"
        else:
            match = "❌"
        print(f"  zsd[{i}] {match}  open: {fmt_zsd(o):50} pyarmor: {fmt_zsd(p)}")

    print(f"  ZSD_ZSS: open={len(zsd_zss_o)}  pyarmor={len(zsd_zss_p)}  "
          f"{'✅' if len(zsd_zss_o) == len(zsd_zss_p) else '❌'}")
    for i in range(max(len(zsd_zss_o), len(zsd_zss_p))):
        o = zsd_zss_o[i] if i < len(zsd_zss_o) else None
        p = zsd_zss_p[i] if i < len(zsd_zss_p) else None
        if o and p:
            match = "✅" if (abs(o.zg - p.zg) < 1e-6 and abs(o.zd - p.zd) < 1e-6
                             and o.zs_type == p.zs_type) else "❌"
        else:
            match = "❌"
        print(f"  zss[{i}] {match}  open: {fmt_zss(o) if o else '---':40} pyarmor: {fmt_zss(p) if p else '---'}")
