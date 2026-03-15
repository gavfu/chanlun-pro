"""
诊断脚本：对比 cl_open 和 cl_pyarmor 的 zsds/qsds/zsd_zss/qsd_zss 输出
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

TEST_DATA = {
    "BTCd":      "tests/test_data/BTC_USDT_d_500.parquet",
    "ETH60":     "tests/test_data/ETH_USDT_60m_1000.parquet",
    "BTC60":     "tests/test_data/BTC_USDT_60m_1000.parquet",
    "BTC5m":     "tests/test_data/BTC_USDT_5m_1000.parquet",
    "ETH5m":     "tests/test_data/ETH_USDT_5m_1000.parquet",
    "BTC4h5k":   "tests/test_data/BTC_USDT_4h_5000.parquet",
    "ETH4h5k":   "tests/test_data/ETH_USDT_4h_5000.parquet",
    "BTCd3k":    "tests/test_data/BTC_USDT_d_3000.parquet",
}

CL_CONFIG = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
}


def fmt_line(line, label=""):
    if line is None:
        return "---"
    try:
        start_k = line.start_line.index if hasattr(line, 'start_line') and line.start_line else "?"
        end_k = line.end_line.index if hasattr(line, 'end_line') and line.end_line else "?"
        return f"{line.type:>4} xd[{start_k}→{end_k}] done={line.done}"
    except Exception:
        return f"{line.type:>4} done={line.done}"


def compare_lines(name, label, lines_o, lines_p):
    match_count = sum(
        1 for i in range(min(len(lines_o), len(lines_p)))
        if (lines_o[i].start_line.index == lines_p[i].start_line.index
            and lines_o[i].end_line.index == lines_p[i].end_line.index
            and lines_o[i].type == lines_p[i].type)
    )
    total = max(len(lines_o), len(lines_p))
    count_match = "✅" if len(lines_o) == len(lines_p) else "❌"
    print(f"\n  [{label}] open={len(lines_o)} pyarmor={len(lines_p)} {count_match}  "
          f"content_match={match_count}/{min(len(lines_o), len(lines_p))}")

    max_show = max(len(lines_o), len(lines_p))
    first_diff = None
    for i in range(max_show):
        o = lines_o[i] if i < len(lines_o) else None
        p = lines_p[i] if i < len(lines_p) else None

        if o and p:
            try:
                start_match = o.start_line.index == p.start_line.index
                end_match = o.end_line.index == p.end_line.index
                match = "✅" if (start_match and end_match and o.type == p.type) else "❌"
                if match == "❌" and first_diff is None:
                    first_diff = i
            except Exception:
                match = "?"
        else:
            match = "❌"
            if first_diff is None:
                first_diff = i

        o_str = fmt_line(o)
        p_str = fmt_line(p)
        print(f"    [{i:>2}] {match} open: {o_str:45} pyarmor: {p_str}")

    if first_diff is not None:
        print(f"  *** 首个差异在 [{label}][{first_diff}] ***")


def compare_zss(name, label, zss_o, zss_p):
    count_match = "✅" if len(zss_o) == len(zss_p) else "❌"
    print(f"\n  [{label}_zss] open={len(zss_o)} pyarmor={len(zss_p)} {count_match}")
    for i in range(max(len(zss_o), len(zss_p))):
        o = zss_o[i] if i < len(zss_o) else None
        p = zss_p[i] if i < len(zss_p) else None
        if o and p:
            match = "✅" if (abs(o.zg - p.zg) < 1e-9 and abs(o.zd - p.zd) < 1e-9
                             and o.zs_type == p.zs_type) else "❌"
        else:
            match = "❌"
        o_str = f"zg={o.zg:.4f} zd={o.zd:.4f} type={o.zs_type}" if o else "---"
        p_str = f"zg={p.zg:.4f} zd={p.zd:.4f} type={p.zs_type}" if p else "---"
        print(f"    [{i:>2}] {match} open: {o_str:45} pyarmor: {p_str}")


def run_case(name, data_path):
    df = pd.read_parquet(data_path)
    cd_o = CL_O("test", "test", config=CL_CONFIG)
    cd_o.process_klines(df)
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)

    zsds_o = cd_o.get_zsds()
    zsds_p = cd_p.get_zsds()
    qsds_o = cd_o.get_qsds()
    qsds_p = cd_p.get_qsds()
    zsd_zss_o = cd_o.get_zsd_zss()
    zsd_zss_p = cd_p.get_zsd_zss()
    qsd_zss_o = cd_o.get_qsd_zss()
    qsd_zss_p = cd_p.get_qsd_zss()

    # Also show XD count for reference
    xds_o = cd_o.get_xds()
    xds_p = cd_p.get_xds()

    print(f"\n{'='*90}")
    print(f"  {name}: XD={len(xds_o)}/{len(xds_p)}  "
          f"ZSD={len(zsds_o)}/{len(zsds_p)} {'✅' if len(zsds_o)==len(zsds_p) else '❌'}  "
          f"QSD={len(qsds_o)}/{len(qsds_p)} {'✅' if len(qsds_o)==len(qsds_p) else '❌'}")
    print(f"{'='*90}")

    compare_lines(name, "zsds", zsds_o, zsds_p)
    compare_zss(name, "zsd", zsd_zss_o, zsd_zss_p)
    compare_lines(name, "qsds", qsds_o, qsds_p)
    compare_zss(name, "qsd", qsd_zss_o, qsd_zss_p)


if __name__ == "__main__":
    cases = sys.argv[1:] if len(sys.argv) > 1 else list(TEST_DATA.keys())
    for case in cases:
        if case in TEST_DATA:
            run_case(case, TEST_DATA[case])
        else:
            print(f"Unknown case: {case}. Available: {list(TEST_DATA.keys())}")
