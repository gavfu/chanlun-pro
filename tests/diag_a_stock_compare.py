"""
对比脚本：cl_open vs cl_pyarmor — 中信证券日线 (SH.600030)
重点检查笔中枢、线段中枢
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

DATA_PATH = os.path.join(os.path.dirname(__file__), 'test_data', 'SH_600030_d.parquet')

CL_CONFIG = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
}

# ========================================================
# 比较辅助函数
# ========================================================

def cmp_bis(label, bis_o, bis_p):
    """比较笔"""
    n_o, n_p = len(bis_o), len(bis_p)
    count_ok = n_o == n_p
    print(f"  [{label}] open={n_o} pyarmor={n_p} {'✅' if count_ok else '❌'}")

    match_count = 0
    boundary_mismatches = 0
    first_diff = None
    for i in range(min(n_o, n_p)):
        bo, bp = bis_o[i], bis_p[i]
        if (bo.start.k.k_index == bp.start.k.k_index
                and bo.end.k.k_index == bp.end.k.k_index
                and bo.type == bp.type):
            match_count += 1
        else:
            boundary_mismatches += 1
            if first_diff is None:
                first_diff = i

    total = max(n_o, n_p)
    if match_count == total and count_ok:
        print(f"    内容全部一致 ✅ ({match_count}/{total})")
    else:
        print(f"    内容匹配 {match_count}/{min(n_o, n_p)}, 边界差异 {boundary_mismatches}")
        if first_diff is not None:
            bo, bp = bis_o[first_diff], bis_p[first_diff]
            print(f"    首个差异 bi[{first_diff}]: open({bo.type} k={bo.start.k.k_index}→{bo.end.k.k_index}) "
                  f"vs pyarmor({bp.type} k={bp.start.k.k_index}→{bp.end.k.k_index})")

    return count_ok and boundary_mismatches == 0


def cmp_lines(label, lines_o, lines_p, show_detail=True):
    """比较线段/走势段/趋势段"""
    n_o, n_p = len(lines_o), len(lines_p)
    count_ok = n_o == n_p
    print(f"  [{label}] open={n_o} pyarmor={n_p} {'✅' if count_ok else '❌'}")

    match_count = 0
    first_diff = None
    for i in range(min(n_o, n_p)):
        lo, lp = lines_o[i], lines_p[i]
        if (lo.start_line.index == lp.start_line.index
                and lo.end_line.index == lp.end_line.index
                and lo.type == lp.type):
            match_count += 1
        elif first_diff is None:
            first_diff = i

    total = max(n_o, n_p)
    if match_count == total and count_ok:
        print(f"    内容全部一致 ✅ ({match_count}/{total})")
    else:
        print(f"    内容匹配 {match_count}/{min(n_o, n_p)}")
        if first_diff is not None and show_detail:
            lo, lp = lines_o[first_diff], lines_p[first_diff]
            print(f"    首个差异 [{first_diff}]:")
            print(f"      open:    {lo.type} line[{lo.start_line.index}→{lo.end_line.index}] done={lo.done}")
            print(f"      pyarmor: {lp.type} line[{lp.start_line.index}→{lp.end_line.index}] done={lp.done}")

    if show_detail:
        max_show = max(n_o, n_p)
        for i in range(max_show):
            o = lines_o[i] if i < n_o else None
            p = lines_p[i] if i < n_p else None
            if o and p:
                start_ok = o.start_line.index == p.start_line.index
                end_ok = o.end_line.index == p.end_line.index
                mark = "✅" if (start_ok and end_ok and o.type == p.type) else "❌"
            else:
                mark = "❌"
            o_s = f"{o.type:>4} line[{o.start_line.index}→{o.end_line.index}] done={o.done}" if o else "---"
            p_s = f"{p.type:>4} line[{p.start_line.index}→{p.end_line.index}] done={p.done}" if p else "---"
            print(f"    [{i:>2}] {mark} open: {o_s:42} pyarmor: {p_s}")

    return count_ok and match_count == total


def cmp_zss(label, zss_o, zss_p, show_detail=True):
    """比较中枢"""
    n_o, n_p = len(zss_o), len(zss_p)
    count_ok = n_o == n_p
    print(f"  [{label}] open={n_o} pyarmor={n_p} {'✅' if count_ok else '❌'}")

    match_count = 0
    first_diff = None
    for i in range(min(n_o, n_p)):
        zo, zp = zss_o[i], zss_p[i]
        if (abs(zo.zg - zp.zg) < 1e-9
                and abs(zo.zd - zp.zd) < 1e-9
                and zo.zs_type == zp.zs_type
                and zo.type == zp.type
                and zo.line_num == zp.line_num):
            match_count += 1
        elif first_diff is None:
            first_diff = i

    total = max(n_o, n_p)
    if match_count == total and count_ok:
        print(f"    内容全部一致 ✅ ({match_count}/{total})")
    else:
        print(f"    内容匹配 {match_count}/{min(n_o, n_p)}")
        if first_diff is not None and show_detail:
            zo, zp = zss_o[first_diff], zss_p[first_diff]
            print(f"    首个差异 [{first_diff}]:")
            print(f"      open:    zg={zo.zg:.4f} zd={zo.zd:.4f} type={zo.type} zs_type={zo.zs_type} lines={zo.line_num} done={zo.done}")
            print(f"      pyarmor: zg={zp.zg:.4f} zd={zp.zd:.4f} type={zp.type} zs_type={zp.zs_type} lines={zp.line_num} done={zp.done}")

    if show_detail:
        max_show = max(n_o, n_p)
        for i in range(max_show):
            o = zss_o[i] if i < n_o else None
            p = zss_p[i] if i < n_p else None
            if o and p:
                mark = "✅" if (abs(o.zg - p.zg) < 1e-9 and abs(o.zd - p.zd) < 1e-9
                                and o.type == p.type and o.line_num == p.line_num
                                and o.zs_type == p.zs_type) else "❌"
            else:
                mark = "❌"
            o_s = f"zg={o.zg:.2f} zd={o.zd:.2f} {o.type:>4} zs_type={o.zs_type} L={o.line_num} done={o.done}" if o else "---"
            p_s = f"zg={p.zg:.2f} zd={p.zd:.2f} {p.type:>4} zs_type={p.zs_type} L={p.line_num} done={p.done}" if p else "---"
            print(f"    [{i:>2}] {mark} open: {o_s:58} pyarmor: {p_s}")

    return count_ok and match_count == total


def cmp_bc_mmd(label, lines_o, lines_p):
    """比较线上的背驰和买卖点"""
    n = min(len(lines_o), len(lines_p))
    if n == 0:
        print(f"  [{label} BC/MMD] 无数据可比较")
        return True

    bc_total = 0
    bc_match = 0
    mmd_total = 0
    mmd_match = 0
    bc_diffs = []
    mmd_diffs = []

    for i in range(n):
        lo, lp = lines_o[i], lines_p[i]

        bcs_o = sorted(lo.line_bcs())
        bcs_p = sorted(lp.line_bcs())
        bc_total += 1
        if bcs_o == bcs_p:
            bc_match += 1
        else:
            bc_diffs.append((i, bcs_o, bcs_p))

        mmds_o = sorted(lo.line_mmds())
        mmds_p = sorted(lp.line_mmds())
        mmd_total += 1
        if mmds_o == mmds_p:
            mmd_match += 1
        else:
            mmd_diffs.append((i, mmds_o, mmds_p))

    bc_ok = bc_match == bc_total
    mmd_ok = mmd_match == mmd_total
    print(f"  [{label} BC]  匹配 {bc_match}/{bc_total} {'✅' if bc_ok else '❌'}")
    print(f"  [{label} MMD] 匹配 {mmd_match}/{mmd_total} {'✅' if mmd_ok else '❌'}")

    for diff_label, diffs in [("BC", bc_diffs), ("MMD", mmd_diffs)]:
        if diffs:
            show = min(5, len(diffs))
            for j in range(show):
                idx, val_o, val_p = diffs[j]
                print(f"    {diff_label} diff [{idx}]: open={val_o} pyarmor={val_p}")
            if len(diffs) > show:
                print(f"    ... 还有 {len(diffs) - show} 个差异")

    return bc_ok and mmd_ok


# ========================================================
# 主运行
# ========================================================

def main():
    print(f"\n{'='*90}")
    print(f"  全面对比 cl_open vs cl_pyarmor — 中信证券 SH.600030 日线")
    print(f"{'='*90}")

    df = pd.read_parquet(DATA_PATH)
    print(f"\n  数据: {len(df)} 条 K线, {df['date'].min()} ~ {df['date'].max()}")
    print(f"  配置: {CL_CONFIG}")

    print(f"\n  正在计算 cl_open ...")
    cd_o = CL_O("SH.600030", "d", config=CL_CONFIG)
    cd_o.process_klines(df)

    print(f"  正在计算 cl_pyarmor ...")
    cd_p = CL_P("SH.600030", "d", config=CL_CONFIG)
    cd_p.process_klines(df)

    results = {}

    # ---- 1. 笔 ----
    print(f"\n{'─'*70}")
    print(f"  1. 笔 (bis)")
    print(f"{'─'*70}")
    bis_o = cd_o.get_bis()
    bis_p = cd_p.get_bis()
    results['bis'] = cmp_bis("bis", bis_o, bis_p)

    # ---- 2. 笔中枢 (重点) ----
    print(f"\n{'─'*70}")
    print(f"  2. 笔中枢 (bi_zss) ★★★")
    print(f"{'─'*70}")
    bi_zss_o = cd_o.get_bi_zss()
    bi_zss_p = cd_p.get_bi_zss()
    results['bi_zss'] = cmp_zss("bi_zss", bi_zss_o, bi_zss_p)

    # ---- 3. 线段 ----
    print(f"\n{'─'*70}")
    print(f"  3. 线段 (xds)")
    print(f"{'─'*70}")
    xds_o = cd_o.get_xds()
    xds_p = cd_p.get_xds()
    results['xds'] = cmp_lines("xds", xds_o, xds_p)

    # ---- 4. 线段中枢 (重点) ----
    print(f"\n{'─'*70}")
    print(f"  4. 线段中枢 (xd_zss) ★★★")
    print(f"{'─'*70}")
    xd_zss_o = cd_o.get_xd_zss()
    xd_zss_p = cd_p.get_xd_zss()
    results['xd_zss'] = cmp_zss("xd_zss", xd_zss_o, xd_zss_p)

    # ---- 5. 笔 BC/MMD ----
    print(f"\n{'─'*70}")
    print(f"  5. 笔背驰 & 买卖点")
    print(f"{'─'*70}")
    results['bi_bc_mmd'] = cmp_bc_mmd("BI", bis_o, bis_p)

    # ---- 6. 线段 BC/MMD ----
    print(f"\n{'─'*70}")
    print(f"  6. 线段背驰 & 买卖点")
    print(f"{'─'*70}")
    results['xd_bc_mmd'] = cmp_bc_mmd("XD", xds_o, xds_p)

    # ---- 7. 走势段 ----
    print(f"\n{'─'*70}")
    print(f"  7. 走势段 (zsds)")
    print(f"{'─'*70}")
    zsds_o = cd_o.get_zsds()
    zsds_p = cd_p.get_zsds()
    results['zsds'] = cmp_lines("zsds", zsds_o, zsds_p)

    # ---- 8. 走势段中枢 ----
    print(f"\n{'─'*70}")
    print(f"  8. 走势段中枢 (zsd_zss)")
    print(f"{'─'*70}")
    zsd_zss_o = cd_o.get_zsd_zss()
    zsd_zss_p = cd_p.get_zsd_zss()
    results['zsd_zss'] = cmp_zss("zsd_zss", zsd_zss_o, zsd_zss_p)

    # ---- 9. 趋势走势段 ----
    print(f"\n{'─'*70}")
    print(f"  9. 趋势走势段 (qsds)")
    print(f"{'─'*70}")
    qsds_o = cd_o.get_qsds()
    qsds_p = cd_p.get_qsds()
    results['qsds'] = cmp_lines("qsds", qsds_o, qsds_p)

    # ---- 10. 趋势段中枢 ----
    print(f"\n{'─'*70}")
    print(f"  10. 趋势段中枢 (qsd_zss)")
    print(f"{'─'*70}")
    qsd_zss_o = cd_o.get_qsd_zss()
    qsd_zss_p = cd_p.get_qsd_zss()
    results['qsd_zss'] = cmp_zss("qsd_zss", qsd_zss_o, qsd_zss_p)

    # ========= 汇总 =========
    print(f"\n{'='*90}")
    print(f"  汇总 — 中信证券 SH.600030 日线")
    print(f"{'='*90}")
    for k, v in results.items():
        print(f"  {k:20s}: {'✅ 一致' if v else '❌ 不一致'}")
    total_ok = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n  总计: {total_ok}/{total} 项一致")


if __name__ == '__main__':
    main()
