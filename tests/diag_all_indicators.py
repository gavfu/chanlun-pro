"""
全面对比脚本：cl_open vs cl_pyarmor 所有缠论技术指标
使用 ETH/USDT 30m 数据 (2025-01-01 ~ 2026-01-01)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

DATA_PATH = os.path.join(os.path.dirname(__file__), 'test_data', 'ETH_USDT_30m_2025.parquet')

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

def cmp_klines(label, klines_o, klines_p):
    """比较原始K线"""
    n_o, n_p = len(klines_o), len(klines_p)
    ok = n_o == n_p
    print(f"  [{label}] open={n_o} pyarmor={n_p} {'✅' if ok else '❌'}")
    if ok and n_o > 0:
        # 抽样验证
        mismatches = 0
        for i in range(0, n_o, max(1, n_o // 20)):
            ko, kp = klines_o[i], klines_p[i]
            if abs(ko.h - kp.h) > 1e-9 or abs(ko.l - kp.l) > 1e-9:
                mismatches += 1
        print(f"    抽样 {min(20, n_o)} 条: {'全部一致 ✅' if mismatches == 0 else f'{mismatches}条不一致 ❌'}")
    return ok


def cmp_cl_klines(label, ck_o, ck_p):
    """比较缠论K线"""
    n_o, n_p = len(ck_o), len(ck_p)
    count_ok = n_o == n_p
    print(f"  [{label}] open={n_o} pyarmor={n_p} {'✅' if count_ok else '❌'}")
    if not count_ok:
        return False
    mismatches = 0
    first_diff = None
    for i in range(n_o):
        co, cp = ck_o[i], ck_p[i]
        if abs(co.h - cp.h) > 1e-9 or abs(co.l - cp.l) > 1e-9 or co.k_index != cp.k_index:
            mismatches += 1
            if first_diff is None:
                first_diff = i
    if mismatches == 0:
        print(f"    内容全部一致 ✅")
    else:
        print(f"    {mismatches} 条不一致 ❌ (首个差异 index={first_diff})")
        i = first_diff
        co, cp = ck_o[i], ck_p[i]
        print(f"      open:    idx={co.index} k_idx={co.k_index} h={co.h:.4f} l={co.l:.4f} n={co.n}")
        print(f"      pyarmor: idx={cp.index} k_idx={cp.k_index} h={cp.h:.4f} l={cp.l:.4f} n={cp.n}")
    return mismatches == 0


def cmp_fxs(label, fxs_o, fxs_p):
    """比较分型"""
    n_o, n_p = len(fxs_o), len(fxs_p)
    count_ok = n_o == n_p
    print(f"  [{label}] open={n_o} pyarmor={n_p} {'✅' if count_ok else '❌'}")
    if not count_ok:
        # 找首个差异
        for i in range(min(n_o, n_p)):
            fo, fp = fxs_o[i], fxs_p[i]
            if fo.type != fp.type or fo.k.k_index != fp.k.k_index:
                print(f"    首个差异 [{i}]: open({fo.type} k_idx={fo.k.k_index}) vs pyarmor({fp.type} k_idx={fp.k.k_index})")
                break
        return False
    mismatches = 0
    for i in range(n_o):
        fo, fp = fxs_o[i], fxs_p[i]
        if fo.type != fp.type or fo.k.k_index != fp.k.k_index or abs(fo.val - fp.val) > 1e-9:
            mismatches += 1
    if mismatches == 0:
        print(f"    内容全部一致 ✅")
    else:
        print(f"    {mismatches} 条不一致 ❌")
    return mismatches == 0


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
            print(f"    首个差异 bi[{first_diff}]: open({bo.type} k={bo.start.k.k_index}→{bo.end.k.k_index}) vs pyarmor({bp.type} k={bp.start.k.k_index}→{bp.end.k.k_index})")

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

    if show_detail and (n_o > 0 or n_p > 0):
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
            print(f"      open:    zg={zo.zg:.4f} zd={zo.zd:.4f} type={zo.type} lines={zo.line_num} done={zo.done}")
            print(f"      pyarmor: zg={zp.zg:.4f} zd={zp.zd:.4f} type={zp.type} lines={zp.line_num} done={zp.done}")

    if show_detail and (n_o > 0 or n_p > 0):
        max_show = max(n_o, n_p)
        for i in range(max_show):
            o = zss_o[i] if i < n_o else None
            p = zss_p[i] if i < n_p else None
            if o and p:
                mark = "✅" if (abs(o.zg - p.zg) < 1e-9 and abs(o.zd - p.zd) < 1e-9
                                and o.type == p.type and o.line_num == p.line_num) else "❌"
            else:
                mark = "❌"
            o_s = f"zg={o.zg:.2f} zd={o.zd:.2f} {o.type:>4} L={o.line_num} done={o.done}" if o else "---"
            p_s = f"zg={p.zg:.2f} zd={p.zd:.2f} {p.type:>4} L={p.line_num} done={p.done}" if p else "---"
            print(f"    [{i:>2}] {mark} open: {o_s:48} pyarmor: {p_s}")

    return count_ok and match_count == total


def cmp_macd(label, idx_o, idx_p):
    """比较 MACD 指标"""
    if 'macd' not in idx_o or 'macd' not in idx_p:
        print(f"  [{label}] MACD 数据缺失 ❌")
        return False

    macd_o = idx_o['macd']
    macd_p = idx_p['macd']
    all_ok = True
    for key in ['dif', 'dea', 'hist']:
        arr_o = np.array(macd_o[key])
        arr_p = np.array(macd_p[key])
        if len(arr_o) != len(arr_p):
            print(f"  [{label}.{key}] 长度不同: open={len(arr_o)} pyarmor={len(arr_p)} ❌")
            all_ok = False
            continue
        # 跳过 NaN 的位置
        valid = ~(np.isnan(arr_o) | np.isnan(arr_p))
        if valid.sum() == 0:
            print(f"  [{label}.{key}] 全部为 NaN")
            continue
        max_diff = np.max(np.abs(arr_o[valid] - arr_p[valid]))
        ok = max_diff < 1e-6
        if ok:
            print(f"  [{label}.{key}] 最大差异={max_diff:.2e} ✅")
        else:
            print(f"  [{label}.{key}] 最大差异={max_diff:.2e} ❌")
            all_ok = False
    return all_ok


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

        # 比较背驰
        bcs_o = sorted(lo.line_bcs())
        bcs_p = sorted(lp.line_bcs())
        bc_total += 1
        if bcs_o == bcs_p:
            bc_match += 1
        else:
            bc_diffs.append((i, bcs_o, bcs_p))

        # 比较买卖点
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

    # 显示前几个差异
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
    print(f"  全面对比 cl_open vs cl_pyarmor — ETH/USDT 30m (2025)")
    print(f"{'='*90}")

    df = pd.read_parquet(DATA_PATH)
    print(f"\n  数据: {len(df)} 条 K线, {df['date'].min()} ~ {df['date'].max()}")
    print(f"  配置: {CL_CONFIG}")

    print(f"\n  正在计算 cl_open ...")
    cd_o = CL_O("ETH/USDT", "30m", config=CL_CONFIG)
    cd_o.process_klines(df)

    print(f"  正在计算 cl_pyarmor ...")
    cd_p = CL_P("ETH/USDT", "30m", config=CL_CONFIG)
    cd_p.process_klines(df)

    results = {}

    # ---- 1. 原始 K线 ----
    print(f"\n{'─'*60}")
    print(f"  1. 原始K线 (src_klines)")
    print(f"{'─'*60}")
    results['src_klines'] = cmp_klines("src_klines", cd_o.get_src_klines(), cd_p.get_src_klines())

    # ---- 2. 缠论K线 ----
    print(f"\n{'─'*60}")
    print(f"  2. 缠论K线 (cl_klines)")
    print(f"{'─'*60}")
    results['cl_klines'] = cmp_cl_klines("cl_klines", cd_o.get_cl_klines(), cd_p.get_cl_klines())

    # ---- 3. 分型 ----
    print(f"\n{'─'*60}")
    print(f"  3. 分型 (fxs)")
    print(f"{'─'*60}")
    results['fxs'] = cmp_fxs("fxs", cd_o.get_fxs(), cd_p.get_fxs())

    # ---- 4. MACD 指标 ----
    print(f"\n{'─'*60}")
    print(f"  4. MACD 指标")
    print(f"{'─'*60}")
    results['macd'] = cmp_macd("MACD", cd_o.get_idx(), cd_p.get_idx())

    # ---- 5. 笔 ----
    print(f"\n{'─'*60}")
    print(f"  5. 笔 (bis)")
    print(f"{'─'*60}")
    bis_o = cd_o.get_bis()
    bis_p = cd_p.get_bis()
    results['bis'] = cmp_bis("bis", bis_o, bis_p)

    # ---- 6. 笔中枢 ----
    print(f"\n{'─'*60}")
    print(f"  6. 笔中枢 (bi_zss)")
    print(f"{'─'*60}")
    bi_zss_o = cd_o.get_bi_zss()
    bi_zss_p = cd_p.get_bi_zss()
    results['bi_zss'] = cmp_zss("bi_zss", bi_zss_o, bi_zss_p, show_detail=False)

    # ---- 7. 笔 BC/MMD ----
    print(f"\n{'─'*60}")
    print(f"  7. 笔背驰 & 买卖点 (bi BC/MMD)")
    print(f"{'─'*60}")
    results['bi_bc_mmd'] = cmp_bc_mmd("BI", bis_o, bis_p)

    # ---- 8. 线段 ----
    print(f"\n{'─'*60}")
    print(f"  8. 线段 (xds)")
    print(f"{'─'*60}")
    xds_o = cd_o.get_xds()
    xds_p = cd_p.get_xds()
    results['xds'] = cmp_lines("xds", xds_o, xds_p, show_detail=True)

    # ---- 9. 线段中枢 ----
    print(f"\n{'─'*60}")
    print(f"  9. 线段中枢 (xd_zss)")
    print(f"{'─'*60}")
    xd_zss_o = cd_o.get_xd_zss()
    xd_zss_p = cd_p.get_xd_zss()
    results['xd_zss'] = cmp_zss("xd_zss", xd_zss_o, xd_zss_p, show_detail=True)

    # ---- 10. 线段 BC/MMD ----
    print(f"\n{'─'*60}")
    print(f"  10. 线段背驰 & 买卖点 (xd BC/MMD)")
    print(f"{'─'*60}")
    results['xd_bc_mmd'] = cmp_bc_mmd("XD", xds_o, xds_p)

    # ---- 11. 走势段 ----
    print(f"\n{'─'*60}")
    print(f"  11. 走势段 (zsds)")
    print(f"{'─'*60}")
    zsds_o = cd_o.get_zsds()
    zsds_p = cd_p.get_zsds()
    results['zsds'] = cmp_lines("zsds", zsds_o, zsds_p, show_detail=True)

    # ---- 12. 走势段中枢 ----
    print(f"\n{'─'*60}")
    print(f"  12. 走势段中枢 (zsd_zss)")
    print(f"{'─'*60}")
    zsd_zss_o = cd_o.get_zsd_zss()
    zsd_zss_p = cd_p.get_zsd_zss()
    results['zsd_zss'] = cmp_zss("zsd_zss", zsd_zss_o, zsd_zss_p, show_detail=True)

    # ---- 13. 走势段 BC/MMD ----
    print(f"\n{'─'*60}")
    print(f"  13. 走势段背驰 & 买卖点 (zsd BC/MMD)")
    print(f"{'─'*60}")
    results['zsd_bc_mmd'] = cmp_bc_mmd("ZSD", zsds_o, zsds_p)

    # ---- 14. 趋势走势段 ----
    print(f"\n{'─'*60}")
    print(f"  14. 趋势走势段 (qsds)")
    print(f"{'─'*60}")
    qsds_o = cd_o.get_qsds()
    qsds_p = cd_p.get_qsds()
    results['qsds'] = cmp_lines("qsds", qsds_o, qsds_p, show_detail=True)

    # ---- 15. 趋势走势段中枢 ----
    print(f"\n{'─'*60}")
    print(f"  15. 趋势走势段中枢 (qsd_zss)")
    print(f"{'─'*60}")
    qsd_zss_o = cd_o.get_qsd_zss()
    qsd_zss_p = cd_p.get_qsd_zss()
    results['qsd_zss'] = cmp_zss("qsd_zss", qsd_zss_o, qsd_zss_p, show_detail=True)

    # ---- 16. 趋势走势段 BC/MMD ----
    print(f"\n{'─'*60}")
    print(f"  16. 趋势走势段背驰 & 买卖点 (qsd BC/MMD)")
    print(f"{'─'*60}")
    results['qsd_bc_mmd'] = cmp_bc_mmd("QSD", qsds_o, qsds_p)

    # ========== 汇总 ==========
    print(f"\n{'='*90}")
    print(f"  汇      总      报      告")
    print(f"{'='*90}")
    for key, ok in results.items():
        status = "✅ 一致" if ok else "❌ 差异"
        print(f"    {key:20s} {status}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  结果: {passed}/{total} 项一致")
    if passed == total:
        print(f"  🎉 全部通过！")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"  差异项: {', '.join(failed)}")
    print()


if __name__ == "__main__":
    main()
