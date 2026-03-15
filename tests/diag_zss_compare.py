"""
诊断中枢差异：在已验证笔完全匹配的数据集上，对比 cl_open vs cl_pyarmor 的中枢输出
用于隔离中枢算法本身的差异（排除笔/线段分歧的影响）
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

# 已验证笔完全匹配的数据集
DATA_FILES = {
    "BTC60": os.path.join(os.path.dirname(__file__), 'test_data', 'BTC_USDT_60m_1000.parquet'),
    "ETH60": os.path.join(os.path.dirname(__file__), 'test_data', 'ETH_USDT_60m_1000.parquet'),
    "BTCd":  os.path.join(os.path.dirname(__file__), 'test_data', 'BTC_USDT_d_500.parquet'),
    "中信日线": os.path.join(os.path.dirname(__file__), 'test_data', 'SH_600030_d.parquet'),
}


def analyze_zss(label, zss_o, zss_p):
    """详细对比中枢"""
    n_o, n_p = len(zss_o), len(zss_p)
    print(f"  [{label}] open={n_o} pyarmor={n_p} {'✅' if n_o == n_p else '❌'}")

    if n_o == 0 and n_p == 0:
        print(f"    无中枢可比较")
        return True

    max_show = max(n_o, n_p)
    all_match = True
    for i in range(max_show):
        o = zss_o[i] if i < n_o else None
        p = zss_p[i] if i < n_p else None

        if o and p:
            zg_ok = abs(o.zg - p.zg) < 1e-9
            zd_ok = abs(o.zd - p.zd) < 1e-9
            type_ok = o.type == p.type
            line_ok = o.line_num == p.line_num
            done_ok = o.done == p.done

            issues = []
            if not zg_ok: issues.append(f"zg: {o.zg:.2f}≠{p.zg:.2f}")
            if not zd_ok: issues.append(f"zd: {o.zd:.2f}≠{p.zd:.2f}")
            if not type_ok: issues.append(f"type: {o.type}≠{p.type}")
            if not line_ok: issues.append(f"L: {o.line_num}≠{p.line_num}")
            if not done_ok: issues.append(f"done: {o.done}≠{p.done}")

            if issues:
                all_match = False
                print(f"    [{i:>2}] ❌ {', '.join(issues)}")
                # 打印详细
                print(f"         open:    zg={o.zg:.4f} zd={o.zd:.4f} gg={o.gg:.4f} dd={o.dd:.4f} "
                      f"type={o.type} L={o.line_num} done={o.done}")
                print(f"         pyarmor: zg={p.zg:.4f} zd={p.zd:.4f} gg={p.gg:.4f} dd={p.dd:.4f} "
                      f"type={p.type} L={p.line_num} done={p.done}")
                # 打印中枢包含的线段索引
                if o.lines:
                    o_lines_idx = [l.index for l in o.lines]
                    print(f"         open  lines idx: {o_lines_idx}")
                if p.lines:
                    p_lines_idx = [l.index for l in p.lines]
                    print(f"         pyarmor lines idx: {p_lines_idx}")
            else:
                print(f"    [{i:>2}] ✅ zg={o.zg:.2f} zd={o.zd:.2f} type={o.type} L={o.line_num}")
        elif o:
            all_match = False
            print(f"    [{i:>2}] ❌ open only: zg={o.zg:.2f} zd={o.zd:.2f} type={o.type} L={o.line_num}")
        elif p:
            all_match = False
            print(f"    [{i:>2}] ❌ pyarmor only: zg={p.zg:.2f} zd={p.zd:.2f} type={p.type} L={p.line_num}")

    return all_match


def check_dataset(name, path):
    print(f"\n{'='*80}")
    print(f"  数据集: {name}")
    print(f"{'='*80}")

    df = pd.read_parquet(path)
    print(f"  K线数: {len(df)}, 日期: {df['date'].min()} ~ {df['date'].max()}")

    cd_o = CL_O("test", "test", config=CL_CONFIG)
    cd_o.process_klines(df)

    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)

    # 先验证笔
    bis_o = cd_o.get_bis()
    bis_p = cd_p.get_bis()
    bi_match = 0
    for i in range(min(len(bis_o), len(bis_p))):
        if (bis_o[i].start.k.k_index == bis_p[i].start.k.k_index
                and bis_o[i].end.k.k_index == bis_p[i].end.k.k_index):
            bi_match += 1
    bi_ok = bi_match == len(bis_o) == len(bis_p)
    print(f"  笔: open={len(bis_o)} pyarmor={len(bis_p)} 匹配={bi_match} {'✅' if bi_ok else '❌'}")

    # 线段
    xds_o = cd_o.get_xds()
    xds_p = cd_p.get_xds()
    xd_match = 0
    for i in range(min(len(xds_o), len(xds_p))):
        if (xds_o[i].start_line.index == xds_p[i].start_line.index
                and xds_o[i].end_line.index == xds_p[i].end_line.index):
            xd_match += 1
    xd_ok = xd_match == len(xds_o) == len(xds_p)
    print(f"  线段: open={len(xds_o)} pyarmor={len(xds_p)} 匹配={xd_match} {'✅' if xd_ok else '❌'}")

    # 笔中枢
    print(f"\n  --- 笔中枢 (bi_zss) ---")
    bi_zss_o = cd_o.get_bi_zss()
    bi_zss_p = cd_p.get_bi_zss()
    bi_zss_ok = analyze_zss("bi_zss", bi_zss_o, bi_zss_p)

    # 线段中枢
    print(f"\n  --- 线段中枢 (xd_zss) ---")
    xd_zss_o = cd_o.get_xd_zss()
    xd_zss_p = cd_p.get_xd_zss()
    xd_zss_ok = analyze_zss("xd_zss", xd_zss_o, xd_zss_p)

    return bi_ok, xd_ok, bi_zss_ok, xd_zss_ok


def main():
    print(f"{'='*80}")
    print(f"  中枢对比诊断 — 重点检查 ZS type/zg/zd 差异")
    print(f"  配置: {CL_CONFIG}")
    print(f"  注: 未传入 zs_bi_type/zs_xd_type，使用各引擎默认值")
    print(f"{'='*80}")

    results = {}
    for name, path in DATA_FILES.items():
        if os.path.exists(path):
            results[name] = check_dataset(name, path)
        else:
            print(f"\n  ⚠️ 跳过 {name}: 文件不存在 {path}")

    print(f"\n{'='*80}")
    print(f"  汇总")
    print(f"{'='*80}")
    for name, (bi_ok, xd_ok, bi_zss_ok, xd_zss_ok) in results.items():
        print(f"  {name:12s}: 笔 {'✅' if bi_ok else '❌'}  线段 {'✅' if xd_ok else '❌'}  "
              f"笔中枢 {'✅' if bi_zss_ok else '❌'}  线段中枢 {'✅' if xd_zss_ok else '❌'}")


if __name__ == '__main__':
    main()
