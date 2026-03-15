"""
诊断：对比 cl_open pre-split bis 与 cl_pyarmor pre-split bis。
通过重建 pyarmor 的 pre-split bis（合并 is_split 标记的连续笔）来间接获取。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

CL_CONFIG = {
    'bi_type': 'bi_type_old', 'fx_qj': 'fx_qj_k', 'fx_qy': 'fx_qy_three',
    'bi_fx_cgd': 'bi_fx_cgd_yes', 'fx_check_k_nums': 13,
    'bi_split_k_cross_nums': '20,1', 'xd_bzh': 'xd_bzh_no',
}

DATASETS = {
    'BTC60': 'tests/test_data/BTC_USDT_60m_1000.parquet',
    'ETH60': 'tests/test_data/ETH_USDT_60m_1000.parquet',
    'BTC5m': 'tests/test_data/BTC_USDT_5m_1000.parquet',
    'ETH5m': 'tests/test_data/ETH_USDT_5m_1000.parquet',
    'BTCd':  'tests/test_data/BTC_USDT_d_500.parquet',
}


def reconstruct_presplit(bis):
    """从 pyarmor final bis 重建 pre-split bis.
    规则：pyarmor 拆分生成 3 笔且只有第 1 笔标记 is_split。
    所以连续 3 笔中第 1 笔 has is_split → 合并为一笔。
    """
    result = []
    i = 0
    while i < len(bis):
        bi = bis[i]
        if bi.is_split and i + 2 < len(bis):
            # 合并 3 笔为 1 笔
            merged_start = bi.start.k.k_index
            merged_end = bis[i + 2].end.k.k_index
            result.append((bi.type, merged_start, merged_end, 'merged'))
            i += 3
        else:
            result.append((bi.type, bi.start.k.k_index, bi.end.k.k_index, ''))
            i += 1
    return result


for name, path in DATASETS.items():
    df = pd.read_parquet(path)
    
    co = CL_O("test", "test", config=CL_CONFIG)
    co.process_klines(df)
    cp = CL_P("test", "test", config=CL_CONFIG)
    cp.process_klines(df)
    
    # cl_open pre-split bis
    pre_split_o = co._build_bis(co.get_fxs())
    open_presplit = [(bi.type, bi.start.k.k_index, bi.end.k.k_index) for bi in pre_split_o]
    
    # pyarmor reconstructed pre-split
    pyarmor_presplit = reconstruct_presplit(cp.get_bis())
    
    # Compare
    n = min(len(open_presplit), len(pyarmor_presplit))
    first_diff = None
    for j in range(n):
        o = open_presplit[j]
        p = pyarmor_presplit[j]
        if o[0] != p[0] or o[1] != p[1] or o[2] != p[2]:
            first_diff = j
            break
    if first_diff is None and len(open_presplit) != len(pyarmor_presplit):
        first_diff = n
    
    status = "✅" if first_diff is None else "❌"
    print(f"{name:10s} pre-split: open={len(open_presplit)} pyarmor_recon={len(pyarmor_presplit)} {status}")
    
    if first_diff is not None:
        print(f"  首个差异 bi[{first_diff}]:")
        if first_diff < len(open_presplit):
            o = open_presplit[first_diff]
            print(f"    open:    {o[0]} k={o[1]}→{o[2]}")
        else:
            print(f"    open:    (beyond end)")
        if first_diff < len(pyarmor_presplit):
            p = pyarmor_presplit[first_diff]
            print(f"    pyarmor: {p[0]} k={p[1]}→{p[2]} {p[3]}")
        else:
            print(f"    pyarmor: (beyond end)")
        # Show context
        for j in range(max(0, first_diff-2), min(first_diff+5, max(len(open_presplit), len(pyarmor_presplit)))):
            o_str = f"{open_presplit[j][0]:>4} k={open_presplit[j][1]:>4}→{open_presplit[j][2]:>4}" if j < len(open_presplit) else "(end)"
            p_str = f"{pyarmor_presplit[j][0]:>4} k={pyarmor_presplit[j][1]:>4}→{pyarmor_presplit[j][2]:>4} {pyarmor_presplit[j][3]}" if j < len(pyarmor_presplit) else "(end)"
            marker = " ←DIFF" if j == first_diff else ""
            print(f"    [{j:>3}] open: {o_str}   pyarmor: {p_str}{marker}")
