"""
诊断脚本：全量对比 cl_open 和 cl_pyarmor 的线段输出
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

TEST_DATA = {
    "BTCd":  "tests/test_data/BTC_USDT_d_500.parquet",
    "ETH60": "tests/test_data/ETH_USDT_60m_1000.parquet",
    "BTC60": "tests/test_data/BTC_USDT_60m_1000.parquet",
    "BTC5m": "tests/test_data/BTC_USDT_5m_1000.parquet",
    "ETH5m": "tests/test_data/ETH_USDT_5m_1000.parquet",
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

def run_case(name, data_path):
    df = pd.read_parquet(data_path)
    cd_open = CL_O("test", "test", config=CL_CONFIG)
    cd_open.process_klines(df)
    cd_pyarmor = CL_P("test", "test", config=CL_CONFIG)
    cd_pyarmor.process_klines(df)
    
    xds_o = cd_open.get_xds()
    xds_p = cd_pyarmor.get_xds()
    bis_o = cd_open.get_bis()
    bis_p = cd_pyarmor.get_bis()
    
    bi_match = len(bis_o) == len(bis_p)
    bi_boundary_match = True
    if bi_match:
        for i in range(len(bis_o)):
            if bis_o[i].start.k.k_index != bis_p[i].start.k.k_index or bis_o[i].end.k.k_index != bis_p[i].end.k.k_index:
                bi_boundary_match = False
                break
    
    print(f"\n{'='*80}")
    print(f"  {name}: BI={len(bis_o)}/{len(bis_p)} {'✅' if bi_match else '❌'}  "
          f"BI boundaries={'PERFECT' if bi_boundary_match else 'differ'}  "
          f"XD={len(xds_o)}/{len(xds_p)} {'✅' if len(xds_o)==len(xds_p) else '❌'}")
    print(f"{'='*80}")
    
    # 对齐线段列表 - 通过 start_bi_index 寻找匹配
    max_len = max(len(xds_o), len(xds_p))
    
    for i in range(max_len):
        o = xds_o[i] if i < len(xds_o) else None
        p = xds_p[i] if i < len(xds_p) else None
        
        if o and p:
            start_match = o.start_line.index == p.start_line.index
            end_match = o.end_line.index == p.end_line.index
            match = "✅" if (start_match and end_match and o.type == p.type) else "❌"
            detail = ""
            if not start_match:
                detail += f" START_DIFF({o.start_line.index}vs{p.start_line.index})"
            if not end_match:
                detail += f" END_DIFF({o.end_line.index}vs{p.end_line.index})"
        else:
            match = "❌"
            detail = " MISSING"
        
        o_str = f"{o.type:>4} bi[{o.start_line.index}→{o.end_line.index}] done={o.done}" if o else "---"
        p_str = f"{p.type:>4} bi[{p.start_line.index}→{p.end_line.index}] done={p.done}" if p else "---"
        
        print(f"  xd[{i:>2}] {match} open: {o_str:40} pyarmor: {p_str:40}{detail}")
    
    # 如果 BI 完全相同但 XD 不同，做详细的特征序列分析
    if bi_match and bi_boundary_match and len(xds_o) != len(xds_p):
        print(f"\n  *** BI完全相同但XD不同 → 纯线段算法差异 ***")
        # 找到第一个分歧点
        for i in range(min(len(xds_o), len(xds_p))):
            o = xds_o[i]
            p = xds_p[i]
            if o.start_line.index != p.start_line.index or o.end_line.index != p.end_line.index:
                print(f"\n  首个分歧在 xd[{i}]:")
                print(f"    open:    {o.type} bi[{o.start_line.index}→{o.end_line.index}]")
                print(f"    pyarmor: {p.type} bi[{p.start_line.index}→{p.end_line.index}]")
                
                # 打印该区域的笔
                start_idx = min(o.start_line.index, p.start_line.index) - 2
                end_idx = max(o.end_line.index, p.end_line.index) + 4
                start_idx = max(0, start_idx)
                end_idx = min(len(bis_o), end_idx)
                print(f"\n    区域笔 bi[{start_idx}..{end_idx-1}]:")
                for j in range(start_idx, end_idx):
                    bi = bis_o[j]
                    marker = ""
                    if j == o.start_line.index: marker += " ←O_START"
                    if j == o.end_line.index: marker += " ←O_END"
                    if j == p.start_line.index: marker += " ←P_START"
                    if j == p.end_line.index: marker += " ←P_END"
                    print(f"      bi[{j:>3}] {bi.type:>4} {bi.start.k.k_index:>5}→{bi.end.k.k_index:>5} "
                          f"h={bi.high:<10.1f} l={bi.low:<10.1f}{marker}")
                break

if __name__ == "__main__":
    cases = sys.argv[1:] if len(sys.argv) > 1 else TEST_DATA.keys()
    for name in cases:
        if name in TEST_DATA:
            run_case(name, TEST_DATA[name])
