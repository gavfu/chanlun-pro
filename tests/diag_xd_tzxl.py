"""
诊断脚本：深入对比 pyarmor 和 cl_open 线段的特征序列
用 BTCd (BI完全相同) 作为分析对象
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

def analyze_case(name, data_path):
    df = pd.read_parquet(data_path)
    cd_o = CL_O("test", "test", config=CL_CONFIG)
    cd_o.process_klines(df)
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    
    bis_o = cd_o.get_bis()
    bis_p = cd_p.get_bis()
    xds_o = cd_o.get_xds()
    xds_p = cd_p.get_xds()
    
    print(f"\n{'='*80}")
    print(f"  {name}: BI={len(bis_o)}/{len(bis_p)} XD={len(xds_o)}/{len(xds_p)}")
    print(f"{'='*80}")
    
    print(f"\n--- Pyarmor XDs with TZXL details ---")
    for i, xd in enumerate(xds_p):
        print(f"\n  xd[{i}] {xd.type} bi[{xd.start_line.index}→{xd.end_line.index}] "
              f"done={xd.done} is_split='{xd.is_split}'")
        
        # Print ding_fx info
        if xd.ding_fx:
            dfx = xd.ding_fx
            print(f"    ding_fx: type={dfx.type} done={dfx.done} qk={dfx.qk} "
                  f"is_line_bad={dfx.is_line_bad}")
            for j, xl in enumerate(dfx.xls):
                if xl is not None:
                    lines_str = ",".join([f"bi[{l.index}]" for l in xl.lines])
                    print(f"      xls[{j}]: max={xl.max:.1f} min={xl.min:.1f} "
                          f"bad={xl.line_bad} bh={xl.bh_direction} lines=[{lines_str}]")
        
        # Print di_fx info  
        if xd.di_fx:
            dfx = xd.di_fx
            print(f"    di_fx: type={dfx.type} done={dfx.done} qk={dfx.qk} "
                  f"is_line_bad={dfx.is_line_bad}")
            for j, xl in enumerate(dfx.xls):
                if xl is not None:
                    lines_str = ",".join([f"bi[{l.index}]" for l in xl.lines])
                    print(f"      xls[{j}]: max={xl.max:.1f} min={xl.min:.1f} "
                          f"bad={xl.line_bad} bh={xl.bh_direction} lines=[{lines_str}]")
        
        # Print tzxls
        if xd.tzxls:
            print(f"    tzxls ({len(xd.tzxls)} elements):")
            for j, xl in enumerate(xd.tzxls):
                lines_str = ",".join([f"bi[{l.index}]" for l in xl.lines])
                print(f"      [{j}] max={xl.max:.1f} min={xl.min:.1f} "
                      f"bad={xl.line_bad} bh={xl.bh_direction} "
                      f"is_up={xl.is_up_line} lines=[{lines_str}]")
    
    print(f"\n--- Open XDs with TZXL details ---")
    for i, xd in enumerate(xds_o):
        print(f"\n  xd[{i}] {xd.type} bi[{xd.start_line.index}→{xd.end_line.index}] "
              f"done={xd.done} is_split='{xd.is_split}'")
        
        if xd.ding_fx:
            dfx = xd.ding_fx
            print(f"    ding_fx: type={dfx.type} done={dfx.done} qk={dfx.qk} "
                  f"is_line_bad={dfx.is_line_bad}")
            for j, xl in enumerate(dfx.xls):
                if xl is not None:
                    lines_str = ",".join([f"bi[{l.index}]" for l in xl.lines])
                    print(f"      xls[{j}]: max={xl.max:.1f} min={xl.min:.1f} "
                          f"bad={xl.line_bad} bh={xl.bh_direction} lines=[{lines_str}]")
        
        if xd.di_fx:
            dfx = xd.di_fx
            print(f"    di_fx: type={dfx.type} done={dfx.done} qk={dfx.qk} "
                  f"is_line_bad={dfx.is_line_bad}")
            for j, xl in enumerate(dfx.xls):
                if xl is not None:
                    lines_str = ",".join([f"bi[{l.index}]" for l in xl.lines])
                    print(f"      xls[{j}]: max={xl.max:.1f} min={xl.min:.1f} "
                          f"bad={xl.line_bad} bh={xl.bh_direction} lines=[{lines_str}]")
        
        if xd.tzxls:
            print(f"    tzxls ({len(xd.tzxls)} elements):")
            for j, xl in enumerate(xd.tzxls):
                lines_str = ",".join([f"bi[{l.index}]" for l in xl.lines])
                print(f"      [{j}] max={xl.max:.1f} min={xl.min:.1f} "
                      f"bad={xl.line_bad} bh={xl.bh_direction} "
                      f"is_up={xl.is_up_line} lines=[{lines_str}]")


    # 打印所有笔供参考
    print(f"\n--- All BIs (pyarmor) ---")
    for bi in bis_p:
        print(f"  bi[{bi.index:>3}] {bi.type:>4} k[{bi.start.k.k_index:>5}→{bi.end.k.k_index:>5}] "
              f"h={bi.high:<12.1f} l={bi.low:<12.1f}")

if __name__ == "__main__":
    # Use BTCd and ETH60 as they have PERFECT BI matches
    cases = {
        "BTCd": "tests/test_data/BTC_USDT_d_500.parquet",
        "ETH60": "tests/test_data/ETH_USDT_60m_1000.parquet",
    }
    for name in (sys.argv[1:] if len(sys.argv) > 1 else cases.keys()):
        if name in cases:
            analyze_case(name, cases[name])
