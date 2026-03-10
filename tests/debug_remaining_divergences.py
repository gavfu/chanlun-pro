"""
Investigate remaining XD divergences:
1. BTCd xd[1]: down[22→24] vs down[22→28]
2. ETH60 xd[10]: down[52→57] vs down[52→56]  
3. BTC60 xd[10]: down[54→57] vs down[54→56]
4. BTC5m xd[9]: down[54→56] vs down[54→64]
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

def investigate(name, data_path):
    df = pd.read_parquet(data_path)
    cd_o = CL_O("test", "test", config=CL_CONFIG)
    cd_o.process_klines(df)
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    
    xds_o = cd_o.get_xds()
    xds_p = cd_p.get_xds()
    bis_o = cd_o.get_bis()
    bis_p = cd_p.get_bis()
    
    print(f"\n{'='*80}")
    print(f"  {name}: BI_open={len(bis_o)} BI_pyarmor={len(bis_p)}")
    print(f"{'='*80}")
    
    # Find divergent XDs
    for i in range(min(len(xds_o), len(xds_p))):
        o = xds_o[i]
        p = xds_p[i]
        if o.start_line.index != p.start_line.index or o.end_line.index != p.end_line.index:
            print(f"\n  xd[{i}] DIVERGENCE:")
            print(f"    open:    {o.type} bi[{o.start_line.index}→{o.end_line.index}] done={o.done}")
            print(f"    pyarmor: {p.type} bi[{p.start_line.index}→{p.end_line.index}] done={p.done}")
            
            # Show BIs around divergence area
            start = min(o.start_line.index, p.start_line.index) - 2
            end = max(o.end_line.index, p.end_line.index) + 4
            start = max(0, start)
            end = min(len(bis_o), end)
            
            print(f"\n    BIs in range [{start}..{end-1}]:")
            for j in range(start, end):
                bi = bis_o[j]
                markers = []
                if j == o.start_line.index: markers.append("O_START")
                if j == o.end_line.index: markers.append("O_END")
                if j == p.start_line.index: markers.append("P_START")
                if j == p.end_line.index: markers.append("P_END")
                m = f"  ← {','.join(markers)}" if markers else ""
                print(f"      bi[{j:>3}] {bi.type:>4} h={bi.high:<12.2f} l={bi.low:<12.2f}{m}")
            
            # For the last incomplete XD case, check if pyarmor end_bi is real
            if not o.done and not p.done:
                print(f"\n    Both incomplete. Checking last TZXL construction...")
                # Check what _find_xd_end returns for both
                prev_xd_o = xds_o[i-1] if i > 0 else None
                prev_xd_p = xds_p[i-1] if i > 0 else None
                if prev_xd_o and prev_xd_p:
                    print(f"    Previous XD open:    {prev_xd_o.type} bi[{prev_xd_o.start_line.index}→{prev_xd_o.end_line.index}]")
                    print(f"    Previous XD pyarmor: {prev_xd_p.type} bi[{prev_xd_p.start_line.index}→{prev_xd_p.end_line.index}]")

            # For done XDs that differ, check TZXL
            if o.done:
                # Run TZXL analysis
                xd_type = o.type
                start_bi = o.start_line.index
                fx_type = 'ding' if xd_type == 'up' else 'di'
                
                lines = []
                for k in range(start_bi, len(bis_o)):
                    lines.append(bis_o[k])
                
                try:
                    tzxls, xlfxs = cd_p._xd_cal_line_xlfx(lines, fx_type, 'no_bh')
                    print(f"\n    Pyarmor TZXL ({fx_type}) from bi[{start_bi}]:")
                    for t, xl in enumerate(tzxls):
                        bad = "BAD" if xl.line_bad else "   "
                        print(f"      TZXL[{t}] {bad} bi[{xl.lines[0].index}..{xl.lines[-1].index}] "
                              f"max={xl.max:<12.2f} min={xl.min:<12.2f}")
                    
                    print(f"    Pyarmor XLFX ({fx_type}):")
                    for f_idx, fx in enumerate(xlfxs):
                        print(f"      XLFX[{f_idx}] {fx.type} @ TZXL[{fx.xl.lines[0].index}..{fx.xl.lines[-1].index}] "
                              f"bad={fx.xl.line_bad}")
                except Exception as e:
                    print(f"    TZXL analysis error: {e}")

for name in ["BTCd", "ETH60", "BTC60", "BTC5m"]:
    investigate(name, TEST_DATA[name])
