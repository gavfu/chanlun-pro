"""Trace segment zhongshu to understand splitting"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P
from chanlun.cl_interface import *

CL_CONFIG = {
    "bi_type": "bi_type_old", "fx_qj": "fx_qj_k", "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes", "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1", "xd_bzh": "xd_bzh_no",
}

def analyze_zs_in_segment(name, data_path, xd_idx):
    """Analyze zhongshu within a specific segment"""
    df = pd.read_parquet(data_path)
    
    cl = CL_O("test", "test", config=CL_CONFIG)
    cl.process_klines(df)
    cl_p = CL_P("test", "test", config=CL_CONFIG)
    cl_p.process_klines(df)
    
    xds_o = cl.get_xds()
    xds_p = cl_p.get_xds()
    bis = cl.get_bis()
    
    print(f"\n{'='*70}")
    print(f"=== {name} open xd[{xd_idx}] ===")
    
    xd_o = xds_o[xd_idx]
    start_idx = xd_o.start_line.index  # BI sequential index
    end_idx = xd_o.end_line.index
    print(f"  Open: {xd_o.type} bi[{start_idx}→{end_idx}]")
    
    # Get segment BIs
    seg_bis = bis[start_idx:end_idx + 1]
    
    print(f"  BIs in segment ({len(seg_bis)}):")
    for b in seg_bis[:20]:
        print(f"    bi[{b.index:2d}] {b.type:4s} h={b.high:10.1f} l={b.low:10.1f}")
    if len(seg_bis) > 20:
        print(f"    ... ({len(seg_bis) - 20} more)")
    
    # Build ZS within segment BIs (reuse filtered seg_bis from above)
    print(f"\n  Building zhongshu from {len(seg_bis)} BIs:")
    
    # Manual ZS construction - find overlapping BI triplets
    zs_list = []
    i = 0
    while i < len(seg_bis) - 2:
        bi1 = seg_bis[i]
        bi2 = seg_bis[i + 1]
        bi3 = seg_bis[i + 2]
        
        # ZS = overlap of 3 consecutive BIs
        zg = min(bi1.high, bi2.high, bi3.high)
        zd = max(bi1.low, bi2.low, bi3.low)
        
        if zg > zd:
            # Valid ZS
            gg = max(bi1.high, bi2.high, bi3.high)
            dd = min(bi1.low, bi2.low, bi3.low)
            zs_type = "up" if bi1.type == "down" else "down"
            lines_in_zs = [bi1, bi2, bi3]
            
            # Extend ZS
            j = i + 3
            while j < len(seg_bis):
                bj = seg_bis[j]
                if bj.low < zg and bj.high > zd:  # overlaps with ZS
                    lines_in_zs.append(bj)
                    gg = max(gg, bj.high)
                    dd = min(dd, bj.low)
                    j += 1
                else:
                    break
            
            zs_info = {
                'start_bi': bi1.index,
                'end_bi': lines_in_zs[-1].index,
                'zg': zg, 'zd': zd, 'gg': gg, 'dd': dd,
                'type': zs_type,
                'line_num': len(lines_in_zs),
            }
            zs_list.append(zs_info)
            print(f"    ZS: bi[{zs_info['start_bi']}→{zs_info['end_bi']}] type={zs_info['type']} lines={zs_info['line_num']} zg={zs_info['zg']:.1f} zd={zs_info['zd']:.1f} gg={zs_info['gg']:.1f} dd={zs_info['dd']:.1f}")
            
            # Check split conditions
            if zs_info['line_num'] > 11:
                print(f"      *** 段内中枢线段超过11 (超过{zs_info['line_num']})")
            if zs_info['type'] != xd_o.type:
                print(f"      *** 段内不同向中枢 (ZS type={zs_info['type']} vs XD type={xd_o.type})")
            
            i = j
        else:
            i += 1
    
    # Show pyarmor xds that correspond to this open segment
    print(f"\n  Pyarmor segments covering same BI range:")
    for j, xd_p in enumerate(xds_p):
        if xd_p.end_line.index >= start_idx and xd_p.start_line.index <= end_idx:
            print(f"    pyarmor xd[{j}] {xd_p.type} bi[{xd_p.start_line.index}→{xd_p.end_line.index}] split='{xd_p.is_split}'")

# BTC60 xd[1] - 段内不同向中枢拆分
analyze_zs_in_segment("BTC60", "tests/test_data/BTC_USDT_60m_1000.parquet", 1)

# ETH60 xd[5] - 段内中枢线段超过11
analyze_zs_in_segment("ETH60", "tests/test_data/ETH_USDT_60m_1000.parquet", 5)

# BTC5m xd[9] - last 2 segments 
analyze_zs_in_segment("BTC5m", "tests/test_data/BTC_USDT_5m_1000.parquet", 9)
