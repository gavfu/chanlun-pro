"""Detailed analysis of pyarmor segment splitting patterns"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

CL_CONFIG = {
    "bi_type": "bi_type_old", "fx_qj": "fx_qj_k", "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes", "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1", "xd_bzh": "xd_bzh_no",
}

def analyze_all_splits(name, data_path):
    """Show ALL pyarmor segments with their split status"""
    df = pd.read_parquet(data_path)
    cl_p = CL_P("test", "test", config=CL_CONFIG)
    cl_p.process_klines(df)
    cl_o = CL_O("test", "test", config=CL_CONFIG)
    cl_o.process_klines(df)
    
    xds_p = cl_p.get_xds()
    xds_o = cl_o.get_xds()
    bis_p = cl_p.get_bis()
    
    print(f"\n{'='*80}")
    print(f"=== {name} PYARMOR segments (all) ===")
    for i, xd in enumerate(xds_p):
        split_tag = f" SPLIT='{xd.is_split}'" if xd.is_split else ""
        bi_count = xd.end_line.index - xd.start_line.index + 1
        print(f"  xd[{i:>2}] {xd.type:>4} bi[{xd.start_line.index:>3}→{xd.end_line.index:>3}] ({bi_count:>2} BIs) done={xd.done}{split_tag}")

    # Analyze each split segment: what ZS pattern exists BEFORE splitting
    print(f"\n  --- Split analysis (what the unsplit segment looks like) ---")
    
    # Group consecutive split segments
    i = 0
    while i < len(xds_p):
        xd = xds_p[i]
        if xd.is_split:
            # Find all consecutive split segments
            group_start = i
            group = [xd]
            j = i + 1
            while j < len(xds_p) and xds_p[j].is_split:
                group.append(xds_p[j])
                j += 1
            group_end = j - 1
            
            # The "original" unsplit segment would span from group_start to first non-split after
            orig_start = group[0].start_line.index
            # If next segment after group is non-split, it might be the continuation
            # But typically the split creates entirely new segments
            orig_end = group[-1].end_line.index
            
            print(f"\n  SPLIT GROUP xd[{group_start}..{group_end}]: bi[{orig_start}→{orig_end}]")
            for g in group:
                gi = xds_p.index(g)
                print(f"    xd[{gi}] {g.type:>4} bi[{g.start_line.index}→{g.end_line.index}] split='{g.is_split}'")
            
            # Show BIs in this range
            range_bis = bis_p[orig_start:orig_end + 1]
            print(f"    BIs in range:")
            for b in range_bis:
                print(f"      bi[{b.index:>3}] {b.type:>4} h={b.high:>12.1f} l={b.low:>12.1f}")
            
            # Build ZS in range
            print(f"    Zhongshu analysis:")
            _analyze_zs(range_bis, group[0].type)
            
            # Check bi_pohuai
            print(f"    Bi pohuai analysis:")
            _analyze_bi_pohuai(range_bis, group[0].type)
            
            i = j
        else:
            i += 1


def _analyze_zs(seg_bis, xd_type):
    """Build zhongshu from BIs"""
    i = 0
    while i < len(seg_bis) - 2:
        bi1 = seg_bis[i]
        bi2 = seg_bis[i + 1]
        bi3 = seg_bis[i + 2]
        
        zg = min(bi1.high, bi2.high, bi3.high)
        zd = max(bi1.low, bi2.low, bi3.low)
        
        if zg > zd:
            gg = max(bi1.high, bi2.high, bi3.high)
            dd = min(bi1.low, bi2.low, bi3.low)
            zs_type = "up" if bi1.type == "down" else "down"
            lines = [bi1, bi2, bi3]
            
            j = i + 3
            while j < len(seg_bis):
                bj = seg_bis[j]
                if bj.low < zg and bj.high > zd:
                    lines.append(bj)
                    gg = max(gg, bj.high)
                    dd = min(dd, bj.low)
                    j += 1
                else:
                    break
            
            dir_match = "SAME" if zs_type == xd_type else "DIFF"
            print(f"      ZS bi[{bi1.index}→{lines[-1].index}] type={zs_type} lines={len(lines)} zg={zg:.1f} zd={zd:.1f} dir={dir_match}")
            if len(lines) > 11:
                print(f"        *** lines > 11!")
            
            i = j
        else:
            i += 1


def _analyze_bi_pohuai(seg_bis, xd_type):
    """Check for bi pohuai (bi destruction) patterns"""
    for i in range(len(seg_bis)):
        b = seg_bis[i]
        if i + 1 < len(seg_bis):
            next_b = seg_bis[i + 1]
            # Check if next bi breaks the segment direction
            if xd_type == "up" and b.type == "down" and next_b.type == "up":
                # In up segment, a down bi followed by up bi - check if down bi goes below prev up start
                if i >= 1:
                    prev_up = seg_bis[i - 1]
                    if prev_up.type == "up" and b.low < prev_up.low:
                        print(f"      BREAK at bi[{b.index}]: down low={b.low:.1f} < prev up low={prev_up.low:.1f}")
            elif xd_type == "down" and b.type == "up" and next_b.type == "down":
                if i >= 1:
                    prev_down = seg_bis[i - 1]
                    if prev_down.type == "down" and b.high > prev_down.high:
                        print(f"      BREAK at bi[{b.index}]: up high={b.high:.1f} > prev down high={prev_down.high:.1f}")


for name, path in [
    ("BTC60", "tests/test_data/BTC_USDT_60m_1000.parquet"),
    ("ETH60", "tests/test_data/ETH_USDT_60m_1000.parquet"),
    ("ETH5m", "tests/test_data/ETH_USDT_5m_1000.parquet"),
]:
    analyze_all_splits(name, path)
