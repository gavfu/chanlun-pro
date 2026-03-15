"""
诊断：比较 cl_open vs cl_pyarmor 的 bi.high/bi.low 值
（不只比较边界位置，还比较实际价格值）
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
    "zs_bi_type": ["zs_type_bz"],
    "zs_xd_type": ["zs_type_bz"],
}

DATA_FILES = {
    "ETH60": os.path.join(os.path.dirname(__file__), 'test_data', 'ETH_USDT_60m_1000.parquet'),
    "BTC60": os.path.join(os.path.dirname(__file__), 'test_data', 'BTC_USDT_60m_1000.parquet'),
    "BTCd":  os.path.join(os.path.dirname(__file__), 'test_data', 'BTC_USDT_d_500.parquet'),
    "中信日线": os.path.join(os.path.dirname(__file__), 'test_data', 'SH_600030_d.parquet'),
}


def check_bi_values(name, path):
    print(f"\n{'='*80}")
    print(f"  {name} — bi.high/bi.low 对比")
    print(f"{'='*80}")

    df = pd.read_parquet(path)
    cd_o = CL_O("test", "test", config=CL_CONFIG)
    cd_o.process_klines(df)
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)

    bis_o = cd_o.get_bis()
    bis_p = cd_p.get_bis()

    n = min(len(bis_o), len(bis_p))
    
    # Check boundary match first
    boundary_match = 0
    for i in range(n):
        if (bis_o[i].start.k.k_index == bis_p[i].start.k.k_index
                and bis_o[i].end.k.k_index == bis_p[i].end.k.k_index):
            boundary_match += 1
    
    print(f"  笔数: open={len(bis_o)} pyarmor={len(bis_p)}, 边界匹配={boundary_match}/{n}")

    # Now check high/low values for matched bis
    value_diffs = 0
    high_diffs = []
    low_diffs = []
    
    for i in range(n):
        bo, bp = bis_o[i], bis_p[i]
        if (bo.start.k.k_index != bp.start.k.k_index or
                bo.end.k.k_index != bp.end.k.k_index):
            continue  # Skip boundary mismatches
        
        h_match = abs(bo.high - bp.high) < 1e-9
        l_match = abs(bo.low - bp.low) < 1e-9
        sv_match = abs(bo.start.val - bp.start.val) < 1e-9
        ev_match = abs(bo.end.val - bp.end.val) < 1e-9
        
        if not (h_match and l_match and sv_match and ev_match):
            value_diffs += 1
            if len(high_diffs) + len(low_diffs) < 10:  # Limit output
                issues = []
                if not h_match: 
                    issues.append(f"high: {bo.high:.4f}≠{bp.high:.4f}")
                    high_diffs.append(i)
                if not l_match: 
                    issues.append(f"low: {bo.low:.4f}≠{bp.low:.4f}")
                    low_diffs.append(i)
                if not sv_match: issues.append(f"start.val: {bo.start.val:.4f}≠{bp.start.val:.4f}")
                if not ev_match: issues.append(f"end.val: {bo.end.val:.4f}≠{bp.end.val:.4f}")
                print(f"    bi[{i}] ❌ {bo.type} k={bo.start.k.k_index}→{bo.end.k.k_index}: {', '.join(issues)}")

    if value_diffs == 0:
        print(f"  所有 {boundary_match} 个匹配笔的 high/low/start.val/end.val 完全一致 ✅")
    else:
        print(f"  {value_diffs} 个匹配笔的数值有差异")
        
    # Also check FX vals
    fxs_o = cd_o.get_fxs()
    fxs_p = cd_p.get_fxs()
    fx_val_diffs = 0
    for i in range(min(len(fxs_o), len(fxs_p))):
        fo, fp = fxs_o[i], fxs_p[i]
        if fo.k.k_index == fp.k.k_index and abs(fo.val - fp.val) >= 1e-9:
            fx_val_diffs += 1
            if fx_val_diffs <= 5:
                print(f"    FX[{i}] k={fo.k.k_index}: open val={fo.val:.4f}, pyarmor val={fp.val:.4f}")
    if fx_val_diffs == 0:
        print(f"  FX val完全一致 ({min(len(fxs_o), len(fxs_p))}个) ✅")
    else:
        print(f"  FX val有 {fx_val_diffs} 个差异")


def main():
    for name, path in DATA_FILES.items():
        if os.path.exists(path):
            check_bi_values(name, path)

if __name__ == '__main__':
    main()
