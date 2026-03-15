"""
逐步追踪 _build_zs_bz 的 gg/dd 计算过程
验证 cl_open 和 cl_pyarmor 的 ZS 是否使用相同的初始化和更新逻辑
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

DATA_PATH = os.path.join(os.path.dirname(__file__), 'test_data', 'ETH_USDT_60m_1000.parquet')


def main():
    df = pd.read_parquet(DATA_PATH)

    cd_o = CL_O("test", "test", config=CL_CONFIG)
    cd_o.process_klines(df)
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)

    bis_o = cd_o.get_bis()
    bis_p = cd_p.get_bis()

    print(f"ETH60: {len(bis_o)} bis (cl_open), {len(bis_p)} bis (cl_pyarmor)")

    # Print all bi.high, bi.low for first 20 bis
    print(f"\n{'='*80}")
    print(f"  bi.high / bi.low 对比 (前20条)")
    print(f"{'='*80}")
    for i in range(min(20, len(bis_o), len(bis_p))):
        bo, bp = bis_o[i], bis_p[i]
        h_ok = abs(bo.high - bp.high) < 1e-9
        l_ok = abs(bo.low - bp.low) < 1e-9
        print(f"  bi[{i:>2}] {bo.type:>4} k={bo.start.k.k_index:>3}→{bo.end.k.k_index:>3}: "
              f"h={bo.high:>10.2f}{'✅' if h_ok else '❌'} "
              f"l={bo.low:>10.2f}{'✅' if l_ok else '❌'} "
              f"(pyarmor h={bp.high:.2f} l={bp.low:.2f})")

    # Manually trace ZS[0] construction
    print(f"\n{'='*80}")
    print(f"  手动追踪 ZS 构建 (cl_open 使用 create_dn_zs 逻辑 + 共享边界)")
    print(f"{'='*80}")

    # Simulate DN+shared boundary (what pyarmor appears to do)
    lines = bis_o
    i = 0
    zs_idx = 0
    while i + 2 < len(lines) and zs_idx < 3:  # Only trace first 3 ZSs
        h0, h1, h2 = lines[i].high, lines[i+1].high, lines[i+2].high
        l0, l1, l2 = lines[i].low, lines[i+1].low, lines[i+2].low
        zg = min(h0, h1, h2)
        zd = max(l0, l1, l2)
        
        if zg > zd:
            print(f"\n  --- ZS[{zs_idx}] 从 bi[{i}] 开始 ---")
            print(f"  初始线段: bi[{i}] h={h0:.2f} l={l0:.2f}")
            print(f"            bi[{i+1}] h={h1:.2f} l={l1:.2f}")
            print(f"            bi[{i+2}] h={h2:.2f} l={l2:.2f}")
            print(f"  zg={zg:.2f}, zd={zd:.2f}")
            print(f"  gg_init=max(h)={max(h0,h1,h2):.2f}, dd_init=min(l)={min(l0,l1,l2):.2f}")
            print(f"  如果 gg=zg={zg:.2f}, dd=zd={zd:.2f}")
            
            gg_max = max(h0, h1, h2)
            dd_min = min(l0, l1, l2)
            gg_zg = zg
            dd_zd = zd
            
            line_indices = [i, i+1, i+2]
            j = i + 3
            while j < len(lines):
                hj = lines[j].high
                lj = lines[j].low
                if hj >= zd and lj <= zg:
                    gg_max = max(gg_max, hj)
                    dd_min = min(dd_min, lj)
                    gg_zg = max(gg_zg, hj)
                    dd_zd = min(dd_zd, lj)
                    line_indices.append(j)
                    print(f"  延伸 bi[{j}]: h={hj:.2f} l={lj:.2f}, "
                          f"gg(max)={gg_max:.2f} dd(min)={dd_min:.2f}, "
                          f"gg(zg)={gg_zg:.2f} dd(zd)={dd_zd:.2f}")
                    j += 1
                else:
                    break
            
            done = j < len(lines)
            print(f"  最终: L={len(line_indices)}, lines={line_indices}")
            print(f"  方案A (gg=max, dd=min): gg={gg_max:.2f}, dd={dd_min:.2f}")
            print(f"  方案B (gg=zg, dd=zd):   gg={gg_zg:.2f}, dd={dd_zd:.2f}")
            
            # Compare with actual pyarmor output
            zss_p = cd_p.get_bi_zss()
            if zs_idx < len(zss_p):
                p = zss_p[zs_idx]
                p_lines = [l.index for l in p.lines]
                print(f"  pyarmor 实际: gg={p.gg:.2f}, dd={p.dd:.2f}, "
                      f"zg={p.zg:.2f}, zd={p.zd:.2f}, type={p.type}, lines={p_lines}")
                
                if abs(gg_max - p.gg) < 1e-9 and abs(dd_min - p.dd) < 1e-9:
                    print(f"  → 方案A (gg=max, dd=min) 匹配 ✅")
                elif abs(gg_zg - p.gg) < 1e-9 and abs(dd_zd - p.dd) < 1e-9:
                    print(f"  → 方案B (gg=zg, dd=zd) 匹配 ✅")
                else:
                    print(f"  → 两种方案都不匹配 ❌")
                    # Try mixed: gg=max, dd=zd
                    if abs(gg_max - p.gg) < 1e-9 and abs(dd_zd - p.dd) < 1e-9:
                        print(f"  → 混合方案 (gg=max, dd=zd) 匹配 ✅")
                    elif abs(gg_zg - p.gg) < 1e-9 and abs(dd_min - p.dd) < 1e-9:
                        print(f"  → 混合方案 (gg=zg, dd=min) 匹配 ✅")
                    else:
                        print(f"  → 所有方案都不匹配")
            
            zs_idx += 1
            i = j - 1  # shared boundary
        else:
            i += 1

    # Now do the same for BTCd
    print(f"\n\n{'='*80}")
    print(f"  BTCd — 追踪")
    print(f"{'='*80}")
    
    btcd_path = os.path.join(os.path.dirname(__file__), 'test_data', 'BTC_USDT_d_500.parquet')
    df2 = pd.read_parquet(btcd_path)
    
    cd_o2 = CL_O("test", "test", config=CL_CONFIG)
    cd_o2.process_klines(df2)
    cd_p2 = CL_P("test", "test", config=CL_CONFIG)
    cd_p2.process_klines(df2)
    
    bis_o2 = cd_o2.get_bis()
    lines = bis_o2
    
    i = 0
    zs_idx = 0
    while i + 2 < len(lines) and zs_idx < 3:
        h0, h1, h2 = lines[i].high, lines[i+1].high, lines[i+2].high
        l0, l1, l2 = lines[i].low, lines[i+1].low, lines[i+2].low
        zg = min(h0, h1, h2)
        zd = max(l0, l1, l2)
        
        if zg > zd:
            print(f"\n  --- ZS[{zs_idx}] 从 bi[{i}] 开始 ---")
            print(f"  初始: bi[{i}] h={h0:.2f} l={l0:.2f}")
            print(f"        bi[{i+1}] h={h1:.2f} l={l1:.2f}")
            print(f"        bi[{i+2}] h={h2:.2f} l={l2:.2f}")
            print(f"  zg={zg:.2f}, zd={zd:.2f}")
            
            gg_max = max(h0, h1, h2)
            dd_min = min(l0, l1, l2)
            gg_zg = zg
            dd_zd = zd
            
            line_indices = [i, i+1, i+2]
            j = i + 3
            while j < len(lines):
                hj = lines[j].high
                lj = lines[j].low
                if hj >= zd and lj <= zg:
                    gg_max = max(gg_max, hj)
                    dd_min = min(dd_min, lj)
                    gg_zg = max(gg_zg, hj)
                    dd_zd = min(dd_zd, lj)
                    line_indices.append(j)
                    j += 1
                else:
                    break
            
            print(f"  最终: L={len(line_indices)}, lines={line_indices}")
            print(f"  方案A (gg=max, dd=min): gg={gg_max:.2f}, dd={dd_min:.2f}")
            print(f"  方案B (gg=zg, dd=zd):   gg={gg_zg:.2f}, dd={dd_zd:.2f}")
            
            zss_p2 = cd_p2.get_bi_zss()
            if zs_idx < len(zss_p2):
                p = zss_p2[zs_idx]
                p_lines = [l.index for l in p.lines]
                print(f"  pyarmor: gg={p.gg:.2f}, dd={p.dd:.2f}, zg={p.zg:.2f}, zd={p.zd:.2f}, lines={p_lines}")
                
                if abs(gg_max - p.gg) < 1e-9 and abs(dd_min - p.dd) < 1e-9:
                    print(f"  → 方案A 匹配 ✅")
                elif abs(gg_zg - p.gg) < 1e-9 and abs(dd_zd - p.dd) < 1e-9:
                    print(f"  → 方案B 匹配 ✅")
                elif abs(gg_max - p.gg) < 1e-9 and abs(dd_zd - p.dd) < 1e-9:
                    print(f"  → 混合 (gg=max, dd=zd) 匹配 ✅")
                elif abs(gg_zg - p.gg) < 1e-9 and abs(dd_min - p.dd) < 1e-9:
                    print(f"  → 混合 (gg=zg, dd=min) 匹配 ✅")
                else:
                    print(f"  → 所有方案不匹配 ❌")
            
            zs_idx += 1
            i = j - 1
        else:
            i += 1


if __name__ == '__main__':
    main()
