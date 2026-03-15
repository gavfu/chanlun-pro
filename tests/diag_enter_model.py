"""
验证假设: pyarmor BZ 使用 "进入段+3段重叠" 模型
- enter = line[i] (进入段，包含在 lines 中但不参与 zg/zd 计算)
- overlap = line[i+1, i+2, i+3] (重叠区间)
- zg = min(h_{i+1}, h_{i+2}, h_{i+3})
- zd = max(l_{i+1}, l_{i+2}, l_{i+3})
- 进入条件: h_i >= zd AND l_i <= zg
- 延伸: j = i+4 起
- gg/dd = max/min of lines[1:-1] (去除首尾)
- 共享边界: i = j - 1
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
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

DATASETS = [
    ("ETH60", "ETH_USDT_60m_1000.parquet"),
    ("BTC60", "BTC_USDT_60m_1000.parquet"),
    ("BTCd",  "BTC_USDT_d_500.parquet"),
    ("中信",  "SH_600030_d.parquet"),
]


def simulate_bz(bis):
    """模拟 enter+overlap 算法"""
    result = []
    i = 0
    while i + 3 < len(bis):
        # overlap = bis[i+1, i+2, i+3]
        h1, l1 = bis[i+1].high, bis[i+1].low
        h2, l2 = bis[i+2].high, bis[i+2].low
        h3, l3 = bis[i+3].high, bis[i+3].low
        zg = min(h1, h2, h3)
        zd = max(l1, l2, l3)

        if zg > zd:
            # Enter condition
            h0, l0 = bis[i].high, bis[i].low
            if h0 >= zd and l0 <= zg:
                # Create ZS: enter + overlap
                line_indices = [i, i+1, i+2, i+3]
                # Extend from j = i+4
                j = i + 4
                while j < len(bis):
                    hj, lj = bis[j].high, bis[j].low
                    if hj >= zd and lj <= zg:
                        line_indices.append(j)
                        j += 1
                    else:
                        break
                done = j < len(bis)

                # gg/dd from lines[1:-1] (exclude enter and last)
                # BUT if done=False (data ends), include last line
                if done and len(line_indices) > 2:
                    inner = line_indices[1:-1]
                else:
                    inner = line_indices[1:]
                if inner:
                    gg = max(bis[k].high for k in inner)
                    dd = min(bis[k].low for k in inner)
                else:
                    gg = zg
                    dd = zd

                result.append({
                    'lines': line_indices,
                    'zg': zg, 'zd': zd,
                    'gg': gg, 'dd': dd,
                    'done': done,
                    'L': len(line_indices),
                })
                i = j - 1  # shared boundary
            else:
                i += 1
        else:
            i += 1
    return result


def main():
    for name, fname in DATASETS:
        path = os.path.join(os.path.dirname(__file__), 'test_data', fname)
        if not os.path.exists(path):
            print(f"  {name}: SKIP (file not found)")
            continue
        df = pd.read_parquet(path)

        cd_p = CL_P("test", "test", config=CL_CONFIG)
        cd_p.process_klines(df)
        zss_p = cd_p.get_bi_zss()
        bis_p = cd_p.get_bis()

        # Simulate
        sim = simulate_bz(bis_p)

        print(f"\n{'='*80}")
        print(f"  {name}: pyarmor={len(zss_p)} ZSs, simulated={len(sim)} ZSs")
        print(f"{'='*80}")

        max_n = max(len(zss_p), len(sim))
        all_match = True
        for idx in range(max_n):
            p = zss_p[idx] if idx < len(zss_p) else None
            s = sim[idx] if idx < len(sim) else None

            if p and s:
                p_lines = [l.index for l in p.lines]
                s_lines = s['lines']
                match_lines = p_lines == s_lines
                match_zg = abs(p.zg - s['zg']) < 1e-6
                match_zd = abs(p.zd - s['zd']) < 1e-6
                match_gg = abs(p.gg - s['gg']) < 1e-6
                match_dd = abs(p.dd - s['dd']) < 1e-6
                match_done = p.done == s['done']

                status = "✅" if all([match_lines, match_zg, match_zd, match_gg, match_dd, match_done]) else "❌"
                if status == "❌":
                    all_match = False

                print(f"  [{idx}] {status}", end="")
                if not match_lines:
                    print(f" lines: sim={s_lines} pyarmor={p_lines}", end="")
                if not match_zg:
                    print(f" zg: {s['zg']:.2f}≠{p.zg:.2f}", end="")
                if not match_zd:
                    print(f" zd: {s['zd']:.2f}≠{p.zd:.2f}", end="")
                if not match_gg:
                    print(f" gg: {s['gg']:.2f}≠{p.gg:.2f}", end="")
                if not match_dd:
                    print(f" dd: {s['dd']:.2f}≠{p.dd:.2f}", end="")
                if not match_done:
                    print(f" done: {s['done']}≠{p.done}", end="")
                if status == "✅":
                    print(f" L={s['L']} lines={s_lines[:4]}{'...' if len(s_lines)>4 else ''}"
                          f" zg={s['zg']:.2f} zd={s['zd']:.2f}"
                          f" gg={s['gg']:.2f} dd={s['dd']:.2f}", end="")
                print()
            elif p and not s:
                print(f"  [{idx}] ❌ pyarmor has ZS but sim doesn't: "
                      f"lines={[l.index for l in p.lines]}")
                all_match = False
            elif s and not p:
                print(f"  [{idx}] ❌ sim has ZS but pyarmor doesn't: lines={s['lines']}")
                all_match = False

        if all_match:
            print(f"  >>> ALL {len(zss_p)} ZSs MATCH! <<<")


if __name__ == '__main__':
    main()
