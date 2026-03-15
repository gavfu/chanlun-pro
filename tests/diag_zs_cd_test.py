"""
测试 pyarmor 的 zs_cd 默认值
比较 default vs explicit zs_cd_three vs explicit zs_cd_more
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_pyarmor import CL as CL_P

BASE_CONFIG = {
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


def fmt_zs(zs):
    lidxs = [l.index for l in zs.lines]
    return (f"zg={zs.zg:.2f} zd={zs.zd:.2f} gg={zs.gg:.2f} dd={zs.dd:.2f} "
            f"type={zs.type} L={zs.line_num} lines={lidxs}")


def main():
    df = pd.read_parquet(DATA_PATH)

    configs = {
        "default": dict(BASE_CONFIG),
        "zs_cd_three": {**BASE_CONFIG, "zs_cd": "zs_cd_three"},
        "zs_cd_more": {**BASE_CONFIG, "zs_cd": "zs_cd_more"},
    }

    results = {}
    for label, config in configs.items():
        cd = CL_P("test", "test", config=config)
        cd.process_klines(df)
        results[label] = cd.get_bi_zss()
        print(f"\n{'='*80}")
        print(f"  pyarmor ({label}): {len(results[label])} ZSs")
        print(f"{'='*80}")
        for i, zs in enumerate(results[label]):
            print(f"  [{i}] {fmt_zs(zs)}")

    # Compare
    print(f"\n{'='*80}")
    print(f"  对比: default vs three, default vs more")
    print(f"{'='*80}")

    for i in range(max(len(results["default"]), len(results["zs_cd_three"]), len(results["zs_cd_more"]))):
        d = results["default"][i] if i < len(results["default"]) else None
        t = results["zs_cd_three"][i] if i < len(results["zs_cd_three"]) else None
        m = results["zs_cd_more"][i] if i < len(results["zs_cd_more"]) else None

        d_lines = [l.index for l in d.lines] if d else []
        t_lines = [l.index for l in t.lines] if t else []
        m_lines = [l.index for l in m.lines] if m else []

        match_dt = "✅" if (d_lines == t_lines and d and t and
                           abs(d.zg - t.zg) < 1e-9 and abs(d.zd - t.zd) < 1e-9 and
                           abs(d.gg - t.gg) < 1e-9 and abs(d.dd - t.dd) < 1e-9) else "❌"
        match_dm = "✅" if (d_lines == m_lines and d and m and
                           abs(d.zg - m.zg) < 1e-9 and abs(d.zd - m.zd) < 1e-9 and
                           abs(d.gg - m.gg) < 1e-9 and abs(d.dd - m.dd) < 1e-9) else "❌"

        print(f"  [{i}] default==three {match_dt}  default==more {match_dm}")


if __name__ == '__main__':
    main()
