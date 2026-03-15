"""
分析 pyarmor ZS type 字段的规律
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
]

def main():
    for name, fname in DATASETS:
        path = os.path.join(os.path.dirname(__file__), 'test_data', fname)
        df = pd.read_parquet(path)

        cd = CL_P("test", "test", config=CL_CONFIG)
        cd.process_klines(df)
        zss = cd.get_bi_zss()
        bis = cd.get_bis()

        print(f"\n{'='*80}")
        print(f"  {name}: {len(zss)} ZSs")
        print(f"{'='*80}")

        prev_zs = None
        for idx, zs in enumerate(zss):
            lines_idx = [l.index for l in zs.lines]
            enter_bi = bis[lines_idx[0]]
            enter_dir = enter_bi.type  # up or down

            # Relations to previous ZS
            rel = "首个"
            if prev_zs:
                if zs.zd > prev_zs.zg:
                    rel = "高于前ZS"
                elif zs.zg < prev_zs.zd:
                    rel = "低于前ZS"
                else:
                    rel = "与前ZS重叠"

            # First/last line directions
            first_line = bis[lines_idx[0]]
            last_line = bis[lines_idx[-1]]
            
            # Overlap lines
            overlap_start = bis[lines_idx[1]]

            # Exit direction: the line after the ZS
            exit_idx = lines_idx[-1] + 1
            exit_dir = bis[exit_idx].type if exit_idx < len(bis) else "N/A"

            print(f"  [{idx}] type={zs.type:>4} | "
                  f"enter={enter_dir:>4} bi[{lines_idx[0]:>2}] | "
                  f"overlap1={overlap_start.type:>4} bi[{lines_idx[1]:>2}] | "
                  f"last={last_line.type:>4} bi[{lines_idx[-1]:>2}] | "
                  f"exit={exit_dir:>4} | "
                  f"rel={rel}")

            prev_zs = zs


if __name__ == '__main__':
    main()
