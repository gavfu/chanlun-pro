"""
测试假设: pyarmor BZ = create_dn_zs (段内中枢)
临时让 _build_zs_bz 直接调用 create_dn_zs，看结果是否匹配
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

def fmt_zs(zs):
    lidxs = [l.index for l in zs.lines]
    return (f"zg={zs.zg:.2f} zd={zs.zd:.2f} gg={zs.gg:.2f} dd={zs.dd:.2f} "
            f"type={zs.type} done={zs.done} L={zs.line_num} lines={lidxs}")

def main():
    for name, fname in DATASETS:
        path = os.path.join(os.path.dirname(__file__), 'test_data', fname)
        df = pd.read_parquet(path)

        # pyarmor with BZ config
        cd_bz = CL_P("test", "test", config=CL_CONFIG)
        cd_bz.process_klines(df)
        zss_bz = cd_bz.get_bi_zss()

        # pyarmor with DN config
        dn_config = dict(CL_CONFIG)
        dn_config["zs_bi_type"] = ["zs_type_dn"]
        cd_dn = CL_P("test", "test", config=dn_config)
        cd_dn.process_klines(df)
        zss_dn = cd_dn.get_bi_zss()

        print(f"\n{'='*80}")
        print(f"  {name}: BZ={len(zss_bz)} ZSs, DN={len(zss_dn)} ZSs")
        print(f"{'='*80}")

        max_n = max(len(zss_bz), len(zss_dn))
        for i in range(max_n):
            bz = zss_bz[i] if i < len(zss_bz) else None
            dn = zss_dn[i] if i < len(zss_dn) else None
            
            bz_str = fmt_zs(bz) if bz else "---"
            dn_str = fmt_zs(dn) if dn else "---"
            
            match = ""
            if bz and dn:
                bz_lines = [l.index for l in bz.lines]
                dn_lines = [l.index for l in dn.lines]
                match = "✅" if (bz_lines == dn_lines and 
                                abs(bz.gg - dn.gg) < 1e-9 and 
                                abs(bz.dd - dn.dd) < 1e-9 and
                                abs(bz.zg - dn.zg) < 1e-9 and
                                abs(bz.zd - dn.zd) < 1e-9) else "❌"
            
            print(f"\n  [{i}] BZ: {bz_str}")
            print(f"  [{i}] DN: {dn_str}  {match}")

if __name__ == '__main__':
    main()
