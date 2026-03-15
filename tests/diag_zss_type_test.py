"""
测试 cl_open 使用 zs_type_dn 时是否与 cl_pyarmor 匹配
验证假设：cl_pyarmor 默认使用 DN (段内中枢) 而非 BZ (标准中枢)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

CL_CONFIG_DN = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
    "zs_bi_type": ["zs_type_dn"],
    "zs_xd_type": ["zs_type_dn"],
}

CL_CONFIG_BZ = {
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

CL_CONFIG_NONE = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
    # no zs_bi_type/zs_xd_type -> use each engine's default
}

DATA_PATH = os.path.join(os.path.dirname(__file__), 'test_data', 'ETH_USDT_60m_1000.parquet')


def compare_zss(label, zss_o, zss_p):
    n_o, n_p = len(zss_o), len(zss_p)
    match = 0
    for i in range(min(n_o, n_p)):
        o, p = zss_o[i], zss_p[i]
        if (abs(o.zg - p.zg) < 1e-9 and abs(o.zd - p.zd) < 1e-9
                and o.type == p.type and o.line_num == p.line_num
                and o.done == p.done):
            match += 1
    total = max(n_o, n_p)
    ok = match == total and n_o == n_p
    print(f"  {label}: open={n_o} pyarmor={n_p} match={match}/{total} {'✅' if ok else '❌'}")
    
    if not ok:
        for i in range(min(n_o, n_p)):
            o, p = zss_o[i], zss_p[i]
            issues = []
            if abs(o.zg - p.zg) >= 1e-9: issues.append(f"zg")
            if abs(o.zd - p.zd) >= 1e-9: issues.append(f"zd")
            if o.type != p.type: issues.append(f"type:{o.type}≠{p.type}")
            if o.line_num != p.line_num: issues.append(f"L:{o.line_num}≠{p.line_num}")
            if o.done != p.done: issues.append(f"done:{o.done}≠{p.done}")
            mark = "✅" if not issues else "❌"
            if issues:
                print(f"    [{i}] {mark} {', '.join(issues)}")
    return ok


def main():
    df = pd.read_parquet(DATA_PATH)
    print(f"数据: ETH60 {len(df)} K线\n")

    # cl_pyarmor with no ZS type config (default)
    cd_p_default = CL_P("test", "test", config=CL_CONFIG_NONE)
    cd_p_default.process_klines(df)
    
    # cl_pyarmor with explicit BZ
    cd_p_bz = CL_P("test", "test", config=CL_CONFIG_BZ)
    cd_p_bz.process_klines(df)
    
    # cl_pyarmor with explicit DN
    cd_p_dn = CL_P("test", "test", config=CL_CONFIG_DN)
    cd_p_dn.process_klines(df)
    
    # cl_open with DN
    cd_o_dn = CL_O("test", "test", config=CL_CONFIG_DN)
    cd_o_dn.process_klines(df)
    
    # cl_open with BZ  
    cd_o_bz = CL_O("test", "test", config=CL_CONFIG_BZ)
    cd_o_bz.process_klines(df)

    print("=== 测试1: pyarmor(default) vs pyarmor(BZ) ===")
    print("  如果一致，说明 pyarmor 默认就是 BZ")
    compare_zss("bi_zss", cd_p_default.get_bi_zss(), cd_p_bz.get_bi_zss())

    print("\n=== 测试2: pyarmor(default) vs pyarmor(DN) ===")
    print("  如果一致，说明 pyarmor 默认就是 DN")
    compare_zss("bi_zss", cd_p_default.get_bi_zss(), cd_p_dn.get_bi_zss())

    print("\n=== 测试3: cl_open(DN) vs pyarmor(DN) ===")
    print("  如果一致，说明 DN 算法一致")
    compare_zss("bi_zss", cd_o_dn.get_bi_zss(), cd_p_dn.get_bi_zss())

    print("\n=== 测试4: cl_open(BZ) vs pyarmor(BZ) ===")
    print("  如果一致，说明 BZ 算法一致")
    compare_zss("bi_zss", cd_o_bz.get_bi_zss(), cd_p_bz.get_bi_zss())
    
    print("\n=== 测试5: cl_open(DN) vs pyarmor(default) ===")
    print("  如果一致，说明 pyarmor 默认是 DN 且 DN 算法一致")
    compare_zss("bi_zss", cd_o_dn.get_bi_zss(), cd_p_default.get_bi_zss())

    # Also show what each looks like
    print("\n=== pyarmor(default) bi_zss detail ===")
    for i, zs in enumerate(cd_p_default.get_bi_zss()):
        lines = [l.index for l in zs.lines]
        print(f"  [{i}] zg={zs.zg:.2f} zd={zs.zd:.2f} type={zs.type} L={zs.line_num} done={zs.done} gg={zs.gg:.2f} dd={zs.dd:.2f} lines={lines}")

    print("\n=== pyarmor(BZ) bi_zss detail ===")
    for i, zs in enumerate(cd_p_bz.get_bi_zss()):
        lines = [l.index for l in zs.lines]
        print(f"  [{i}] zg={zs.zg:.2f} zd={zs.zd:.2f} type={zs.type} L={zs.line_num} done={zs.done} gg={zs.gg:.2f} dd={zs.dd:.2f} lines={lines}")

    print("\n=== pyarmor(DN) bi_zss detail ===")
    for i, zs in enumerate(cd_p_dn.get_bi_zss()):
        lines = [l.index for l in zs.lines]
        print(f"  [{i}] zg={zs.zg:.2f} zd={zs.zd:.2f} type={zs.type} L={zs.line_num} done={zs.done} gg={zs.gg:.2f} dd={zs.dd:.2f} lines={lines}")

    print("\n=== cl_open(DN) bi_zss detail ===")
    for i, zs in enumerate(cd_o_dn.get_bi_zss()):
        lines = [l.index for l in zs.lines]
        print(f"  [{i}] zg={zs.zg:.2f} zd={zs.zd:.2f} type={zs.type} L={zs.line_num} done={zs.done} gg={zs.gg:.2f} dd={zs.dd:.2f} lines={lines}")


if __name__ == '__main__':
    main()
