"""
查看 BTC60_500 的 XD 差异，检查 ZSD 是否有 mutation 问题
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pandas as pd
from chanlun.cl_pyarmor import CL as CL_P
from chanlun.cl_open import CL as CL_O

CL_CONFIG = {
    'bi_type': 'bi_type_old', 'fx_qj': 'fx_qj_k', 'fx_qy': 'fx_qy_three',
    'bi_fx_cgd': 'bi_fx_cgd_yes', 'fx_check_k_nums': 13,
    'bi_split_k_cross_nums': '20,1', 'xd_bzh': 'xd_bzh_no',
}

for name, path in [
    ("BTC60_500", "tests/test_data/BTC_USDT_60m_500.parquet"),
]:
    df = pd.read_parquet(path)
    cd_p = CL_P('test','test',config=CL_CONFIG)
    cd_p.process_klines(df)
    cd_o = CL_O('test','test',config=CL_CONFIG)
    cd_o.process_klines(df)

    xds_p = cd_p.get_xds()
    xds_o = cd_o.get_xds()
    zsds_o = cd_o.get_zsds()

    print(f"{name}: xds_p={len(xds_p)} xds_o={len(xds_o)}")
    print("XDs pyarmor:")
    for xd in xds_p:
        print(f"  xd[{xd.index}]: {xd.type} bi[{xd.start_line.index}→{xd.end_line.index}] done={xd.done} high={xd.high:.2f} low={xd.low:.2f}")
    print("XDs open:")
    for xd in xds_o:
        print(f"  xd[{xd.index}]: {xd.type} bi[{xd.start_line.index}→{xd.end_line.index}] done={xd.done} high={xd.high:.2f} low={xd.low:.2f}")
    print("ZSDs open (after zsd computation, check if xds mutated):")
    for zsd in zsds_o:
        sl = zsd.start_line
        el = zsd.end_line
        print(f"  zsd: {zsd.type} start_line.index={sl.index} end_line.index={el.index} done={zsd.done}")
        print(f"       start_line type={sl.type} bi[{sl.start_line.index}→{sl.end_line.index}]")
    print("XDs open after ZSD computation:")
    for xd in xds_o:
        print(f"  xd[{xd.index}]: {xd.type} bi[{xd.start_line.index}→{xd.end_line.index}] done={xd.done}")
