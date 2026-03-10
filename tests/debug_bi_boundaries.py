"""Show BI boundary differences for BTC60 and BTC5m."""
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
}

for name, file in [("BTC60", "BTC_USDT_60m_1000.parquet"), ("BTC5m", "BTC_USDT_5m_1000.parquet")]:
    df = pd.read_parquet(f"tests/test_data/{file}")
    cd_o = CL_O("test", "test", config=CL_CONFIG)
    cd_o.process_klines(df)
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)

    bis_o = cd_o.get_bis()
    bis_p = cd_p.get_bis()
    
    print(f"\n{'='*60}")
    print(f"  {name}: BI count open={len(bis_o)} pyarmor={len(bis_p)}")
    print(f"{'='*60}")
    
    if len(bis_o) != len(bis_p):
        print("  BI count mismatch, skipping boundary check")
        continue
    
    diffs = 0
    for i in range(len(bis_o)):
        bo = bis_o[i]
        bp = bis_p[i]
        s_diff = bo.start.k.k_index != bp.start.k.k_index
        e_diff = bo.end.k.k_index != bp.end.k.k_index
        if s_diff or e_diff:
            s_mark = f" START({bo.start.k.k_index}vs{bp.start.k.k_index})" if s_diff else ""
            e_mark = f" END({bo.end.k.k_index}vs{bp.end.k.k_index})" if e_diff else ""
            # Show FX details for the differing boundary
            print(f"  bi[{i:>2}] {bo.type:>4}{s_mark}{e_mark}")
            if s_diff:
                so = bo.start
                sp = bp.start
                print(f"    Start FX: open val={so.val:.2f} k_index={so.k.k_index} klines=[{','.join(str(k.k_index) for k in so.klines)}]")
                print(f"              pya  val={sp.val:.2f} k_index={sp.k.k_index} klines=[{','.join(str(k.k_index) for k in sp.klines)}]")
            if e_diff:
                eo = bo.end
                ep = bp.end
                print(f"    End FX:   open val={eo.val:.2f} k_index={eo.k.k_index} klines=[{','.join(str(k.k_index) for k in eo.klines)}]")
                print(f"              pya  val={ep.val:.2f} k_index={ep.k.k_index} klines=[{','.join(str(k.k_index) for k in ep.klines)}]")
            diffs += 1
    
    if diffs == 0:
        print("  All BI boundaries PERFECT")
    else:
        print(f"\n  Total boundary diffs: {diffs}")
