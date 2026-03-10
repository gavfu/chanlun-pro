"""Compare FX sequences for BTC60 and BTC5m near boundary divergence points."""
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

for name, file, check_ranges in [
    ("BTC60", "BTC_USDT_60m_1000.parquet", [(290, 330), (900, 970)]),
    ("BTC5m", "BTC_USDT_5m_1000.parquet", [(860, 920)]),
]:
    df = pd.read_parquet(f"tests/test_data/{file}")
    cd_o = CL_O("test", "test", config=CL_CONFIG)
    cd_o.process_klines(df)
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)

    fxs_o = cd_o.get_fxs()
    fxs_p = cd_p.get_fxs()
    
    print(f"\n{'='*60}")
    print(f"  {name}: FX count open={len(fxs_o)} pyarmor={len(fxs_p)}")
    print(f"{'='*60}")
    
    # Find first FX difference
    min_len = min(len(fxs_o), len(fxs_p))
    first_diff = None
    for i in range(min_len):
        fo = fxs_o[i]
        fp = fxs_p[i]
        if fo.k.k_index != fp.k.k_index or fo.type != fp.type:
            first_diff = i
            break
    
    if first_diff is not None:
        print(f"  First FX difference at index {first_diff}:")
        # Show surrounding FXes
        for j in range(max(0, first_diff - 2), min(min_len, first_diff + 5)):
            fo = fxs_o[j]
            fp = fxs_p[j]
            match = "✅" if fo.k.k_index == fp.k.k_index and fo.type == fp.type else "❌"
            print(f"    {match} fx[{j}] open: {fo.type:>4} k={fo.k.k_index} val={fo.val:.2f}  "
                  f"pya: {fp.type:>4} k={fp.k.k_index} val={fp.val:.2f}")
    elif len(fxs_o) != len(fxs_p):
        print(f"  FX sequences match up to index {min_len-1}, but counts differ")
    else:
        print(f"  All FX sequences identical")
    
    # Check merged K-lines near divergence
    cklines_o = cd_o.get_cl_klines()
    cklines_p = cd_p.get_cl_klines()
    
    print(f"\n  Merged K-lines: open={len(cklines_o)} pyarmor={len(cklines_p)}")
    
    # Find first K-line difference
    min_ck = min(len(cklines_o), len(cklines_p))
    first_ck_diff = None
    for i in range(min_ck):
        co = cklines_o[i]
        cp = cklines_p[i]
        if abs(co.h - cp.h) > 0.001 or abs(co.l - cp.l) > 0.001:
            first_ck_diff = i
            break
    
    if first_ck_diff is not None:
        print(f"  First merged K-line h/l difference at ck[{first_ck_diff}]:")
        for j in range(max(0, first_ck_diff - 1), min(min_ck, first_ck_diff + 5)):
            co = cklines_o[j]
            cp = cklines_p[j]
            match = "✅" if abs(co.h - cp.h) < 0.001 and abs(co.l - cp.l) < 0.001 else "❌"
            raws_o = [k.index for k in co.klines]
            raws_p = [k.index for k in cp.klines]
            rawsame = "✅" if raws_o == raws_p else "❌"
            print(f"    {match} ck[{j}] oh={co.h:.2f} ph={cp.h:.2f} ol={co.l:.2f} pl={cp.l:.2f} raws{rawsame}")
    else:
        print(f"  All merged K-lines h/l identical (up to ck[{min_ck-1}])")
        # Check raw K-line count
        if len(cklines_o) != len(cklines_p):
            print(f"  BUT merged K-line count differs!")
