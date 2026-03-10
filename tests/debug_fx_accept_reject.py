"""
Dump ALL attributes of pyarmor XLFX and TZXL for BTCd vs ETH60 comparison.
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
}

def dump_attrs(obj, prefix=""):
    for attr in sorted(dir(obj)):
        if attr.startswith('__'):
            continue
        try:
            val = getattr(obj, attr)
            if callable(val):
                continue
            if hasattr(val, '__len__') and not isinstance(val, str):
                if len(val) > 0 and hasattr(val[0], 'index'):
                    val_str = f"[{', '.join(f'bi[{v.index}]' for v in val)}]"
                elif len(val) > 0 and hasattr(val[0], 'lines'):
                    val_str = f"[{len(val)} TZXLs]"
                else:
                    val_str = repr(val) if len(val) < 5 else f"[len={len(val)}]"
            elif hasattr(val, 'index') and hasattr(val, 'type'):
                val_str = f"bi[{val.index}] {val.type}"
            elif hasattr(val, 'lines'):
                bi_indices = [l.index for l in val.lines]
                val_str = f"TZXL(bi{bi_indices}, bad={val.line_bad}, max={val.max:.2f}, min={val.min:.2f})"
            else:
                val_str = str(val)
            print(f"  {prefix}{attr} = {val_str}")
        except Exception as e:
            print(f"  {prefix}{attr} = ERROR: {e}")

def analyze(name, data_path, start_bi_idx, xd_type):
    df = pd.read_parquet(data_path)
    cd = CL_P("test", "test", config=CL_CONFIG)
    cd.process_klines(df)
    bis = cd.get_bis()
    
    target_fx_type = "ding" if xd_type == "up" else "di"
    lines = [bis[k] for k in range(start_bi_idx, len(bis))]
    tzxls, xlfxs = cd._xd_cal_line_xlfx(lines, target_fx_type, 'no_bh')
    
    print(f"\n{'='*80}")
    print(f"  {name}: {xd_type} from bi[{start_bi_idx}]")
    print(f"{'='*80}")
    
    for i in range(min(4, len(tzxls))):
        xl = tzxls[i]
        bi_indices = [l.index for l in xl.lines]
        print(f"\n  TZXL[{i}] bi{bi_indices}:")
        dump_attrs(xl, "    ")
    
    for f_idx in range(len(xlfxs)):
        fx = xlfxs[f_idx]
        bi_indices = [l.index for l in fx.xl.lines]
        print(f"\n  XLFX[{f_idx}] @ bi{bi_indices}:")
        dump_attrs(fx, "    ")

analyze("ETH60 down[28→30] MATCH", "tests/test_data/ETH_USDT_60m_1000.parquet", 28, "down")
analyze("BTCd down[22→?] DIVERGE", "tests/test_data/BTC_USDT_d_500.parquet", 22, "down")
