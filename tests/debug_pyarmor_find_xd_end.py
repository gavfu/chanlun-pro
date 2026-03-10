"""
Hook pyarmor's _xd_check_xd_is_ok and _xd_get_next_tzfx to observe behavior for BTCd.
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

df = pd.read_parquet("tests/test_data/BTC_USDT_d_500.parquet")
cd = CL_P("test", "test", config=CL_CONFIG)

# Hook _xd_check_xd_is_ok
original_check = cd._xd_check_xd_is_ok
def patched_check(*args, **kwargs):
    result = original_check(*args, **kwargs)
    # Print args info
    arg_strs = []
    for a in args:
        if hasattr(a, 'index'):
            arg_strs.append(f"bi[{a.index}]")
        elif hasattr(a, 'type') and hasattr(a, 'start_line'):
            arg_strs.append(f"XD({a.type} bi[{a.start_line.index}→{a.end_line.index}])")
        elif hasattr(a, 'type') and hasattr(a, 'xl'):
            bi_indices = [l.index for l in a.xl.lines]
            arg_strs.append(f"XLFX({a.type}@bi{bi_indices})")
        elif isinstance(a, list) and len(a) > 0:
            if hasattr(a[0], 'index'):
                arg_strs.append(f"bis[{a[0].index}..{a[-1].index}]")
            elif hasattr(a[0], 'lines'):
                arg_strs.append(f"tzxls[{len(a)}]")
            else:
                arg_strs.append(f"list[{len(a)}]")
        elif isinstance(a, str):
            arg_strs.append(f"'{a}'")
        elif isinstance(a, (int, float, bool)):
            arg_strs.append(str(a))
        else:
            arg_strs.append(type(a).__name__)
    print(f"  _xd_check_xd_is_ok({', '.join(arg_strs)}) → {result}")
    return result
cd._xd_check_xd_is_ok = patched_check

# Hook _xd_get_next_tzfx
original_get_next = cd._xd_get_next_tzfx
def patched_get_next(*args, **kwargs):
    result = original_get_next(*args, **kwargs)
    if result is not None:
        if hasattr(result, 'type') and hasattr(result, 'xl'):
            bi_indices = [l.index for l in result.xl.lines]
            bad = result.xl.line_bad if hasattr(result.xl, 'line_bad') else '?'
            print(f"  _xd_get_next_tzfx → XLFX({result.type}@bi{bi_indices}, bad={bad})")
        else:
            print(f"  _xd_get_next_tzfx → {type(result).__name__}")
    else:
        print(f"  _xd_get_next_tzfx → None")
    return result
cd._xd_get_next_tzfx = patched_get_next

# Hook _xd_done_pre_xd
original_done = cd._xd_done_pre_xd
def patched_done(*args, **kwargs):
    result = original_done(*args, **kwargs)
    print(f"  _xd_done_pre_xd called, result type={type(result).__name__ if result is not None else 'None'}")
    return result
cd._xd_done_pre_xd = patched_done

cd.process_klines(df)
