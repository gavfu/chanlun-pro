"""Trace what prints the authorization message during web startup."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'web', 'chanlun_chart'))

# Monkey-patch print to catch the auth message
_orig_print = print
def traced_print(*args, **kwargs):
    import traceback
    msg = ' '.join(str(a) for a in args)
    if '授权' in msg or 'trial' in msg or '缠论数据计算' in msg or 'Chanlun-Pro' in msg or 'gitee' in msg:
        _orig_print(f"=== CAUGHT: {msg} ===")
        traceback.print_stack()
    _orig_print(*args, **kwargs)

import builtins
builtins.print = traced_print

# Simulate app.py imports
from cl_app import create_app
