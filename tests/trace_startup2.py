"""Trace what prints the authorization message during web startup."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'web', 'chanlun_chart'))

# Monkey-patch print to catch the auth message
_orig_print = print
def traced_print(*args, **kwargs):
    import traceback
    msg = ' '.join(str(a) for a in args)
    if '授权' in msg or 'trial' in msg or '缠论数据计算' in msg or 'Chanlun-Pro' in msg or 'gitee' in msg or 'Python' in msg:
        _orig_print(f"=== CAUGHT: {msg} ===")
        traceback.print_stack()
    _orig_print(*args, **kwargs)

import builtins
builtins.print = traced_print

# Simulate app.py imports
try:
    from cl_app import create_app
    _orig_print("--- create_app imported successfully ---")
except Exception as e:
    _orig_print(f"--- create_app import failed: {e} ---")

# Check if cl_pyarmor got loaded
if 'chanlun.cl_pyarmor' in sys.modules:
    _orig_print("!!! cl_pyarmor IS in sys.modules !!!")
else:
    _orig_print("--- cl_pyarmor NOT in sys.modules ---")

if 'pyarmor_runtime_005445' in sys.modules:
    _orig_print("!!! pyarmor_runtime IS in sys.modules !!!")
else:
    _orig_print("--- pyarmor_runtime NOT in sys.modules ---")
