"""Inspect pyarmor CL class methods"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chanlun.cl import CL as CL_P

# List all methods that look relevant to XD building
methods = [m for m in dir(CL_P) if not m.startswith('__')]
print("Pyarmor CL methods:")
for m in sorted(methods):
    print(f"  {m}")
