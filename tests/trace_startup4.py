"""Trace the authorization message source - use port 9999 to avoid conflict"""
import pathlib, sys, traceback, signal

src_path = pathlib.Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
web_path = pathlib.Path(__file__).parent.parent / "web" / "chanlun_chart"
sys.path.insert(0, str(web_path))

class TracingWriter:
    def __init__(self, orig):
        self._orig = orig
    def write(self, s):
        if any(k in s for k in ['授权', 'trial', 'Chanlun', 'gitee', '缠论数据']):
            self._orig.write(f"\n=== CAUGHT via stdout.write: {s!r} ===\n")
            traceback.print_stack(file=self._orig)
        return self._orig.write(s)
    def flush(self):
        return self._orig.flush()
    def __getattr__(self, name):
        return getattr(self._orig, name)

sys.stdout = TracingWriter(sys.stdout)
sys.stderr = TracingWriter(sys.stderr)

import builtins
_orig_print = builtins.print
def traced_print(*args, **kwargs):
    msg = ' '.join(str(a) for a in args)
    if any(k in msg for k in ['授权', 'trial', 'Chanlun', 'gitee', '缠论数据']):
        _orig_print(f"\n=== CAUGHT via print(): {msg!r} ===")
        traceback.print_stack()
    return _orig_print(*args, **kwargs)
builtins.print = traced_print

import chanlun.encodefix
from chanlun import config
from cl_app import create_app

app = create_app()
_orig_print("--- create_app done ---")

from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.wsgi import WSGIContainer
from concurrent.futures import ThreadPoolExecutor
import urllib.request

def stop(*a):
    _orig_print("\n--- stopping ---")
    IOLoop.current().stop()

signal.signal(signal.SIGALRM, stop)

s = HTTPServer(WSGIContainer(app, executor=ThreadPoolExecutor(10)))
s.bind(9999)
_orig_print("启动成功 (port 9999)")
s.start(1)

def make_request():
    try:
        urllib.request.urlopen("http://127.0.0.1:9999/")
    except Exception:
        pass
    # schedule stop after request
    IOLoop.current().call_later(1, stop)

signal.alarm(10)
IOLoop.current().call_later(0.5, make_request)
IOLoop.current().start()

_orig_print("\n=== Pyarmor modules loaded ===")
for name in sorted(sys.modules.keys()):
    if 'pyarmor' in name or 'cl_pyarmor' in name:
        _orig_print(f"  {name}: {sys.modules[name]}")
_orig_print("=== Done ===")
