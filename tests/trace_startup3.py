"""Trace the authorization message source by intercepting stdout.write"""
import pathlib, sys, os, io, traceback

# Setup paths same as app.py
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

# Also patch builtins.print
import builtins
_orig_print = builtins.print
def traced_print(*args, **kwargs):
    msg = ' '.join(str(a) for a in args)
    if any(k in msg for k in ['授权', 'trial', 'Chanlun', 'gitee', '缠论数据']):
        _orig_print(f"\n=== CAUGHT via print(): {msg!r} ===")
        traceback.print_stack()
    return _orig_print(*args, **kwargs)
builtins.print = traced_print

# Now import and create app like app.py does
import chanlun.encodefix
from chanlun import config
from cl_app import create_app

app = create_app()
_orig_print("--- create_app done, starting server ---")

# Start server briefly then exit
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.wsgi import WSGIContainer
from concurrent.futures import ThreadPoolExecutor
import signal

def stop_loop(signum, frame):
    _orig_print("\n--- ALARM, stopping ---")
    IOLoop.current().stop()

signal.signal(signal.SIGALRM, stop_loop)

s = HTTPServer(WSGIContainer(app, executor=ThreadPoolExecutor(10)))
s.bind(9900, config.WEB_HOST)
_orig_print("启动成功")
s.start(1)

# Open web page to trigger first request
import urllib.request
signal.alarm(5)
try:
    # Make a request to trigger route handlers
    IOLoop.current().call_later(0.5, lambda: urllib.request.urlopen("http://127.0.0.1:9900/"))
    IOLoop.current().start()
except Exception as e:
    _orig_print(f"Error: {e}")

_orig_print("\n=== Modules with pyarmor ===")
for name in sorted(sys.modules.keys()):
    if 'pyarmor' in name or 'cl_pyarmor' in name:
        _orig_print(f"  {name}: {sys.modules[name]}")
