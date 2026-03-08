"""Test XD match between cl_open and cl_pyarmor after algorithm rewrite."""
import sys; sys.path.insert(0, 'src')
import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')

cd_o = CL_O('BTC/USDT', '60m')
cd_o.process_klines(df)

cd_p = CL_P('BTC/USDT', '60m')
cd_p.process_klines(df)

print("=== XD Comparison ===")
print(f"Open XDs: {len(cd_o.xds)}, Pyarmor XDs: {len(cd_p.xds)}")

print("\n--- Open XDs ---")
for x in cd_o.xds:
    print(f"  [{x.index}] {x.type} bi[{x.start_line.index}]→bi[{x.end_line.index}] done={x.done}")

print("\n--- Pyarmor XDs ---")
for x in cd_p.xds:
    print(f"  [{x.index}] {x.type} bi[{x.start_line.index}]→bi[{x.end_line.index}] done={x.done}")

# Compare TZXL for matching XDs
for i in range(min(len(cd_o.xds), len(cd_p.xds))):
    xd_o = cd_o.xds[i]
    xd_p = cd_p.xds[i]
    print(f"\n=== XD[{i}] TZXL comparison ===")
    print(f"  Open:   {xd_o.type} bi[{xd_o.start_line.index}]→bi[{xd_o.end_line.index}]")
    print(f"  Pyarmor:{xd_p.type} bi[{xd_p.start_line.index}]→bi[{xd_p.end_line.index}]")
    
    max_t = max(len(xd_o.tzxls), len(xd_p.tzxls))
    for j in range(max_t):
        o_str = ""
        p_str = ""
        if j < len(xd_o.tzxls):
            t = xd_o.tzxls[j]
            bis_str = ','.join(str(l.index) for l in t.lines)
            o_str = f"bi[{bis_str}] max={t.max:.1f} min={t.min:.1f} bad={t.line_bad}"
        if j < len(xd_p.tzxls):
            t = xd_p.tzxls[j]
            bis_str = ','.join(str(l.index) for l in t.lines)
            p_str = f"bi[{bis_str}] max={t.max:.1f} min={t.min:.1f} bad={t.line_bad}"
        
        match = "✓" if o_str == p_str else "✗"
        print(f"  [{j}] {match} O: {o_str}")
        if o_str != p_str:
            print(f"       P: {p_str}")

# Summary
all_match = True
if len(cd_o.xds) != len(cd_p.xds):
    all_match = False
    print(f"\n✗ XD count mismatch: {len(cd_o.xds)} vs {len(cd_p.xds)}")
else:
    for i in range(len(cd_o.xds)):
        xd_o = cd_o.xds[i]
        xd_p = cd_p.xds[i]
        if (xd_o.type != xd_p.type or 
            xd_o.start_line.index != xd_p.start_line.index or
            xd_o.end_line.index != xd_p.end_line.index):
            all_match = False
            break

if all_match:
    print(f"\n✓ ALL {len(cd_o.xds)} XDs MATCH!")
else:
    print(f"\n✗ XD mismatch detected")
