"""Check all FXes between di@95 and di@101 — what does pyarmor see in the scan?
The scan after end_fx=di@95 would be:
- idx=67: ?
- idx=68: ?
- idx=69: ding@99 → confirmation attempt
- idx=70: ?
- idx=71: di@101

If ding@99 confirmation FAILS in pyarmor but PASSES in k_gap variant,
that would explain the divergence.
"""
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

df = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cd_o = CL_O("test", "test", config=CL_CONFIG)
cd_o.process_klines(df)
cd_p = CL_P("test", "test", config=CL_CONFIG)
cd_p.process_klines(df)

fxs = cd_o.get_fxs()
qj = cd_o.fx_qj; qy = cd_o.fx_qy

# Show all FXes by index from 64 to 75
print("=== FXes from idx=64 to idx=78 ===")
for i in range(64, min(79, len(fxs))):
    fx = fxs[i]
    print(f"  idx={i}: {fx.type:4s} k={fx.k.k_index} val={fx.val:.2f} "
          f"h={fx.high(qj,qy):.2f} l={fx.low(qj,qy):.2f}")

# Check: di@95 → ding@99 confirmation
# In standard scan: end_fx = di@95, next is ding (opposite type) → confirmation check
di95 = None
ding99 = None
di101 = None
for fx in fxs:
    if fx.k.k_index == 95 and fx.type == "di": di95 = fx
    if fx.k.k_index == 99 and fx.type == "ding": ding99 = fx
    if fx.k.k_index == 101 and fx.type == "di": di101 = fx

# In the k_gap trace, after extending to di@95 at idx=66:
# Next FX at idx=67 might be a ding (between 95 and 99)
print(f"\n=== FX at idx=67 ===")
fx67 = fxs[67] if 67 < len(fxs) else None
if fx67:
    print(f"  {fx67.type} k={fx67.k.k_index} val={fx67.val:.2f}")

# Actually wait — the trace showed ding@99 at "confirm" step after extend to 95.
# But between idx=66 (di@95) and idx=69 (ding@99), there might be more FXes.
# Let's check idx 67 and 68:
print(f"\n=== Trace: after end_fx = di@95 (idx=66) ===")
for idx in range(67, 75):
    fx = fxs[idx] if idx < len(fxs) else None
    if fx:
        if fx.type == "di":
            # same type as end_fx → extension check
            k_gap_from_start = fx.k.k_index - 69  # from ding@69
            val_compare = f"val={fx.val:.2f} vs end_fx val={di95.val:.2f}"
            extend = fx.val <= di95.val and k_gap_from_start >= 4
            print(f"  idx={idx}: di@{fx.k.k_index} → EXTEND? val={fx.val:.2f}, "
                  f"end_val={di95.val:.2f}, lower={fx.val<=di95.val}")
        else:
            # confirmation check  
            k_gap = fx.k.k_index - di95.k.k_index
            cl_gap = fx.k.index - di95.k.index
            # For BI confirmation from di@95:
            # di (start of new BI) → ding: UP BI
            # C1: start.high > end.high
            # C2: end.low < start.low  
            h_s = di95.high(qj, qy); l_s = di95.low(qj, qy)
            h_e = fx.high(qj, qy); l_e = fx.low(qj, qy)
            c1 = h_s > h_e
            c2 = l_e < l_s
            print(f"  idx={idx}: ding@{fx.k.k_index} → CONFIRM? k_gap={k_gap} cl_gap={cl_gap} "
                  f"C1={c1} C2={c2} strict={'FAIL' if c1 or c2 else 'PASS'}")

# KEY QUESTION: is there a di between 95 and 99? Check indices 67, 68:
print(f"\n=== What's between di@95 and ding@99? ===")
for fx in fxs:
    if 95 < fx.k.k_index < 99:
        print(f"  {fx.type} k={fx.k.k_index} val={fx.val:.2f}")
# Or by idx:
for i in range(65, 72):
    fx = fxs[i]
    print(f"  fxs[{i}]: {fx.type:4s} k={fx.k.k_index} val={fx.val:.2f}")

# Now the REAL question: baseline (cl_gap variant) also gets here.
# In baseline, ding@69 is start. Di@78 is candidate. 
# But di@78→ding@84: cl_gap = 57-54 = 3 < 4 → GAP FAIL
# di@78→ding@89: cl_gap = 61-54 = 7, strict C1 and C2?

print(f"\n=== Baseline trace: ding@69 ===")
start_fx = None
for fx in fxs:
    if fx.k.k_index == 69 and fx.type == "ding": start_fx = fx; break

di78 = None
for fx in fxs:
    if fx.k.k_index == 78 and fx.type == "di": di78 = fx; break

# Baseline: di@78 is first candidate (cl_gap=5 >= 4)
# Then scan for ding to confirm or di to extend:
print(f"  candidate: di@78 cl_gap={di78.k.index - start_fx.k.index}")

# ding@84: cl_gap from di@78 = 57-54 = 3 < 4 → FAIL
ding84 = None
for fx in fxs:
    if fx.k.k_index == 84 and fx.type == "ding": ding84 = fx; break
print(f"  confirm ding@84: cl_gap={ding84.k.index - di78.k.index} → {'FAIL' if ding84.k.index - di78.k.index < 4 else 'PASS'}")

# di@87: same type, can extend? val=87711 > 87650 → NO (not lower)
di87 = None
for fx in fxs:
    if fx.k.k_index == 87 and fx.type == "di": di87 = fx; break
print(f"  extend di@87: val={di87.val:.2f} vs {di78.val:.2f} → {'YES' if di87.val <= di78.val else 'NO (not lower)'}")

# ding@89: cl_gap from di@78 = 61-54 = 7
ding89 = None
for fx in fxs:
    if fx.k.k_index == 89 and fx.type == "ding": ding89 = fx; break
cl_89 = ding89.k.index - di78.k.index
k_89 = ding89.k.k_index - di78.k.k_index
h_s = di78.high(qj, qy); l_s = di78.low(qj, qy)
h_e = ding89.high(qj, qy); l_e = ding89.low(qj, qy)
c1 = h_s > h_e; c2 = l_e < l_s
print(f"  confirm ding@89: cl_gap={cl_89} k_gap={k_89} "
      f"C1={c1} C2={c2} strict={'FAIL' if c1 or c2 else 'PASS'}")

# If ding@89 confirmation fails, scan continues:
# di@95: val=83338 < 87650 → EXTEND
print(f"  extend di@95: val={di95.val:.2f} vs {di78.val:.2f} → {'YES' if di95.val <= di78.val else 'NO'}")

# ding@99: cl_gap from di@95 = 69-66 = 3 < 4 → FAIL in baseline!
print(f"  confirm ding@99: cl_gap={ding99.k.index - di95.k.index} → {'FAIL' if ding99.k.index - di95.k.index < 4 else 'PASS'}")

# di@101: val=81000 < 83338 → EXTEND
print(f"  extend di@101: val={di101.val:.2f} vs {di95.val:.2f} → {'YES' if di101.val <= di95.val else 'NO'}")

# Now confirmation from di@101:
# ding@105: cl_gap = 74-71 = 3 < 4 → FAIL
ding105 = None
for fx in fxs:
    if fx.k.k_index == 105 and fx.type == "ding": ding105 = fx; break
print(f"  confirm ding@105: cl_gap={ding105.k.index - di101.k.index} → {'FAIL' if ding105.k.index - di101.k.index < 4 else 'PASS'}")

# ding@107
ding107 = None
for fx in fxs:
    if fx.k.k_index == 107 and fx.type == "ding": ding107 = fx; break
if ding107:
    print(f"  confirm ding@107: cl_gap={ding107.k.index - di101.k.index} → {'FAIL' if ding107.k.index - di101.k.index < 4 else 'PASS'}")

# Eventually what does baseline create?
bis_o = cd_o.get_bis()
print(f"\n  Baseline BI: {bis_o[2].type} {bis_o[2].start.k.k_index}→{bis_o[2].end.k.k_index}")
print(f"  Confirmation would be at: {bis_o[3].start.k.k_index}→{bis_o[3].end.k.k_index}")
