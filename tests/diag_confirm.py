# -*- coding: utf-8 -*-
"""
Check strict check for confirm pairs at divergence points.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pathlib, pandas as pd
from chanlun.cl_open import CL as CLOpen
from chanlun.cl_interface import Config

df = pd.read_parquet(pathlib.Path(__file__).parent / "test_data" / "BTC_USDT_60m_500.parquet")
c = CLOpen("BTC/USDT", "60m", {})
c.process_klines(df)
fxs = c.fxs

qj_ck = Config.FX_QJ_CK.value
qy_mid = Config.FX_QY_MIDDLE.value
qj_k = Config.FX_QJ_K.value
qy_three = Config.FX_QY_THREE.value

def check_pair(a, b, label=""):
    fx_a, fx_b = fxs[a], fxs[b]
    cl_gap = fx_b.k.index - fx_a.k.index
    k_gap = fx_b.k.k_index - fx_a.k.k_index
    print(f"\n  FX{a}({fx_a.type})->FX{b}({fx_b.type}) {label}")
    print(f"    cl_gap={cl_gap}, k_gap={k_gap}")
    for qj, qy in [(qj_ck, qy_mid), (qj_ck, qy_three), (qj_k, qy_mid), (qj_k, qy_three)]:
        sh = fx_a.high(qj, qy)
        sl = fx_a.low(qj, qy)
        eh = fx_b.high(qj, qy)
        el = fx_b.low(qj, qy)
        if fx_a.type == "ding" and fx_b.type == "di":
            fail = sl < el or eh > sh
            detail = f"sl({sl})<el({el})={sl<el} | eh({eh})>sh({sh})={eh>sh}"
        elif fx_a.type == "di" and fx_b.type == "ding":
            fail = sh > eh or el < sl
            detail = f"sh({sh})>eh({eh})={sh>eh} | el({el})<sl({sl})={el<sl}"
        else:
            fail = False; detail = "same type"
        print(f"    {qj}+{qy}: FAIL={fail} ({detail})")

# The confirm pair in divergence #1: FX22(ding)->FX25(di) confirms bi FX19->FX22
print("=== Divergence #1: confirm pair ===")
check_pair(22, 25, "(FX22->FX25 confirms bi FX19->FX22 in open)")

# The confirm pair in divergence #2: FX57(di)->FX64(ding) confirms bi FX54->FX57 in open
print("\n=== Divergence #2: confirm pair ===")
check_pair(57, 64, "(FX57->FX64 confirms bi FX54->FX57 in open)")

# The confirm pair in divergence #3: FX130(ding)->FX135(di) confirms bi FX129->FX130 in open
print("\n=== Divergence #3: confirm pair ===")
check_pair(130, 135, "(FX130->FX135 confirms bi FX129->FX130 in open)")

print("\n\n=== Summary ===")
print("If ANY of the above confirm pairs should FAIL the strict check,")
print("that would prevent premature confirmation.")


df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "test_data", "BTC_USDT_60m_500.parquet"))
cd = CL("BTC/USDT", "60m")
cd.process_klines(df)

fxs = cd.get_fxs()
fx_map = {fx.k.index: fx for fx in fxs}

# Case 1: di 96 -> ding 101 (end_fx), then di 106 (confirm)
# The CONFIRMATION is: _bi_fx_valid(ding 101, di 106)
# This is a DOWN stroke check: ding->di
# Strict check for down: start.low < end.low → block
s = fx_map[101]  # ding
e = fx_map[106]  # di
k_gap = e.k.k_index - s.k.k_index
print(f"CONFIRMATION: ding 101 -> di 106 (k_gap={k_gap})")
for qj in ['fx_qj_k', 'fx_qj_ck']:
    for qy in ['fx_qy_middle', 'fx_qy_three']:
        sh = s.high(qj, qy)
        sl = s.low(qj, qy)
        eh = e.high(qj, qy)
        el = e.low(qj, qy)
        # For down: start_fx.low < end_fx.low → block
        check_l = sl < el
        # Also: start.high < end.high
        check_h = sh < eh
        print(f"  {qj:10s} {qy:14s}: sl={sl:8.1f} el={el:8.1f} sl<el={check_l} | sh={sh:8.1f} eh={eh:8.1f} sh<eh={check_h}")

print()

# Case 2: di 40 -> ding 44 (end_fx), then what's the confirmation fractal?
# In strict OFF: ding 44 is confirmed → next is di 49
# So confirmation is: _bi_fx_valid(ding 44, di 49)
if 49 in fx_map:
    s = fx_map[44]  # ding
    e = fx_map[49]  # di
    k_gap = e.k.k_index - s.k.k_index
    print(f"CONFIRMATION: ding 44 -> di 49 (k_gap={k_gap})")
    for qj in ['fx_qj_k', 'fx_qj_ck']:
        for qy in ['fx_qy_middle', 'fx_qy_three']:
            sl = s.low(qj, qy)
            el = e.low(qj, qy)
            sh = s.high(qj, qy)
            eh = e.high(qj, qy)
            check_l = sl < el
            check_h = sh < eh
            print(f"  {qj:10s} {qy:14s}: sl={sl:8.1f} el={el:8.1f} sl<el={check_l} | sh={sh:8.1f} eh={eh:8.1f} sh<eh={check_h}")

print()

# Case 3: di 262 -> ding 267 (end_fx), then di 270 (confirm)
# Confirmation is: _bi_fx_valid(ding 267, di 270)
if 270 in fx_map:
    s = fx_map[267]  # ding
    e = fx_map[270]  # di
    k_gap = e.k.k_index - s.k.k_index
    print(f"CONFIRMATION: ding 267 -> di 270 (k_gap={k_gap})")
    print(f"  cl_gap={e.k.index - s.k.index}")
    for qj in ['fx_qj_k', 'fx_qj_ck']:
        for qy in ['fx_qy_middle', 'fx_qy_three']:
            sl = s.low(qj, qy)
            el = e.low(qj, qy)
            sh = s.high(qj, qy)
            eh = e.high(qj, qy)
            check_l = sl < el
            check_h = sh < eh
            print(f"  {qj:10s} {qy:14s}: sl={sl:8.1f} el={el:8.1f} sl<el={check_l} | sh={sh:8.1f} eh={eh:8.1f} sh<eh={check_h}")

    # Try di 274 as confirmation instead
    print()
    s = fx_map[267]
    e = fx_map[274]
    k_gap = e.k.k_index - s.k.k_index
    print(f"CONFIRMATION: ding 267 -> di 274 (k_gap={k_gap})")
    print(f"  cl_gap={e.k.index - s.k.index}")
    for qj in ['fx_qj_k', 'fx_qj_ck']:
        for qy in ['fx_qy_middle', 'fx_qy_three']:
            sl = s.low(qj, qy)
            el = e.low(qj, qy)
            sh = s.high(qj, qy)
            eh = e.high(qj, qy)
            check_l = sl < el
            check_h = sh < eh
            print(f"  {qj:10s} {qy:14s}: sl={sl:8.1f} el={el:8.1f} sl<el={check_l} | sh={sh:8.1f} eh={eh:8.1f} sh<eh={check_h}")

# Now let's also check: what fractals exist near 96-113 to trace the full flow
print("\n=== Fractals 96-113 ===")
for fx in fxs:
    if 96 <= fx.k.index <= 113:
        print(f"  ck={fx.k.index:3d} {fx.type:4s} val={fx.val:8.1f} k_idx={fx.k.k_index}")

# What happens after ding 101 is set as end_fx?
# We scan: di 106 (opposite), check _bi_fx_valid(101, 106)
# If that fails (confirmation blocked), we continue...
# ding 108? — check if same type as end_fx and higher
# ding 113? — check if same type as end_fx and higher
print("\n=== Full trace from di 96 ===")
start = fx_map[96]
print(f"start: ck={start.k.index} {start.type} val={start.val:.1f}")
end_fx = None
for fx in fxs:
    if fx.k.index <= 96:
        continue
    if fx.k.index > 120:
        break
    
    cl_gap = fx.k.index - start.k.index
    k_gap_s = fx.k.k_index - start.k.k_index
    
    if end_fx is None:
        if fx.type != start.type:
            # Opposite: check valid
            cl_ok = cl_gap >= 4
            k_ok = k_gap_s >= 4
            print(f"  candidate end: ck={fx.k.index} {fx.type} val={fx.val:.1f} cl_gap={cl_gap} k_gap={k_gap_s} valid={cl_ok and k_ok}")
            if cl_ok and k_ok:
                end_fx = fx
                print(f"  -> SET end_fx = ck={fx.k.index}")
        else:
            if fx.val < start.val:
                print(f"  lower di: ck={fx.k.index} val={fx.val:.1f} -> update start")
    else:
        if fx.type == end_fx.type:
            # Same type as end: extend?
            better = (fx.type == "ding" and fx.val >= end_fx.val) or \
                     (fx.type == "di" and fx.val <= end_fx.val)
            if better:
                cl_gap_s = fx.k.index - start.k.index
                k_gap_sf = fx.k.k_index - start.k.k_index
                valid = cl_gap_s >= 4 and k_gap_sf >= 4
                print(f"  extend: ck={fx.k.index} {fx.type} val={fx.val:.1f} better={better} valid={valid}")
                if valid:
                    end_fx = fx
                    print(f"  -> EXTEND end_fx = ck={fx.k.index}")
        else:
            # Opposite: confirmation?
            k_gap_conf = fx.k.k_index - end_fx.k.k_index
            cl_gap_conf = fx.k.index - end_fx.k.index
            conf_valid = cl_gap_conf >= 4 and k_gap_conf >= 4
            print(f"  confirm attempt: ck={fx.k.index} {fx.type} val={fx.val:.1f} "
                  f"cl_gap={cl_gap_conf} k_gap={k_gap_conf} gap_valid={conf_valid}")
            if conf_valid and k_gap_conf < 13:
                # Strict check on confirmation
                qj, qy = 'fx_qj_k', 'fx_qy_three'
                if end_fx.type == "ding" and fx.type == "di":
                    sl = end_fx.low(qj, qy)
                    el = fx.low(qj, qy)
                    strict_block = sl < el
                    print(f"    strict(down conf): sl={sl:.1f} < el={el:.1f} = {strict_block}")
                elif end_fx.type == "di" and fx.type == "ding":
                    sh = end_fx.high(qj, qy)
                    eh = fx.high(qj, qy)
                    strict_block = sh > eh
                    print(f"    strict(up conf): sh={sh:.1f} > eh={eh:.1f} = {strict_block}")
