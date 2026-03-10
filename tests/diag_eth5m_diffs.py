# -*- coding: utf-8 -*-
"""Trace _build_bis step by step for ETH5m around key diffs"""
import sys, os, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor
from chanlun.cl_interface import Config

DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
df = pd.read_parquet(os.path.join(DATA_DIR, 'ETH_USDT_5m_1000.parquet'))

# Get pyarmor FXs and BIs for reference
cd_py = CL_Pyarmor("ETH/USDT", "5m", config={})
cd_py.process_klines(df)

# Get cl_open to access its FXs
cd_open = CL_Open("ETH/USDT", "5m", config={})
cd_open.process_klines(df)

fxs = cd_open.fxs
qj = cd_open.fx_qj
qy = cd_open.fx_qy

# ----- Diff #24: d[240→253] vs d[240→244] -----
# cl_open extends endpoint from 244 to 253, pyarmor stops at 244
print("=" * 70)
print("Diff #24: cl_open d[240→253] vs pyarmor d[240→244]")
print("=" * 70)

# Find FXs around this area
print("\nFXs from index 235 to 265:")
for fx in fxs:
    if 235 <= fx.k.index <= 265:
        print(f"  FX({fx.k.index},{fx.type}) val={fx.val:.2f} k_idx={fx.k.k_index} h={fx.high(qj,qy):.2f} l={fx.low(qj,qy):.2f}")

# Find FX(244) and FX(253)
fx_244 = fx_253 = fx_240 = None
for fx in fxs:
    if fx.k.index == 240: fx_240 = fx
    if fx.k.index == 244: fx_244 = fx
    if fx.k.index == 253: fx_253 = fx

if fx_244 and fx_253:
    print(f"\nFX(244): type={fx_244.type} val={fx_244.val:.2f}")
    print(f"FX(253): type={fx_253.type} val={fx_253.val:.2f}")
    print(f"FX(253).val <= FX(244).val? {fx_253.val <= fx_244.val}")
    print(f"FX(253).val < FX(244).val? {fx_253.val < fx_244.val}")
    print(f"FX(253).val == FX(244).val? {fx_253.val == fx_244.val}")

# Check what FXs appear between 244 and 253
print(f"\nFXs between 244 and 264:")
for fx in fxs:
    if 244 <= fx.k.index <= 264:
        print(f"  FX({fx.k.index},{fx.type}) val={fx.val:.2f}")
        
# ----- Diff #34: d[346→357] vs d[346→363] -----
print("\n" + "=" * 70)
print("Diff #34: cl_open d[346→357] vs pyarmor d[346→363]")
print("=" * 70)

print("\nFXs from index 340 to 380:")
for fx in fxs:
    if 340 <= fx.k.index <= 380:
        print(f"  FX({fx.k.index},{fx.type}) val={fx.val:.2f} k_idx={fx.k.k_index} h={fx.high(qj,qy):.2f} l={fx.low(qj,qy):.2f}")

# Check what happens at FX(357)
fx_346 = fx_357 = fx_363 = None
for fx in fxs:
    if fx.k.index == 346: fx_346 = fx
    if fx.k.index == 357: fx_357 = fx
    if fx.k.index == 363: fx_363 = fx

if fx_357:
    print(f"\nFX(357): type={fx_357.type} val={fx_357.val:.2f}")
    print(f"FX(363): type={fx_363.type} val={fx_363.val:.2f}" if fx_363 else "FX(363): NOT FOUND")
    
    # Find what confirms the BI at 357 in cl_open
    # The BI ends at 357, so a confirming FX must exist after 357
    # that makes end_fx → confirming_fx a valid BI
    print(f"\nChecking what confirms at 357:")
    for fx in fxs:
        if fx.k.index > 357 and fx.k.index < 365:
            v = cd_open._bi_fx_valid(fx_357, fx)
            print(f"  _bi_fx_valid(FX{fx_357.k.index}, FX{fx.k.index}): {v} (type={fx.type})")

# ----- Diff #43: u[448→454] vs u[448→461] -----
print("\n" + "=" * 70)
print("Diff #43: cl_open u[448→454] vs pyarmor u[448→461]")
print("=" * 70)

print("\nFXs from index 445 to 475:")
for fx in fxs:
    if 445 <= fx.k.index <= 475:
        print(f"  FX({fx.k.index},{fx.type}) val={fx.val:.2f} k_idx={fx.k.k_index} h={fx.high(qj,qy):.2f} l={fx.low(qj,qy):.2f}")

fx_454 = fx_461 = fx_448 = None
for fx in fxs:
    if fx.k.index == 448: fx_448 = fx
    if fx.k.index == 454: fx_454 = fx
    if fx.k.index == 461: fx_461 = fx

if fx_454 and fx_461:
    print(f"\nFX(454): type={fx_454.type} val={fx_454.val:.2f}")
    print(f"FX(461): type={fx_461.type} val={fx_461.val:.2f}")
    print(f"FX(461).val >= FX(454).val? {fx_461.val >= fx_454.val}")

# Check what confirms at 454
if fx_454:
    print(f"\nChecking what confirms at 454:")
    for fx in fxs:
        if fx.k.index > 454 and fx.k.index < 470:
            v = cd_open._bi_fx_valid(fx_454, fx)
            if fx.type != fx_454.type:
                print(f"  _bi_fx_valid(FX{fx_454.k.index}, FX{fx.k.index}): {v} (type={fx.type}, val={fx.val:.2f})")

# ----- Diff #55: u[581→594] vs u[581→601] -----
print("\n" + "=" * 70)
print("Diff #55: cl_open u[581→594] vs pyarmor u[581→601]")
print("=" * 70)

print("\nFXs from index 578 to 615:")
for fx in fxs:
    if 578 <= fx.k.index <= 615:
        print(f"  FX({fx.k.index},{fx.type}) val={fx.val:.2f} k_idx={fx.k.k_index} h={fx.high(qj,qy):.2f} l={fx.low(qj,qy):.2f}")

fx_594 = fx_601 = fx_581 = None
for fx in fxs:
    if fx.k.index == 581: fx_581 = fx
    if fx.k.index == 594: fx_594 = fx
    if fx.k.index == 601: fx_601 = fx

if fx_594 and fx_601:
    print(f"\nFX(594): type={fx_594.type} val={fx_594.val:.2f}")
    print(f"FX(601): type={fx_601.type} val={fx_601.val:.2f}")
    print(f"FX(601).val >= FX(594).val? {fx_601.val >= fx_594.val}")

# Check what confirms at 594
if fx_594:
    print(f"\nChecking what confirms at 594:")
    for fx in fxs:
        if fx.k.index > 594 and fx.k.index < 610:
            v = cd_open._bi_fx_valid(fx_594, fx)
            if fx.type != fx_594.type:
                print(f"  _bi_fx_valid(FX{fx_594.k.index}, FX{fx.k.index}): {v} (type={fx.type}, val={fx.val:.2f})")
