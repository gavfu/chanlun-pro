# -*- coding: utf-8 -*-
"""Diagnose why _select_split_up/down fails for specific BIs"""
import sys, os, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_interface import Config, FX, BI

DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')


LIMITS = {
    "BTC/USDT_d": 500, "BTC/USDT_60m": 1000, "BTC/USDT_5m": 1000,
    "ETH/USDT_60m": 1000, "ETH/USDT_5m": 1000,
}


def load(symbol, freq):
    limit = LIMITS.get(f"{symbol}_{freq}", 1000)
    cache_key = f"{symbol.replace('/', '_')}_{freq}_{limit}"
    return pd.read_parquet(os.path.join(DATA_DIR, f"{cache_key}.parquet"))


def make_cl(symbol, freq):
    df = load(symbol, freq)
    cd = CL_Open(symbol, freq, config={})
    pre_split = []
    orig_fn = cd._bi_special_bi_split.__func__

    def cap(self, bis):
        pre_split.extend(bis)
        return orig_fn(self, bis)

    cd._bi_special_bi_split = types.MethodType(cap, cd)
    cd.process_klines(df)
    return cd, pre_split


def trace_split(cd, pre_split, bi_start, bi_end, bi_type):
    """Trace split selection for a specific BI"""
    qj, qy = cd.fx_qj, cd.fx_qy

    # Find BI in pre-split
    target = None
    for bi in pre_split:
        if bi.start.k.index == bi_start and bi.end.k.index == bi_end:
            target = bi
            break
    if target is None:
        print(f"  BI {bi_type}[{bi_start}→{bi_end}] NOT FOUND in pre-split")
        for bi in pre_split:
            if abs(bi.start.k.index - bi_start) < 10 or abs(bi.end.k.index - bi_end) < 10:
                print(f"    Nearby: {bi.type}[{bi.start.k.index}→{bi.end.k.index}]")
        return

    print(f"\n{'='*70}")
    print(f"  Split trace: {bi_type}[{bi_start}→{bi_end}]")
    print(f"  start: FX({target.start.k.index},{target.start.type}) k_idx={target.start.k.k_index} "
          f"h={target.start.high(qj,qy):.2f} l={target.start.low(qj,qy):.2f}")
    print(f"  end:   FX({target.end.k.index},{target.end.type}) k_idx={target.end.k.k_index} "
          f"h={target.end.high(qj,qy):.2f} l={target.end.low(qj,qy):.2f}")
    print(f"  bi_type={cd.bi_type}, fx_check_k_nums={cd.fx_check_k_nums}, strict={cd.allow_bi_fx_strict}")
    print(f"{'='*70}")

    internal = [fx for fx in cd.fxs if target.start.k.index < fx.k.index < target.end.k.index]
    print(f"\nInternal FXs ({len(internal)}):")
    for fx in internal:
        print(f"  FX({fx.k.index},{fx.type}) val={fx.val:.2f} k_idx={fx.k.k_index} "
              f"h={fx.high(qj,qy):.2f} l={fx.low(qj,qy):.2f}")

    if bi_type == "up":
        # _select_split_up: start(di)→ding_fx→di_fx→end(ding)
        first_fxs = sorted([f for f in internal if f.type == "ding"], key=lambda f: f.val, reverse=True)
        second_fxs = sorted([f for f in internal if f.type == "di"], key=lambda f: f.val)
        order_check = lambda a, b: a.k.k_index < b.k.k_index
        gap_fn = lambda a, b: b.k.index - a.k.index
    else:
        # _select_split_down: start(ding)→di_fx→ding_fx→end(di)
        first_fxs = sorted([f for f in internal if f.type == "di"], key=lambda f: f.val)
        second_fxs = sorted([f for f in internal if f.type == "ding"], key=lambda f: f.val, reverse=True)
        order_check = lambda a, b: a.k.k_index < b.k.k_index
        gap_fn = lambda a, b: b.k.index - a.k.index

    print(f"\nChecking pairs:")
    for fx1 in first_fxs:
        for fx2 in second_fxs:
            label = f"({fx1.type[0]}FX{fx1.k.index}, {fx2.type[0]}FX{fx2.k.index})"

            if not order_check(fx1, fx2):
                continue
            gap = gap_fn(fx1, fx2)
            if gap < 4:
                print(f"  {label}: SKIP gap={gap}<4")
                continue

            # Check start→fx1
            v1 = cd._bi_fx_valid(target.start, fx1)
            s, e = target.start, fx1
            r1 = _explain(cd, s, e, qj, qy)

            # Check fx2→end
            v2 = cd._bi_fx_valid(fx2, target.end)
            r2 = _explain(cd, fx2, target.end, qj, qy)

            st = "✅" if v1 and v2 else "❌"
            print(f"  {st} {label} gap={gap}: start→fx1: {v1}({r1}), fx2→end: {v2}({r2})")
            if v1 and v2:
                if bi_type == "up":
                    print(f"     → up[{bi_start}→{fx1.k.index}] down[{fx1.k.index}→{fx2.k.index}] up[{fx2.k.index}→{bi_end}]")
                else:
                    print(f"     → down[{bi_start}→{fx1.k.index}] up[{fx1.k.index}→{fx2.k.index}] down[{fx2.k.index}→{bi_end}]")
                return
    print("\n  NO valid split found!")


def _explain(cd, s, e, qj, qy):
    if s.type == e.type:
        return f"same_type"
    cl_gap = e.k.index - s.k.index
    k_gap = e.k.k_index - s.k.k_index
    if cd.bi_type == Config.BI_TYPE_OLD.value and cl_gap < 4:
        return f"cl_gap={cl_gap}<4"
    if cd.bi_type == Config.BI_TYPE_JDB.value and k_gap < 4:
        return f"k_gap={k_gap}<4"
    if cd.bi_type == Config.BI_TYPE_DD.value and cl_gap < 1:
        return f"cl_gap={cl_gap}<1"
    if k_gap < cd.fx_check_k_nums and cd.allow_bi_fx_strict:
        if s.type == "ding" and e.type == "di":
            if s.low(qj, qy) < e.low(qj, qy):
                return f"strict: s.low={s.low(qj,qy):.2f}<e.low={e.low(qj,qy):.2f}"
            if e.high(qj, qy) > s.high(qj, qy):
                return f"strict: e.high={e.high(qj,qy):.2f}>s.high={s.high(qj,qy):.2f}"
        elif s.type == "di" and e.type == "ding":
            if s.high(qj, qy) > e.high(qj, qy):
                return f"strict: s.high={s.high(qj,qy):.2f}>e.high={e.high(qj,qy):.2f}"
            if e.low(qj, qy) < s.low(qj, qy):
                return f"strict: e.low={e.low(qj,qy):.2f}<s.low={s.low(qj,qy):.2f}"
    return "OK"


# ---- BTCd ----
print("=" * 70)
print("BTCd: Checking up[330→352]")
print("=" * 70)
cd_d, ps_d = make_cl("BTC/USDT", "d")
trace_split(cd_d, ps_d, 330, 352, "up")

# ---- BTC5m ----
print("\n\n" + "=" * 70)
print("BTC5m: Checking down[429→445]")
print("=" * 70)
cd_5m, ps_5m = make_cl("BTC/USDT", "5m")
trace_split(cd_5m, ps_5m, 429, 445, "down")
