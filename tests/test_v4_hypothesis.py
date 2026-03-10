# -*- coding: utf-8 -*-
"""Test the look-ahead hypothesis: when confirming, check if next FX is better endpoint"""
import sys, os, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor
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


def build_bis_v4(self, fxs):
    """v4: strict < / > for extension + look-ahead before confirmation"""
    bis = []
    if len(fxs) < 2:
        return bis

    start_fx = fxs[0]
    start_idx = 0
    end_fx = None
    end_idx = -1
    has_confirmed_bi = False

    i = 1
    while i < len(fxs):
        cur_fx = fxs[i]

        if end_fx is None:
            if cur_fx.type == start_fx.type:
                if not has_confirmed_bi:
                    if start_fx.type == "ding" and cur_fx.val > start_fx.val:
                        start_fx = cur_fx
                        start_idx = i
                    elif start_fx.type == "di" and cur_fx.val < start_fx.val:
                        start_fx = cur_fx
                        start_idx = i
            else:
                if self._bi_fx_valid(start_fx, cur_fx):
                    end_fx = cur_fx
                    end_idx = i
            i += 1
        else:
            if cur_fx.type == end_fx.type:
                # Extension with <= / >= (same as v3)
                if end_fx.type == "di" and cur_fx.val <= end_fx.val:
                    if self._bi_fx_valid(start_fx, cur_fx):
                        end_fx = cur_fx
                        end_idx = i
                elif end_fx.type == "ding" and cur_fx.val >= end_fx.val:
                    if self._bi_fx_valid(start_fx, cur_fx):
                        end_fx = cur_fx
                        end_idx = i
                i += 1
            else:
                confirm = self._bi_fx_valid(end_fx, cur_fx)
                # CGD check
                if confirm and self.bi_fx_cgd == Config.BI_FX_CHD_NO.value:
                    k_gap_confirm = cur_fx.k.k_index - end_fx.k.k_index
                    if k_gap_confirm < self.fx_check_k_nums:
                        for j in range(end_idx + 1, i):
                            mid_fx = fxs[j]
                            if mid_fx.type == cur_fx.type:
                                if cur_fx.type == "ding" and mid_fx.val > cur_fx.val:
                                    confirm = False
                                    break
                                elif cur_fx.type == "di" and mid_fx.val < cur_fx.val:
                                    confirm = False
                                    break

                # Look-ahead: check if next FX could extend endpoint better
                if confirm and i + 1 < len(fxs):
                    next_fx = fxs[i + 1]
                    if next_fx.type == end_fx.type:
                        better = False
                        if end_fx.type == "di" and next_fx.val < end_fx.val:
                            better = True
                        elif end_fx.type == "ding" and next_fx.val > end_fx.val:
                            better = True
                        if better and self._bi_fx_valid(start_fx, next_fx):
                            confirm = False

                if confirm:
                    bi_type = "down" if start_fx.type == "ding" else "up"
                    bi = BI(start=start_fx, end=end_fx, _type=bi_type,
                            index=len(bis), default_zs_type=self.default_bi_zs_type)
                    bis.append(bi)
                    has_confirmed_bi = True
                    start_fx = end_fx
                    start_idx = end_idx
                    end_fx = None
                    end_idx = -1
                    i = start_idx + 1
                else:
                    i += 1

    if end_fx is not None:
        bi_type = "down" if start_fx.type == "ding" else "up"
        bi = BI(start=start_fx, end=end_fx, _type=bi_type,
                index=len(bis), default_zs_type=self.default_bi_zs_type)
        bis.append(bi)

    return bis


# Test all 5 cases
for symbol, freq in [("BTC/USDT", "60m"), ("BTC/USDT", "5m"), ("BTC/USDT", "d"),
                      ("ETH/USDT", "60m"), ("ETH/USDT", "5m")]:
    df = load(symbol, freq)

    # cl_open with v4 _build_bis
    cd = CL_Open(symbol, freq, config={})
    cd._build_bis = types.MethodType(build_bis_v4, cd)

    # Capture pre-split
    pre_split_v4 = []
    orig_split = cd._bi_special_bi_split.__func__
    def cap(self, bis, ps=pre_split_v4):
        ps.extend(bis)
        return orig_split(self, bis)
    cd._bi_special_bi_split = types.MethodType(cap, cd)
    cd.process_klines(df)

    # pyarmor
    cd_py = CL_Pyarmor(symbol, freq, config={})
    cd_py.process_klines(df)

    # Reconstruct pyarmor pre-split
    py_bis = list(cd_py.bis)
    py_presplit = []
    j = 0
    while j < len(py_bis):
        bi = py_bis[j]
        if getattr(bi, 'is_split', False):
            merged = BI(start=bi.start, end=py_bis[j + 2].end, _type=bi.type,
                        index=len(py_presplit), default_zs_type=bi.default_zs_type)
            py_presplit.append(merged)
            j += 3
        else:
            py_presplit.append(bi)
            j += 1

    # Compare pre-split
    v4_count = len(pre_split_v4)
    py_ps_count = len(py_presplit)
    diffs = sum(1 for k in range(min(v4_count, py_ps_count))
                if pre_split_v4[k].start.k.index != py_presplit[k].start.k.index
                or pre_split_v4[k].end.k.index != py_presplit[k].end.k.index)

    status = "✅" if len(cd.bis) == len(cd_py.bis) else "❌"
    ps_status = "✅" if v4_count == py_ps_count and diffs == 0 else f"({diffs} diffs)"

    print(f"{symbol:>12} {freq:>4}: v4={len(cd.bis):>3} pyarmor={len(cd_py.bis):>3} {status}  "
          f"pre-split: v4={v4_count:>3} py={py_ps_count:>3} {ps_status}")

    if diffs > 0:
        for k in range(min(v4_count, py_ps_count)):
            o = pre_split_v4[k]
            p = py_presplit[k]
            if o.start.k.index != p.start.k.index or o.end.k.index != p.end.k.index:
                print(f"    #{k}: v4={o.type[0]}[{o.start.k.index}→{o.end.k.index}] "
                      f"py={p.type[0]}[{p.start.k.index}→{p.end.k.index}]")
