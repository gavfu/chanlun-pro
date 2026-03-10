# -*- coding: utf-8 -*-
"""
Test hypothesis: remove start_fx replacement + always use _bi_fx_valid.
Compare all 5 test cases.
"""
import os, sys, types, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor
from chanlun.cl_interface import FX, BI, Config

DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")

def load(symbol, freq, limit):
    cache_key = f"{symbol.replace('/', '_')}_{freq}_{limit}"
    return pd.read_parquet(os.path.join(DATA_DIR, f"{cache_key}.parquet"))


def build_bis_v2(self, fxs):
    """
    Hypothesis: no start_fx replacement, always use _bi_fx_valid.
    """
    bis = []
    if len(fxs) < 2:
        return bis

    start_fx = fxs[0]
    start_idx = 0
    end_fx = None
    end_idx = -1

    i = 1
    while i < len(fxs):
        cur_fx = fxs[i]

        if end_fx is None:
            if cur_fx.type != start_fx.type:
                # Opposite type: check _bi_fx_valid (always strict)
                if self._bi_fx_valid(start_fx, cur_fx):
                    end_fx = cur_fx
                    end_idx = i
            # Same type: do NOT replace start_fx
            i += 1
        else:
            if cur_fx.type == end_fx.type:
                # Same as end: extend if better AND valid
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
                # Opposite to end: try to confirm
                confirm = self._bi_fx_valid(end_fx, cur_fx)
                if confirm and self.bi_fx_cgd == Config.BI_FX_CHD_NO.value:
                    k_gap_confirm = cur_fx.k.k_index - end_fx.k.k_index
                    if k_gap_confirm < self.fx_check_k_nums:
                        for j in range(end_idx + 1, i):
                            mid_fx = fxs[j]
                            if mid_fx.type == cur_fx.type:
                                if cur_fx.type == "ding" and mid_fx.val > cur_fx.val:
                                    confirm = False; break
                                elif cur_fx.type == "di" and mid_fx.val < cur_fx.val:
                                    confirm = False; break
                if confirm:
                    bi_type = "down" if start_fx.type == "ding" else "up"
                    bi = BI(start=start_fx, end=end_fx, _type=bi_type,
                            index=len(bis), default_zs_type=self.default_bi_zs_type)
                    bis.append(bi)
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


def build_bis_v3(self, fxs):
    """
    Hypothesis v3: replace start_fx ONLY before first BI confirmed.
    Always use _bi_fx_valid (no relaxation).
    """
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
                # Same type: replace start_fx ONLY if no BI confirmed yet
                if not has_confirmed_bi:
                    if start_fx.type == "ding" and cur_fx.val > start_fx.val:
                        start_fx = cur_fx
                        start_idx = i
                    elif start_fx.type == "di" and cur_fx.val < start_fx.val:
                        start_fx = cur_fx
                        start_idx = i
            else:
                # Opposite type: always use _bi_fx_valid
                if self._bi_fx_valid(start_fx, cur_fx):
                    end_fx = cur_fx
                    end_idx = i
            i += 1
        else:
            if cur_fx.type == end_fx.type:
                # Same as end: extend if better AND valid
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
                # Opposite to end: try to confirm
                confirm = self._bi_fx_valid(end_fx, cur_fx)
                if confirm and self.bi_fx_cgd == Config.BI_FX_CHD_NO.value:
                    k_gap_confirm = cur_fx.k.k_index - end_fx.k.k_index
                    if k_gap_confirm < self.fx_check_k_nums:
                        for j in range(end_idx + 1, i):
                            mid_fx = fxs[j]
                            if mid_fx.type == cur_fx.type:
                                if cur_fx.type == "ding" and mid_fx.val > cur_fx.val:
                                    confirm = False; break
                                elif cur_fx.type == "di" and mid_fx.val < cur_fx.val:
                                    confirm = False; break
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


def test_case(symbol, freq, limit):
    df = load(symbol, freq, limit)

    # Pyarmor reference
    cd_p = CL_Pyarmor(symbol, freq, config={})
    cd_p.process_klines(df)
    p_bis = cd_p.get_bis()

    # cl_open with original _build_bis (get pre-split count)
    cd_o = CL_Open(symbol, freq, config={})
    orig_pre = []
    orig_fn = cd_o._bi_special_bi_split.__func__
    def cap_orig(self, bis):
        orig_pre.extend(bis)
        return orig_fn(self, bis)
    cd_o._bi_special_bi_split = types.MethodType(cap_orig, cd_o)
    cd_o.process_klines(df)

    # cl_open with modified _build_bis (v2: no replacement at all)
    cd_v2 = CL_Open(symbol, freq, config={})
    v2_pre = []
    def cap_v2(self, bis):
        v2_pre.extend(bis)
        return orig_fn(self, bis)
    cd_v2._build_bis = types.MethodType(build_bis_v2, cd_v2)
    cd_v2._bi_special_bi_split = types.MethodType(cap_v2, cd_v2)
    cd_v2.process_klines(df)
    v2_bis = cd_v2.get_bis()

    # cl_open with modified _build_bis (v3: replace only before first BI)
    cd_v3 = CL_Open(symbol, freq, config={})
    v3_pre = []
    def cap_v3(self, bis):
        v3_pre.extend(bis)
        return orig_fn(self, bis)
    cd_v3._build_bis = types.MethodType(build_bis_v3, cd_v3)
    cd_v3._bi_special_bi_split = types.MethodType(cap_v3, cd_v3)
    cd_v3.process_klines(df)

    # Reconstruct pyarmor pre-split from is_split
    # Count pyarmor pre-split: total - 2 * number_of_split_groups
    # Each split group has 3 consecutive sub-BIs, adding 2 extra BIs
    split_groups = 0
    i = 0
    while i < len(p_bis):
        if p_bis[i].is_split:
            split_groups += 1
            i += 3  # skip the triplet
        else:
            i += 1
    p_pre = len(p_bis) - 2 * split_groups

    # Reconstruct pyarmor pre-split BI sequence
    p_pre_bis = []
    i = 0
    while i < len(p_bis):
        if p_bis[i].is_split:
            # merge triplet back
            p_pre_bis.append((p_bis[i].type, p_bis[i].start.k.index, p_bis[i+2].end.k.index))
            i += 3
        else:
            p_pre_bis.append((p_bis[i].type, p_bis[i].start.k.index, p_bis[i].end.k.index))
            i += 1

    print(f"\n{symbol} {freq}:")
    print(f"  Pyarmor:      final={len(p_bis):>3}, pre_split≈{p_pre}")
    print(f"  Open(orig):   final={len(cd_o.get_bis()):>3}, pre_split={len(orig_pre)}")
    print(f"  Open(v2):     final={len(v2_bis):>3}, pre_split={len(v2_pre)}")
    print(f"  Open(v3):     final={len(cd_v3.get_bis()):>3}, pre_split={len(v3_pre)}")

    # Compare v3 pre-split with pyarmor
    v3_pre_tuples = [(b.type, b.start.k.index, b.end.k.index) for b in v3_pre]
    diffs_v3 = 0
    max_i = min(len(p_pre_bis), len(v3_pre_tuples))
    for j in range(max_i):
        if p_pre_bis[j] != v3_pre_tuples[j]:
            if diffs_v3 < 5:
                print(f"    v3 #{j}: v3={v3_pre_tuples[j]} pyarmor_pre={p_pre_bis[j]}")
            diffs_v3 += 1
    if len(p_pre_bis) != len(v3_pre_tuples):
        print(f"    v3 Count: v3={len(v3_pre_tuples)} vs pyarmor_pre={len(p_pre_bis)}")
    if diffs_v3 == 0 and len(p_pre_bis) == len(v3_pre_tuples):
        print(f"    v3 Pre-split: ALL MATCH ✅")
    else:
        print(f"    v3 Total diffs: {diffs_v3}")

    # Compare v2 pre-split with pyarmor
    v2_pre_tuples = [(b.type, b.start.k.index, b.end.k.index) for b in v2_pre]
    diffs_v2 = 0
    max_i = min(len(p_pre_bis), len(v2_pre_tuples))
    for j in range(max_i):
        if p_pre_bis[j] != v2_pre_tuples[j]:
            diffs_v2 += 1
    if len(p_pre_bis) != len(v2_pre_tuples):
        print(f"    v2 Count: v2={len(v2_pre_tuples)} vs pyarmor_pre={len(p_pre_bis)}, diffs={diffs_v2}")
    elif diffs_v2 == 0:
        print(f"    v2 Pre-split: ALL MATCH ✅")
    else:
        print(f"    v2 Total diffs: {diffs_v2}")


if __name__ == "__main__":
    cases = [
        ("BTC/USDT", "60m", 1000),
        ("BTC/USDT", "5m", 1000),
        ("BTC/USDT", "d", 500),
        ("ETH/USDT", "60m", 1000),
        ("ETH/USDT", "5m", 1000),
    ]
    for s, f, l in cases:
        test_case(s, f, l)
