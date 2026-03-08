# -*- coding: utf-8 -*-
"""
Test hypothesis: strict check only on end_fx setting, NOT on confirmation.
Modify _bi_fx_valid to accept skip_strict flag for confirmation calls.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd

# Monkey-patch cl_open to test hypothesis
import chanlun.cl_open as cl_open_mod
from chanlun.cl_interface import FX, BI, Config
from typing import List

original_build_bis = cl_open_mod.CL._build_bis

def patched_build_bis(self, fxs: List[FX]) -> List[BI]:
    """Modified _build_bis where confirmation uses _bi_fx_valid WITHOUT strict check"""
    bis: List[BI] = []
    if len(fxs) < 2:
        return bis

    fx_pos = {id(fx): idx for idx, fx in enumerate(fxs)}

    start_fx = fxs[0]
    start_idx = 0
    end_fx = None
    end_idx = -1

    i = 1
    while i < len(fxs):
        cur_fx = fxs[i]

        if end_fx is None:
            if cur_fx.type == start_fx.type:
                if start_fx.type == "ding" and cur_fx.val > start_fx.val:
                    start_fx = cur_fx
                    start_idx = i
                elif start_fx.type == "di" and cur_fx.val < start_fx.val:
                    start_fx = cur_fx
                    start_idx = i
            else:
                if self._bi_fx_valid(start_fx, cur_fx):  # strict ON for end_fx setting
                    end_fx = cur_fx
                    end_idx = i
            i += 1
        else:
            if cur_fx.type == end_fx.type:
                if end_fx.type == "di" and cur_fx.val <= end_fx.val:
                    if self._bi_fx_valid(start_fx, cur_fx):  # strict ON for extension
                        end_fx = cur_fx
                        end_idx = i
                elif end_fx.type == "ding" and cur_fx.val >= end_fx.val:
                    if self._bi_fx_valid(start_fx, cur_fx):
                        end_fx = cur_fx
                        end_idx = i
                i += 1
            else:
                # CONFIRMATION: use gap-only check (no strict)
                confirm = _bi_fx_valid_gap_only(self, end_fx, cur_fx)
                # CGD check still applies
                if confirm and self.bi_fx_cgd == Config.BI_FX_CHD_YES.value:
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
                if confirm:
                    bi_type = "down" if start_fx.type == "ding" else "up"
                    bi = BI(
                        start=start_fx,
                        end=end_fx,
                        _type=bi_type,
                        index=len(bis),
                        default_zs_type=self.default_bi_zs_type,
                    )
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
        bi = BI(
            start=start_fx,
            end=end_fx,
            _type=bi_type,
            index=len(bis),
            default_zs_type=self.default_bi_zs_type,
        )
        bis.append(bi)

    return bis


def _bi_fx_valid_gap_only(self, start_fx: FX, end_fx: FX) -> bool:
    """Only check gap (type alternation + distance), no strict check"""
    if start_fx.type == end_fx.type:
        return False

    cl_gap = end_fx.k.index - start_fx.k.index
    k_gap = end_fx.k.k_index - start_fx.k.k_index

    if self.bi_type == Config.BI_TYPE_DD.value:
        if cl_gap < 1:
            return False
    elif self.bi_type == Config.BI_TYPE_JDB.value:
        if k_gap < 4:
            return False
    else:
        if cl_gap < 4:
            return False
        if k_gap < 4:
            return False

    return True


# Apply patch
cl_open_mod.CL._build_bis = patched_build_bis

from chanlun.cl_open import CL

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "test_data", "BTC_USDT_60m_500.parquet"))
cd = CL("BTC/USDT", "60m")
cd.process_klines(df)

print("=== Hypothesis: strict only on end_fx, not confirmation ===")
for i, bi in enumerate(cd.bis):
    direction = "up" if bi.type == "up" else "down"
    start_idx = bi.start.k.index
    end_idx = bi.end.k.index
    print(f"bi[{i:2d}] {direction:4s} {start_idx:3d} -> {end_idx:3d}")

# Compare with pyarmor
from chanlun.cl_pyarmor import CL as CL_P
cd_p = CL_P("BTC/USDT", "60m")
cd_p.process_klines(df)

print(f"\n=== Pyarmor ({len(cd_p.bis)} strokes) ===")
for i, bi in enumerate(cd_p.bis):
    direction = "up" if bi.type == "up" else "down"
    start_idx = bi.start.k.index
    end_idx = bi.end.k.index
    print(f"bi[{i:2d}] {direction:4s} {start_idx:3d} -> {end_idx:3d}")

# Compare
print(f"\n=== Comparison ===")
print(f"Our strokes: {len(cd.bis)}")
print(f"Pyarmor strokes: {len(cd_p.bis)}")

match_count = 0
for i in range(min(len(cd.bis), len(cd_p.bis))):
    our = cd.bis[i]
    their = cd_p.bis[i]
    match = (our.start.k.index == their.start.k.index and 
             our.end.k.index == their.end.k.index and
             our.type == their.type)
    if match:
        match_count += 1
    else:
        print(f"  FIRST DIFF at bi[{i}]: ours={our.type} {our.start.k.index}->{our.end.k.index}, theirs={their.type} {their.start.k.index}->{their.end.k.index}")
        break
print(f"Matching strokes: {match_count}/{min(len(cd.bis), len(cd_p.bis))}")
