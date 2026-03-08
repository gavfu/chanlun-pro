# -*- coding: utf-8 -*-
"""
Test hypothesis: when main stroke's k_gap >= fx_check_k_nums, 
confirmation uses relaxed (no strict) check.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
import chanlun.cl_open as cl_open_mod
from chanlun.cl_interface import FX, BI, Config
from typing import List

original_bi_fx_valid = cl_open_mod.CL._bi_fx_valid

def patched_build_bis(self, fxs: List[FX]) -> List[BI]:
    """Modified: when main stroke gap >= fx_check_k_nums, confirmation skips strict"""
    bis: List[BI] = []
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
            if cur_fx.type == start_fx.type:
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
                # CONFIRMATION
                # Check if main stroke gap >= threshold
                main_k_gap = end_fx.k.k_index - start_fx.k.k_index
                if main_k_gap >= self.fx_check_k_nums:
                    # Relaxed confirmation: gap-only, no strict
                    confirm = _gap_only_check(self, end_fx, cur_fx)
                else:
                    confirm = self._bi_fx_valid(end_fx, cur_fx)
                
                # CGD check
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
                    bi = BI(start=start_fx, end=end_fx, _type=bi_type, index=len(bis),
                            default_zs_type=self.default_bi_zs_type)
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
        bi = BI(start=start_fx, end=end_fx, _type=bi_type, index=len(bis),
                default_zs_type=self.default_bi_zs_type)
        bis.append(bi)
    return bis

def _gap_only_check(self, start_fx, end_fx):
    if start_fx.type == end_fx.type:
        return False
    cl_gap = end_fx.k.index - start_fx.k.index
    k_gap = end_fx.k.k_index - start_fx.k.k_index
    if self.bi_type == Config.BI_TYPE_DD.value:
        if cl_gap < 1: return False
    elif self.bi_type == Config.BI_TYPE_JDB.value:
        if k_gap < 4: return False
    else:
        if cl_gap < 4: return False
        if k_gap < 4: return False
    return True

cl_open_mod.CL._build_bis = patched_build_bis

from chanlun.cl_open import CL
from chanlun.cl_pyarmor import CL as CL_P

df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "test_data", "BTC_USDT_60m_500.parquet"))

cd = CL("BTC/USDT", "60m")
cd.process_klines(df)

cd_p = CL_P("BTC/USDT", "60m")
cd_p.process_klines(df)

print(f"=== Hypothesis: main k_gap >= fx_check_k_nums → relaxed confirm ===")
print(f"Our: {len(cd.bis)} strokes, Pyarmor: {len(cd_p.bis)} strokes\n")

max_n = max(len(cd.bis), len(cd_p.bis))
for i in range(max_n):
    o = cd.bis[i] if i < len(cd.bis) else None
    p = cd_p.bis[i] if i < len(cd_p.bis) else None
    o_str = f"{o.type:4s} {o.start.k.index:3d} -> {o.end.k.index:3d}" if o else "---"
    p_str = f"{p.type:4s} {p.start.k.index:3d} -> {p.end.k.index:3d}" if p else "---"
    match = o and p and o.type == p.type and o.start.k.index == p.start.k.index and o.end.k.index == p.end.k.index
    flag = "OK" if match else "DIFF"
    print(f"bi[{i:2d}] {o_str:20s}  {p_str:20s}  {flag}")
