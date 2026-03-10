"""Test: confirmation uses val-based check instead of strict FX interval check.
i.e., for up BI confirmation (di→ding), only check ding.val > di.val (obviously true).
Actually, the val check is if the FX vals are in correct relationship — which they always are
for a valid BI direction. So confirmation just needs gap check.

Let me try EXACTLY: confirmation skips strict check IF cl_gap >= 4 (which means 
the gap check was ok). Only apply strict for primary BI.
No wait - I already tested this. "confirm: cl>=4 no strict" gave ETH5m=73 but broke others.

The question: why does removing strict for confirmation break ETH60/BTC60/BTC5m?
Let me trace WHICH specific confirmation rejections cause the extra BIs in ETH60."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P
from chanlun.cl_interface import FX, BI, Config

CL_CONFIG = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
}

# Trace _build_bis for ETH60 with confirmation strict disabled
# to see WHERE the extra BIs come from

def build_bis_trace(self, fxs, trace=False):
    """Custom _build_bis that traces confirmation rejections."""
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
                # CONFIRMATION: skip strict check, only gap check
                confirm_orig = self._bi_fx_valid(end_fx, cur_fx)
                
                # Gap-only check for confirmation
                cl_gap = cur_fx.k.index - end_fx.k.index
                k_gap = cur_fx.k.k_index - end_fx.k.k_index
                confirm_nostrict = (cl_gap >= 4 and end_fx.type != cur_fx.type)
                
                if confirm_nostrict and self.bi_fx_cgd == Config.BI_FX_CHD_NO.value:
                    k_gap_confirm = cur_fx.k.k_index - end_fx.k.k_index
                    if k_gap_confirm < self.fx_check_k_nums:
                        for j in range(end_idx + 1, i):
                            mid_fx = fxs[j]
                            if mid_fx.type == cur_fx.type:
                                if cur_fx.type == "ding" and mid_fx.val > cur_fx.val:
                                    confirm_nostrict = False
                                    break
                                elif cur_fx.type == "di" and mid_fx.val < cur_fx.val:
                                    confirm_nostrict = False
                                    break
                
                # Log cases where strict differs from no-strict
                if confirm_nostrict and not confirm_orig and trace:
                    print(f"  EXTRA CONFIRM: end={end_fx.type}@{end_fx.k.k_index} → "
                          f"cur={cur_fx.type}@{cur_fx.k.k_index} "
                          f"cl_gap={cl_gap} k_gap={k_gap} "
                          f"(BI: {start_fx.type}@{start_fx.k.k_index}→{end_fx.type}@{end_fx.k.k_index})")
                
                if confirm_nostrict:
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

# Trace ETH60
datasets = [
    ("ETH60", "tests/test_data/ETH_USDT_60m_1000.parquet"),
    ("BTC60", "tests/test_data/BTC_USDT_60m_1000.parquet"),
]

for name, path in datasets:
    print(f"\n{'='*60}")
    print(f" {name}")
    print(f"{'='*60}")
    
    df = pd.read_parquet(path)
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    
    cd_o = CL_O("test", "test", config=CL_CONFIG)
    cd_o.process_klines(df)
    
    fxs = cd_o.get_fxs()
    
    orig_build_bis = CL_O._build_bis
    CL_O._build_bis = lambda self, fxs: build_bis_trace(self, fxs, trace=True)
    cd_test = CL_O("test", "test", config=CL_CONFIG)
    cd_test.process_klines(df)
    CL_O._build_bis = orig_build_bis
    
    print(f"\nPyarmor: {len(cd_p.get_bis())} BIs")
    print(f"Original: {len(cd_o.get_bis())} BIs")
    print(f"No-strict-confirm: {len(cd_test.get_bis())} BIs")
