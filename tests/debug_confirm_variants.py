"""Test: skip strict check ONLY for confirmation BIs, keep for primary BIs,
and additionally keep the cl_gap < 4 for primary but use a different rule for confirmation.

Let's test multiple approaches:
1. No strict check for confirmation (original failed attempt)
2. No strict check for confirmation, BUT keep cl_gap >= 4 for confirmation gap check  
3. No strict check for confirmation, use k_gap >= 4 for confirmation gap check
4. Different: use k_gap for gap check in _bi_fx_valid, but only when called for confirmation
"""
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

datasets = [
    ("BTCd", "tests/test_data/BTC_USDT_d_500.parquet"),
    ("ETH60", "tests/test_data/ETH_USDT_60m_1000.parquet"),
    ("BTC60", "tests/test_data/BTC_USDT_60m_1000.parquet"),
    ("BTC5m", "tests/test_data/BTC_USDT_5m_1000.parquet"),
    ("ETH5m", "tests/test_data/ETH_USDT_5m_1000.parquet"),
]

# Save original methods
original_bi_fx_valid = CL_O._bi_fx_valid
original_build_bis = CL_O._build_bis

def make_build_bis_with_confirm_mode(confirm_valid_fn):
    """Create a _build_bis that uses confirm_valid_fn for confirmation checks."""
    def _build_bis_custom(self, fxs):
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
                    # CONFIRMATION: use custom function instead of _bi_fx_valid
                    confirm = confirm_valid_fn(self, end_fx, cur_fx)
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
    return _build_bis_custom

# Variant 1: confirmation uses gap-check only (no strict), cl_gap >= 4
def confirm_gap_only_cl(self, start_fx, end_fx):
    if start_fx.type == end_fx.type:
        return False
    cl_gap = end_fx.k.index - start_fx.k.index
    if cl_gap < 4:
        return False
    return True

# Variant 2: confirmation uses gap-check only (no strict), k_gap >= 4
def confirm_gap_only_k(self, start_fx, end_fx):
    if start_fx.type == end_fx.type:
        return False
    k_gap = end_fx.k.k_index - start_fx.k.k_index
    if k_gap < 4:
        return False
    return True

# Variant 3: confirmation uses k_gap >= 4 AND cl_gap >= 1
def confirm_k_gap_cl1(self, start_fx, end_fx):
    if start_fx.type == end_fx.type:
        return False
    cl_gap = end_fx.k.index - start_fx.k.index
    k_gap = end_fx.k.k_index - start_fx.k.k_index
    if cl_gap < 1:
        return False
    if k_gap < 4:
        return False
    return True

# Variant 4: full _bi_fx_valid (same as original = baseline)
def confirm_full(self, start_fx, end_fx):
    return self._bi_fx_valid(start_fx, end_fx)

variants = [
    ("baseline (cl+strict all)", confirm_full),
    ("confirm: cl>=4 no strict", confirm_gap_only_cl),
    ("confirm: k>=4 no strict", confirm_gap_only_k),
    ("confirm: cl>=1,k>=4 no strict", confirm_k_gap_cl1),
]

print(f"{'Dataset':>8} | {'pyarmor':>7}", end="")
for vname, _ in variants:
    print(f" | {vname:>25}", end="")
print()
print("-" * (8 + 10 + len(variants) * 28))

for name, path in datasets:
    df = pd.read_parquet(path)
    
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    n_p = len(cd_p.get_bis())
    
    print(f"{name:>8} | {n_p:>7}", end="")
    
    for vname, confirm_fn in variants:
        CL_O._build_bis = make_build_bis_with_confirm_mode(confirm_fn)
        cd_o = CL_O("test", "test", config=CL_CONFIG)
        cd_o.process_klines(df)
        n_o = len(cd_o.get_bis())
        marker = "✅" if n_o == n_p else "❌"
        print(f" | {n_o:>23}{marker}", end="")
    
    print()

# Restore original
CL_O._bi_fx_valid = original_bi_fx_valid
CL_O._build_bis = original_build_bis
