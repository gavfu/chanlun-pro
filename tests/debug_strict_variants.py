"""Test strict check variants - which specific strict condition to remove/modify."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P
from chanlun.cl_interface import Config

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

original_method = CL_O._bi_fx_valid

# Variant A: Only check condition 1 (not condition 2) in strict check
# For down BI (ding→di): only check start.low < end.low (remove end.high > start.high)
# For up BI (di→ding): only check start.high > end.high (remove end.low < start.low)
def variant_strict_cond1_only(self, start_fx, end_fx):
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
    if k_gap < self.fx_check_k_nums:
        if self.allow_bi_fx_strict:
            qj = self.fx_qj
            qy = self.fx_qy
            if start_fx.type == "ding" and end_fx.type == "di":
                if start_fx.low(qj, qy) < end_fx.low(qj, qy):
                    return False
                # REMOVED: end_fx.high > start_fx.high check
            elif start_fx.type == "di" and end_fx.type == "ding":
                if start_fx.high(qj, qy) > end_fx.high(qj, qy):
                    return False
                # REMOVED: end_fx.low < start_fx.low check
    return True

# Variant B: Only check condition 2 (not condition 1) in strict check
def variant_strict_cond2_only(self, start_fx, end_fx):
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
    if k_gap < self.fx_check_k_nums:
        if self.allow_bi_fx_strict:
            qj = self.fx_qj
            qy = self.fx_qy
            if start_fx.type == "ding" and end_fx.type == "di":
                # REMOVED: start.low < end.low check
                if end_fx.high(qj, qy) > start_fx.high(qj, qy):
                    return False
            elif start_fx.type == "di" and end_fx.type == "ding":
                # REMOVED: start.high > end.high check
                if end_fx.low(qj, qy) < start_fx.low(qj, qy):
                    return False
    return True

# Variant C: Use val instead of high/low for the check
def variant_strict_val(self, start_fx, end_fx):
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
    if k_gap < self.fx_check_k_nums:
        if self.allow_bi_fx_strict:
            if start_fx.type == "ding" and end_fx.type == "di":
                # Use val (center K-line's high/low) instead of interval
                if start_fx.val < end_fx.val:  # ding.val < di.val → invalid
                    return False
            elif start_fx.type == "di" and end_fx.type == "ding":
                if start_fx.val > end_fx.val:  # di.val > ding.val → invalid
                    return False
    return True

# Variant D: Use the merged K-line's h/l (not FX interval) for strict check
def variant_strict_k_hl(self, start_fx, end_fx):
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
    if k_gap < self.fx_check_k_nums:
        if self.allow_bi_fx_strict:
            if start_fx.type == "ding" and end_fx.type == "di":
                # Check merged K-line values (not FX interval)
                if start_fx.k.l < end_fx.k.l:
                    return False
                if end_fx.k.h > start_fx.k.h:
                    return False
            elif start_fx.type == "di" and end_fx.type == "ding":
                if start_fx.k.h > end_fx.k.h:
                    return False
                if end_fx.k.l < start_fx.k.l:
                    return False
    return True

# Variant E: No strict check at all (baseline with just gap)
def variant_no_strict(self, start_fx, end_fx):
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
    return True

variants = [
    ("baseline", original_method),
    ("cond1_only", variant_strict_cond1_only),
    ("cond2_only", variant_strict_cond2_only),
    ("val_check", variant_strict_val),
    ("k_hl_check", variant_strict_k_hl),
    ("no_strict", variant_no_strict),
]

print(f"{'Dataset':>8} | {'pyarmor':>7}", end="")
for vname, _ in variants:
    print(f" | {vname:>12}", end="")
print()
print("-" * (20 + len(variants) * 15))

for name, path in datasets:
    df = pd.read_parquet(path)
    
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    n_p = len(cd_p.get_bis())
    
    print(f"{name:>8} | {n_p:>7}", end="")
    
    for vname, fn in variants:
        CL_O._bi_fx_valid = fn
        cd_o = CL_O("test", "test", config=CL_CONFIG)
        cd_o.process_klines(df)
        n_o = len(cd_o.get_bis())
        marker = "✅" if n_o == n_p else "❌"
        print(f" | {n_o:>10}{marker}", end="")
    
    print()

CL_O._bi_fx_valid = original_method
