"""Test: does pyarmor use different FX high/low computation in strict check?
Options: fx_qj_ck (merged), fx_qy_middle (center K only), or ck values directly."""
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

# Variant 1: Use fx_qj_ck (merged K-line values) for strict check instead of fx_qj_k
def variant_ck_three(self, start_fx, end_fx):
    if start_fx.type == end_fx.type: return False
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
            # Use merged K-line (ck) values with three-kline range
            qj = Config.FX_QJ_CK.value  # ck instead of k
            qy = Config.FX_QY_THREE.value
            if start_fx.type == "ding" and end_fx.type == "di":
                if start_fx.low(qj, qy) < end_fx.low(qj, qy): return False
                if end_fx.high(qj, qy) > start_fx.high(qj, qy): return False
            elif start_fx.type == "di" and end_fx.type == "ding":
                if start_fx.high(qj, qy) > end_fx.high(qj, qy): return False
                if end_fx.low(qj, qy) < start_fx.low(qj, qy): return False
    return True

# Variant 2: Use center K-line only (fx_qy_middle) for strict check
def variant_k_middle(self, start_fx, end_fx):
    if start_fx.type == end_fx.type: return False
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
            qj = Config.FX_QJ_K.value
            qy = Config.FX_QY_MIDDLE.value  # middle instead of three
            if start_fx.type == "ding" and end_fx.type == "di":
                if start_fx.low(qj, qy) < end_fx.low(qj, qy): return False
                if end_fx.high(qj, qy) > start_fx.high(qj, qy): return False
            elif start_fx.type == "di" and end_fx.type == "ding":
                if start_fx.high(qj, qy) > end_fx.high(qj, qy): return False
                if end_fx.low(qj, qy) < start_fx.low(qj, qy): return False
    return True

# Variant 3: Use merged K-line center only (ck_middle)
def variant_ck_middle(self, start_fx, end_fx):
    if start_fx.type == end_fx.type: return False
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
            qj = Config.FX_QJ_CK.value
            qy = Config.FX_QY_MIDDLE.value
            if start_fx.type == "ding" and end_fx.type == "di":
                if start_fx.low(qj, qy) < end_fx.low(qj, qy): return False
                if end_fx.high(qj, qy) > start_fx.high(qj, qy): return False
            elif start_fx.type == "di" and end_fx.type == "ding":
                if start_fx.high(qj, qy) > end_fx.high(qj, qy): return False
                if end_fx.low(qj, qy) < start_fx.low(qj, qy): return False
    return True

# Variant 4: Use just the center merged K-line's h/l directly (simplest)
def variant_direct_k(self, start_fx, end_fx):
    if start_fx.type == end_fx.type: return False
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
                if start_fx.k.l < end_fx.k.l: return False
                if end_fx.k.h > start_fx.k.h: return False
            elif start_fx.type == "di" and end_fx.type == "ding":
                if start_fx.k.h > end_fx.k.h: return False
                if end_fx.k.l < start_fx.k.l: return False
    return True

# Variant 5: Condition 1 only with ck three values
def variant_ck_three_c1(self, start_fx, end_fx):
    if start_fx.type == end_fx.type: return False
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
            qj = Config.FX_QJ_CK.value
            qy = Config.FX_QY_THREE.value
            if start_fx.type == "ding" and end_fx.type == "di":
                if start_fx.low(qj, qy) < end_fx.low(qj, qy): return False
            elif start_fx.type == "di" and end_fx.type == "ding":
                if start_fx.high(qj, qy) > end_fx.high(qj, qy): return False
    return True

variants = [
    ("baseline", original_method),
    ("ck_three", variant_ck_three),
    ("k_middle", variant_k_middle),
    ("ck_middle", variant_ck_middle),
    ("direct_k", variant_direct_k),
    ("ck3_c1", variant_ck_three_c1),
]

print(f"{'Dataset':>8} | {'pyarmor':>7}", end="")
for vname, _ in variants:
    print(f" | {vname:>10}", end="")
print()
print("-" * (20 + len(variants) * 13))

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
        print(f" | {n_o:>8}{marker}", end="")
    
    print()

CL_O._bi_fx_valid = original_method
