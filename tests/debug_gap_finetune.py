"""Fine-tune the hybrid: cl_gap >= 4 OR k_gap >= X."""
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

def make_hybrid_and(cl_thresh, k_thresh):
    """Gap check: cl_gap >= cl_thresh AND k_gap >= k_thresh"""
    def variant(self, start_fx, end_fx):
        if start_fx.type == end_fx.type: return False
        cl_gap = end_fx.k.index - start_fx.k.index
        k_gap = end_fx.k.k_index - start_fx.k.k_index
        if self.bi_type == Config.BI_TYPE_DD.value:
            if cl_gap < 1: return False
        elif self.bi_type == Config.BI_TYPE_JDB.value:
            if k_gap < 4: return False
        else:
            if cl_gap < cl_thresh or k_gap < k_thresh:
                return False
        if k_gap < self.fx_check_k_nums:
            if self.allow_bi_fx_strict:
                qj = self.fx_qj
                qy = self.fx_qy
                if start_fx.type == "ding" and end_fx.type == "di":
                    if start_fx.low(qj, qy) < end_fx.low(qj, qy): return False
                    if end_fx.high(qj, qy) > start_fx.high(qj, qy): return False
                elif start_fx.type == "di" and end_fx.type == "ding":
                    if start_fx.high(qj, qy) > end_fx.high(qj, qy): return False
                    if end_fx.low(qj, qy) < start_fx.low(qj, qy): return False
        return True
    return variant

def make_hybrid_or(cl_thresh, k_thresh):
    """Gap check: cl_gap >= cl_thresh OR k_gap >= k_thresh"""
    def variant(self, start_fx, end_fx):
        if start_fx.type == end_fx.type: return False
        cl_gap = end_fx.k.index - start_fx.k.index
        k_gap = end_fx.k.k_index - start_fx.k.k_index
        if self.bi_type == Config.BI_TYPE_DD.value:
            if cl_gap < 1: return False
        elif self.bi_type == Config.BI_TYPE_JDB.value:
            if k_gap < 4: return False
        else:
            if cl_gap < cl_thresh and k_gap < k_thresh:
                return False
        if k_gap < self.fx_check_k_nums:
            if self.allow_bi_fx_strict:
                qj = self.fx_qj
                qy = self.fx_qy
                if start_fx.type == "ding" and end_fx.type == "di":
                    if start_fx.low(qj, qy) < end_fx.low(qj, qy): return False
                    if end_fx.high(qj, qy) > start_fx.high(qj, qy): return False
                elif start_fx.type == "di" and end_fx.type == "ding":
                    if start_fx.high(qj, qy) > end_fx.high(qj, qy): return False
                    if end_fx.low(qj, qy) < start_fx.low(qj, qy): return False
        return True
    return variant

# Test more OR hybrids
or_hybrids = [(4, 4), (4, 5), (4, 6), (4, 7), (4, 8)]
# Also AND hybrids (cl AND k)
and_hybrids = [(1, 4), (2, 4), (3, 4), (1, 5), (2, 5)]

print("=== OR hybrids (cl >= X OR k >= Y) ===")
print(f"{'Dataset':>8} | {'py':>4}", end="")
for cl, k in or_hybrids:
    print(f" | cl{cl}|k{k}", end="")
print()
print("-" * (14 + len(or_hybrids) * 9))

for name, path in datasets:
    df = pd.read_parquet(path)
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    n_p = len(cd_p.get_bis())
    print(f"{name:>8} | {n_p:>4}", end="")
    
    for cl, k in or_hybrids:
        CL_O._bi_fx_valid = make_hybrid_or(cl, k)
        cd_o = CL_O("test", "test", config=CL_CONFIG)
        cd_o.process_klines(df)
        n_o = len(cd_o.get_bis())
        m = "✅" if n_o == n_p else "❌"
        print(f" | {n_o:>5}{m}", end="")
    print()

print("\n=== AND hybrids (cl >= X AND k >= Y) ===")
print(f"{'Dataset':>8} | {'py':>4}", end="")
for cl, k in and_hybrids:
    print(f" | cl{cl}&k{k}", end="")
print()
print("-" * (14 + len(and_hybrids) * 10))

for name, path in datasets:
    df = pd.read_parquet(path)
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    n_p = len(cd_p.get_bis())
    print(f"{name:>8} | {n_p:>4}", end="")
    
    for cl, k in and_hybrids:
        CL_O._bi_fx_valid = make_hybrid_and(cl, k)
        cd_o = CL_O("test", "test", config=CL_CONFIG)
        cd_o.process_klines(df)
        n_o = len(cd_o.get_bis())
        m = "✅" if n_o == n_p else "❌"
        print(f" | {n_o:>6}{m}", end="")
    print()

CL_O._bi_fx_valid = original_method
