"""Test different cl_gap thresholds for _bi_fx_valid."""
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

def make_variant(cl_threshold):
    def variant(self, start_fx, end_fx):
        if start_fx.type == end_fx.type: return False
        cl_gap = end_fx.k.index - start_fx.k.index
        k_gap = end_fx.k.k_index - start_fx.k.k_index
        if self.bi_type == Config.BI_TYPE_DD.value:
            if cl_gap < 1: return False
        elif self.bi_type == Config.BI_TYPE_JDB.value:
            if k_gap < 4: return False
        else:
            if cl_gap < cl_threshold: return False
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

# Also test hybrid: cl_gap >= 4 OR k_gap >= some_threshold
def make_hybrid(cl_thresh, k_thresh):
    def variant(self, start_fx, end_fx):
        if start_fx.type == end_fx.type: return False
        cl_gap = end_fx.k.index - start_fx.k.index
        k_gap = end_fx.k.k_index - start_fx.k.k_index
        if self.bi_type == Config.BI_TYPE_DD.value:
            if cl_gap < 1: return False
        elif self.bi_type == Config.BI_TYPE_JDB.value:
            if k_gap < 4: return False
        else:
            # Accept if EITHER cl_gap >= cl_thresh OR k_gap >= k_thresh
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

thresholds = [1, 2, 3, 4, 5]
hybrids = [(4, 5), (4, 6), (3, 5), (3, 6), (2, 5)]

print(f"{'Dataset':>8} | {'py':>4}", end="")
for t in thresholds:
    print(f" | cl>={t:>1}", end="")
for cl, k in hybrids:
    print(f" | cl{cl}k{k}", end="")
print()
print("-" * (14 + len(thresholds) * 8 + len(hybrids) * 8))

for name, path in datasets:
    df = pd.read_parquet(path)
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    n_p = len(cd_p.get_bis())
    
    print(f"{name:>8} | {n_p:>4}", end="")
    
    for t in thresholds:
        CL_O._bi_fx_valid = make_variant(t)
        cd_o = CL_O("test", "test", config=CL_CONFIG)
        cd_o.process_klines(df)
        n_o = len(cd_o.get_bis())
        m = "✅" if n_o == n_p else "❌"
        print(f" | {n_o:>4}{m}", end="")
    
    for cl, k in hybrids:
        CL_O._bi_fx_valid = make_hybrid(cl, k)
        cd_o = CL_O("test", "test", config=CL_CONFIG)
        cd_o.process_klines(df)
        n_o = len(cd_o.get_bis())
        m = "✅" if n_o == n_p else "❌"
        print(f" | {n_o:>4}{m}", end="")
    
    print()

CL_O._bi_fx_valid = original_method
