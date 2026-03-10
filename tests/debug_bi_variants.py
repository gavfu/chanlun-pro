"""Trace _bi_fx_valid calls during _build_bis for ETH60 to see which calls 
are rejected by gap check vs strict check, and compare with pyarmor."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P
from chanlun.cl_interface import FX

CL_CONFIG = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
}

# Test: with k_gap-based gap check but NO strict check, what BI count do we get?
# Monkey-patch _bi_fx_valid to test different rules

df = pd.read_parquet("tests/test_data/ETH_USDT_60m_1000.parquet")

# Reference: pyarmor  
cd_p = CL_P("test", "test", config=CL_CONFIG)
cd_p.process_klines(df)
print(f"Pyarmor: {len(cd_p.get_bis())} BIs")

# Variant 1: Original (cl_gap < 4 + strict check)
cd_o1 = CL_O("test", "test", config=CL_CONFIG)
cd_o1.process_klines(df)
print(f"Original (cl_gap<4 + strict): {len(cd_o1.get_bis())} BIs")

# Variant 2: k_gap < 4 + strict check (what we just tried)
original_method = CL_O._bi_fx_valid

def variant_k_gap_strict(self, start_fx, end_fx):
    if start_fx.type == end_fx.type:
        return False
    cl_gap = end_fx.k.index - start_fx.k.index
    k_gap = end_fx.k.k_index - start_fx.k.k_index
    # Use k_gap for gap check
    if k_gap < 4:
        return False
    # Keep strict check
    if k_gap < self.fx_check_k_nums:
        if self.allow_bi_fx_strict:
            qj = self.fx_qj
            qy = self.fx_qy
            if start_fx.type == "ding" and end_fx.type == "di":
                if start_fx.low(qj, qy) < end_fx.low(qj, qy):
                    return False
                if end_fx.high(qj, qy) > start_fx.high(qj, qy):
                    return False
            elif start_fx.type == "di" and end_fx.type == "ding":
                if start_fx.high(qj, qy) > end_fx.high(qj, qy):
                    return False
                if end_fx.low(qj, qy) < start_fx.low(qj, qy):
                    return False
    return True

CL_O._bi_fx_valid = variant_k_gap_strict
cd_o2 = CL_O("test", "test", config=CL_CONFIG)
cd_o2.process_klines(df)
print(f"k_gap<4 + strict: {len(cd_o2.get_bis())} BIs")

# Variant 3: k_gap < 4, NO strict check
def variant_k_gap_no_strict(self, start_fx, end_fx):
    if start_fx.type == end_fx.type:
        return False
    k_gap = end_fx.k.k_index - start_fx.k.k_index
    if k_gap < 4:
        return False
    return True

CL_O._bi_fx_valid = variant_k_gap_no_strict
cd_o3 = CL_O("test", "test", config=CL_CONFIG)
cd_o3.process_klines(df)
print(f"k_gap<4, no strict: {len(cd_o3.get_bis())} BIs")

# Variant 4: cl_gap < 4, NO strict check
def variant_cl_gap_no_strict(self, start_fx, end_fx):
    if start_fx.type == end_fx.type:
        return False
    cl_gap = end_fx.k.index - start_fx.k.index
    if cl_gap < 4:
        return False
    return True

CL_O._bi_fx_valid = variant_cl_gap_no_strict
cd_o4 = CL_O("test", "test", config=CL_CONFIG)
cd_o4.process_klines(df)
print(f"cl_gap<4, no strict: {len(cd_o4.get_bis())} BIs")

# Restore original
CL_O._bi_fx_valid = original_method

# Now repeat for all datasets
print("\n" + "="*60)
datasets = [
    ("BTCd", "tests/test_data/BTC_USDT_d_500.parquet"),
    ("ETH60", "tests/test_data/ETH_USDT_60m_1000.parquet"),
    ("BTC60", "tests/test_data/BTC_USDT_60m_1000.parquet"),
    ("BTC5m", "tests/test_data/BTC_USDT_5m_1000.parquet"),
    ("ETH5m", "tests/test_data/ETH_USDT_5m_1000.parquet"),
]

print(f"{'Dataset':>8} | {'pyarmor':>7} | {'cl+strict':>9} | {'k+strict':>8} | {'k+none':>6} | {'cl+none':>7}")
print("-"*65)

for name, path in datasets:
    df = pd.read_parquet(path)
    
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    n_p = len(cd_p.get_bis())
    
    cd_o1 = CL_O("test", "test", config=CL_CONFIG)
    cd_o1.process_klines(df)
    n_o1 = len(cd_o1.get_bis())
    
    CL_O._bi_fx_valid = variant_k_gap_strict
    cd_o2 = CL_O("test", "test", config=CL_CONFIG)
    cd_o2.process_klines(df)
    n_o2 = len(cd_o2.get_bis())
    
    CL_O._bi_fx_valid = variant_k_gap_no_strict
    cd_o3 = CL_O("test", "test", config=CL_CONFIG)
    cd_o3.process_klines(df)
    n_o3 = len(cd_o3.get_bis())
    
    CL_O._bi_fx_valid = variant_cl_gap_no_strict
    cd_o4 = CL_O("test", "test", config=CL_CONFIG)
    cd_o4.process_klines(df)
    n_o4 = len(cd_o4.get_bis())
    
    CL_O._bi_fx_valid = original_method
    
    m1 = "✅" if n_o1 == n_p else "❌"
    m2 = "✅" if n_o2 == n_p else "❌"
    m3 = "✅" if n_o3 == n_p else "❌"
    m4 = "✅" if n_o4 == n_p else "❌"
    print(f"{name:>8} | {n_p:>7} | {n_o1:>7}{m1} | {n_o2:>6}{m2} | {n_o3:>4}{m3} | {n_o4:>5}{m4}")
