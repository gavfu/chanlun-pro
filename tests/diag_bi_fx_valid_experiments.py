"""
Experiment: compare pre-split bis with different _bi_fx_valid configurations
to find the root cause of divergence.

Test 1: Original (cl_gap < 4 + strict check)
Test 2: No strict check (cl_gap < 4 only)
Test 3: cl_gap < 5 (stricter gap)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P
from chanlun.cl_interface import Config, FX

CL_CONFIG = {
    'bi_type': 'bi_type_old', 'fx_qj': 'fx_qj_k', 'fx_qy': 'fx_qy_three',
    'bi_fx_cgd': 'bi_fx_cgd_yes', 'fx_check_k_nums': 13,
    'bi_split_k_cross_nums': '20,1', 'xd_bzh': 'xd_bzh_no',
}

DATASETS = {
    'BTC60': 'tests/test_data/BTC_USDT_60m_1000.parquet',
    'ETH60': 'tests/test_data/ETH_USDT_60m_1000.parquet',
    'BTC5m': 'tests/test_data/BTC_USDT_5m_1000.parquet',
    'ETH5m': 'tests/test_data/ETH_USDT_5m_1000.parquet',
    'BTCd':  'tests/test_data/BTC_USDT_d_500.parquet',
}


def reconstruct_presplit(bis):
    result = []
    i = 0
    while i < len(bis):
        bi = bis[i]
        if bi.is_split and i + 2 < len(bis):
            merged_end = bis[i + 2].end.k.k_index
            result.append((bi.type, bi.start.k.k_index, merged_end))
            i += 3
        else:
            result.append((bi.type, bi.start.k.k_index, bi.end.k.k_index))
            i += 1
    return result


def compare_presplit(open_presplit, pyarmor_presplit):
    n = min(len(open_presplit), len(pyarmor_presplit))
    diffs = 0
    for j in range(n):
        if open_presplit[j] != pyarmor_presplit[j]:
            diffs += 1
    diffs += abs(len(open_presplit) - len(pyarmor_presplit))
    return diffs


# Get pyarmor results (reference)
pyarmor_results = {}
for name, path in DATASETS.items():
    df = pd.read_parquet(path)
    cp = CL_P("test", "test", config=CL_CONFIG)
    cp.process_klines(df)
    pyarmor_results[name] = {
        'presplit': reconstruct_presplit(cp.get_bis()),
        'fxs_kidx': [(fx.type, fx.k.k_index) for fx in cp.get_fxs()],
    }

print("=" * 80)
print("EXPERIMENT: Testing different _bi_fx_valid configurations")
print("=" * 80)

# Test 1: Original (current code)
print("\n--- Test 1: Original (cl_gap < 4 + strict check) ---")
for name, path in DATASETS.items():
    df = pd.read_parquet(path)
    co = CL_O("test", "test", config=CL_CONFIG)
    co.process_klines(df)
    presplit = [(bi.type, bi.start.k.k_index, bi.end.k.k_index) for bi in co._build_bis(co.get_fxs())]
    diffs = compare_presplit(presplit, pyarmor_results[name]['presplit'])
    status = "✅" if diffs == 0 else f"❌ ({diffs} diffs)"
    print(f"  {name:10s}: open={len(presplit)} pyarmor={len(pyarmor_results[name]['presplit'])} {status}")


# Test 2: Patch _bi_fx_valid to skip strict check
print("\n--- Test 2: No strict check (cl_gap < 4 only) ---")
original_bi_fx_valid = CL_O._bi_fx_valid

def bi_fx_valid_no_strict(self, start_fx: FX, end_fx: FX) -> bool:
    if start_fx.type == end_fx.type:
        return False
    cl_gap = end_fx.k.index - start_fx.k.index
    if self.bi_type == Config.BI_TYPE_DD.value:
        if cl_gap < 1: return False
    elif self.bi_type == Config.BI_TYPE_JDB.value:
        k_gap = end_fx.k.k_index - start_fx.k.k_index
        if k_gap < 4: return False
    else:
        if cl_gap < 4: return False
    return True

CL_O._bi_fx_valid = bi_fx_valid_no_strict
for name, path in DATASETS.items():
    df = pd.read_parquet(path)
    co = CL_O("test", "test", config=CL_CONFIG)
    co.process_klines(df)
    presplit = [(bi.type, bi.start.k.k_index, bi.end.k.k_index) for bi in co._build_bis(co.get_fxs())]
    diffs = compare_presplit(presplit, pyarmor_results[name]['presplit'])
    status = "✅" if diffs == 0 else f"❌ ({diffs} diffs)"
    print(f"  {name:10s}: open={len(presplit)} pyarmor={len(pyarmor_results[name]['presplit'])} {status}")
CL_O._bi_fx_valid = original_bi_fx_valid


# Test 3: cl_gap < 5
print("\n--- Test 3: cl_gap < 5 (stricter gap) ---")
def bi_fx_valid_gap5(self, start_fx: FX, end_fx: FX) -> bool:
    if start_fx.type == end_fx.type:
        return False
    cl_gap = end_fx.k.index - start_fx.k.index
    k_gap = end_fx.k.k_index - start_fx.k.k_index
    if self.bi_type == Config.BI_TYPE_DD.value:
        if cl_gap < 1: return False
    elif self.bi_type == Config.BI_TYPE_JDB.value:
        if k_gap < 4: return False
    else:
        if cl_gap < 5: return False
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

CL_O._bi_fx_valid = bi_fx_valid_gap5
for name, path in DATASETS.items():
    df = pd.read_parquet(path)
    co = CL_O("test", "test", config=CL_CONFIG)
    co.process_klines(df)
    presplit = [(bi.type, bi.start.k.k_index, bi.end.k.k_index) for bi in co._build_bis(co.get_fxs())]
    diffs = compare_presplit(presplit, pyarmor_results[name]['presplit'])
    status = "✅" if diffs == 0 else f"❌ ({diffs} diffs)"
    print(f"  {name:10s}: open={len(presplit)} pyarmor={len(pyarmor_results[name]['presplit'])} {status}")
CL_O._bi_fx_valid = original_bi_fx_valid


# Test 4: k_gap < 5 instead of cl_gap < 4
print("\n--- Test 4: k_gap < 5 (use original K-line gap) ---")
def bi_fx_valid_kgap5(self, start_fx: FX, end_fx: FX) -> bool:
    if start_fx.type == end_fx.type:
        return False
    cl_gap = end_fx.k.index - start_fx.k.index
    k_gap = end_fx.k.k_index - start_fx.k.k_index
    if self.bi_type == Config.BI_TYPE_DD.value:
        if cl_gap < 1: return False
    elif self.bi_type == Config.BI_TYPE_JDB.value:
        if k_gap < 4: return False
    else:
        if k_gap < 5: return False
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

CL_O._bi_fx_valid = bi_fx_valid_kgap5
for name, path in DATASETS.items():
    df = pd.read_parquet(path)
    co = CL_O("test", "test", config=CL_CONFIG)
    co.process_klines(df)
    presplit = [(bi.type, bi.start.k.k_index, bi.end.k.k_index) for bi in co._build_bis(co.get_fxs())]
    diffs = compare_presplit(presplit, pyarmor_results[name]['presplit'])
    status = "✅" if diffs == 0 else f"❌ ({diffs} diffs)"
    print(f"  {name:10s}: open={len(presplit)} pyarmor={len(pyarmor_results[name]['presplit'])} {status}")
CL_O._bi_fx_valid = original_bi_fx_valid


# Test 5: cl_gap < 4 + strict + k_gap >= 5 extra check
print("\n--- Test 5: cl_gap < 4 AND k_gap < 5 (double gap check) ---")
def bi_fx_valid_double(self, start_fx: FX, end_fx: FX) -> bool:
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
        if k_gap < 5: return False
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

CL_O._bi_fx_valid = bi_fx_valid_double
for name, path in DATASETS.items():
    df = pd.read_parquet(path)
    co = CL_O("test", "test", config=CL_CONFIG)
    co.process_klines(df)
    presplit = [(bi.type, bi.start.k.k_index, bi.end.k.k_index) for bi in co._build_bis(co.get_fxs())]
    diffs = compare_presplit(presplit, pyarmor_results[name]['presplit'])
    status = "✅" if diffs == 0 else f"❌ ({diffs} diffs)"
    print(f"  {name:10s}: open={len(presplit)} pyarmor={len(pyarmor_results[name]['presplit'])} {status}")
CL_O._bi_fx_valid = original_bi_fx_valid
