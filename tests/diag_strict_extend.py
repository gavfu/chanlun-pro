"""
Experiment: Test strict inequality for extension in _build_bis.
Hypothesis: pyarmor uses strict < and > (no equal) for extending end_fx.

Tests the hypothesis against 5 small datasets.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

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
    first_diff = None
    for j in range(n):
        if open_presplit[j] != pyarmor_presplit[j]:
            diffs += 1
            if first_diff is None:
                first_diff = (j, open_presplit[j], pyarmor_presplit[j])
    diffs += abs(len(open_presplit) - len(pyarmor_presplit))
    return diffs, first_diff


# Get pyarmor results (reference)
pyarmor_results = {}
for name, path in DATASETS.items():
    df = pd.read_parquet(path)
    cp = CL_P("test", "test", config=CL_CONFIG)
    cp.process_klines(df)
    pyarmor_results[name] = reconstruct_presplit(cp.get_bis())

# Save original _build_bis method
original_build_bis = CL_O._build_bis

# Test 1: Original (baseline)
print("=== Test 1: Original (val <= / val >=) ===")
for name, path in DATASETS.items():
    df = pd.read_parquet(path)
    co = CL_O("test", "test", config=CL_CONFIG)
    co.process_klines(df)
    presplit = [(bi.type, bi.start.k.k_index, bi.end.k.k_index) for bi in co._build_bis(co.get_fxs())]
    d, fd = compare_presplit(presplit, pyarmor_results[name])
    status = "✅" if d == 0 else f"❌ ({d} diffs)"
    detail = ""
    if fd:
        detail = f" first: bi[{fd[0]}] open={fd[1][0]}({fd[1][1]}→{fd[1][2]}) pyarmor={fd[2][0]}({fd[2][1]}→{fd[2][2]})"
    print(f"  {name:10s}: {status}{detail}")

# Test 2: Strict inequality for extension
print("\n=== Test 2: Strict inequality (val < / val >) ===")

def build_bis_strict_extend(self, fxs):
    """Modified _build_bis with strict inequality for extension."""
    from chanlun.cl_interface import BI
    if len(fxs) < 2:
        return []
    bis = []
    start_fx = fxs[0]
    end_fx = None
    has_confirmed_bi = False
    
    for i in range(1, len(fxs)):
        cur_fx = fxs[i]
        
        if end_fx is None:
            # State 1: no candidate end_fx
            if cur_fx.type == start_fx.type:
                # Same direction - potential start update
                if not has_confirmed_bi:
                    if start_fx.type == "ding" and cur_fx.val >= start_fx.val:
                        start_fx = cur_fx
                    elif start_fx.type == "di" and cur_fx.val <= start_fx.val:
                        start_fx = cur_fx
            else:
                # Reverse direction - candidate
                if self._bi_fx_valid(start_fx, cur_fx):
                    end_fx = cur_fx
        else:
            # State 2: has candidate end_fx
            if cur_fx.type == end_fx.type:
                # Same as end_fx - try extend with STRICT inequality
                extend = False
                if end_fx.type == "di" and cur_fx.val < end_fx.val:  # STRICT <
                    if self._bi_fx_valid(start_fx, cur_fx):
                        extend = True
                elif end_fx.type == "ding" and cur_fx.val > end_fx.val:  # STRICT >
                    if self._bi_fx_valid(start_fx, cur_fx):
                        extend = True
                if extend:
                    end_fx = cur_fx
            else:
                # Reverse of end_fx - try confirmation
                confirm = self._bi_fx_valid(end_fx, cur_fx)
                
                if confirm and self.bi_fx_cgd != "bi_fx_cgd_yes":
                    # cgd check (skipped when bi_fx_cgd_yes)
                    pass
                
                if confirm:
                    # Confirm bi
                    bi_type = "up" if start_fx.type == "di" else "down"
                    bi = BI(
                        start=start_fx, end=end_fx, _type=bi_type,
                        index=len(bis),
                        default_zs_type=self.default_bi_zs_type,
                    )
                    bis.append(bi)
                    start_fx = end_fx
                    end_fx = None
                    has_confirmed_bi = True
                    # Recurse: the current fx becomes the new start_fx's first candidate
                    # Actually, need to process cur_fx as a candidate for the new bi
                    if self._bi_fx_valid(start_fx, cur_fx):
                        end_fx = cur_fx
    
    # Handle last unconfirmed bi
    if end_fx is not None:
        bi_type = "up" if start_fx.type == "di" else "down"
        bi = BI(
            start=start_fx, end=end_fx, _type=bi_type,
            index=len(bis),
            default_zs_type=self.default_bi_zs_type,
        )
        bis.append(bi)
    
    return bis

CL_O._build_bis = build_bis_strict_extend
for name, path in DATASETS.items():
    df = pd.read_parquet(path)
    co = CL_O("test", "test", config=CL_CONFIG)
    co.process_klines(df)
    presplit = [(bi.type, bi.start.k.k_index, bi.end.k.k_index) for bi in co._build_bis(co.get_fxs())]
    d, fd = compare_presplit(presplit, pyarmor_results[name])
    status = "✅" if d == 0 else f"❌ ({d} diffs)"
    detail = ""
    if fd:
        detail = f" first: bi[{fd[0]}] open={fd[1][0]}({fd[1][1]}→{fd[1][2]}) pyarmor={fd[2][0]}({fd[2][1]}→{fd[2][2]})"
    print(f"  {name:10s}: {status}{detail}")
CL_O._build_bis = original_build_bis
