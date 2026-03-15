"""
Diagnostic: Test hypothesis that pyarmor uses _bi_fx_valid (not just _split_gap_ok)
for split1 candidate validation.

Hypothesis: _find_split1_from_triplet should iterate candidates by POSITION (first),
selecting the first one where _bi_fx_valid(bi.start, candidate) returns True.

If bi[8]'s di(192) fails _bi_fx_valid strict check, algorithm skips to di(199).
If bi[16]'s di(370) passes _bi_fx_valid, it's selected first by position.
"""
import sys, os
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from chanlun.cl_interface import Config

CL_CONFIG = {
    'bi_type': 'bi_type_old', 'fx_qj': 'fx_qj_k', 'fx_qy': 'fx_qy_three',
    'bi_fx_cgd': 'bi_fx_cgd_yes', 'fx_check_k_nums': 13,
    'bi_split_k_cross_nums': '20,1', 'xd_bzh': 'xd_bzh_no',
}
CL_CONFIG_NOSPLIT = {
    'bi_type': 'bi_type_old', 'fx_qj': 'fx_qj_k', 'fx_qy': 'fx_qy_three',
    'bi_fx_cgd': 'bi_fx_cgd_yes', 'fx_check_k_nums': 13,
    'bi_split_k_cross_nums': '0', 'xd_bzh': 'xd_bzh_no',
}

from chanlun.cl_open import CL as CL_open
from chanlun.cl_pyarmor import CL as CL_pyarmor

# Load BTC60 data
data_path = os.path.join(os.path.dirname(__file__), 'test_data', 'BTC_USDT_60m_1000.parquet')
klines = pd.read_parquet(data_path)

cl_o = CL_open("test", "test", config=CL_CONFIG)
cl_o.process_klines(klines)
cl_p = CL_pyarmor("test", "test", config=CL_CONFIG)
cl_p.process_klines(klines)
cl_nosplit = CL_open("test", "test", config=CL_CONFIG_NOSPLIT)
cl_nosplit.process_klines(klines)

# Find which pre-split bis differ after splitting
print("=" * 80)
print("SPLIT CASES IN BTC60")
print("=" * 80)

# Compare pyarmor bis to find split bis
pyarmor_split_indices = set()
for bi in cl_p.bis:
    if hasattr(bi, 'is_split') and bi.is_split:
        pyarmor_split_indices.add(bi.index)

# Find pre-split bis that get split by pyarmor
# Look for consecutive pyarmor bis that map to a single pre-split bi
print("\nPyarmor split bis:", [i for i in sorted(pyarmor_split_indices)])

# Find all pre-split bis that contain overlap >= threshold
print("\n" + "=" * 80)
print("TESTING _bi_fx_valid HYPOTHESIS FOR ALL TRIGGERED SPLITS")
print("=" * 80)

qj = Config.FX_QJ_K.value
qy = Config.FX_QY_THREE.value
threshold = 20
tolerance = 1

for pre_bi in cl_nosplit.bis:
    start_idx = pre_bi.start.k.index
    end_idx = pre_bi.end.k.index

    # Get internal fxs
    internal_fxs = [fx for fx in cl_nosplit.fxs
                    if start_idx < fx.k.index < end_idx]
    if len(internal_fxs) < 3:
        continue

    # Check for triggered triplet
    triggered_ti = -1
    hit_count_triggered = 0
    end_ki = pre_bi.end.k.k_index

    for ti in range(len(internal_fxs) - 2):
        fx1 = internal_fxs[ti]
        fx2 = internal_fxs[ti + 1]
        fx3 = internal_fxs[ti + 2]

        h1, l1 = fx1.high(qj, qy), fx1.low(qj, qy)
        h2, l2 = fx2.high(qj, qy), fx2.low(qj, qy)
        h3, l3 = fx3.high(qj, qy), fx3.low(qj, qy)

        hit_count = 0
        miss_count = 0
        for ki in range(fx1.k.k_index, end_ki):
            k = cl_nosplit.src_klines[ki]
            if (k.h >= l1 and k.l <= h1
                    and k.h >= l2 and k.l <= h2
                    and k.h >= l3 and k.l <= h3):
                hit_count += 1
                miss_count = 0
            else:
                miss_count += 1
            if miss_count > tolerance:
                break

        if hit_count >= threshold:
            triggered_ti = ti
            hit_count_triggered = hit_count
            break

    if triggered_ti < 0:
        continue

    triplet = (internal_fxs[triggered_ti],
               internal_fxs[triggered_ti + 1],
               internal_fxs[triggered_ti + 2])

    print(f"\n{'='*60}")
    print(f"Pre-split BI[{pre_bi.index}]: {pre_bi.type} "
          f"k={pre_bi.start.k.index}→{pre_bi.end.k.index} "
          f"(k_idx={pre_bi.start.k.k_index}→{pre_bi.end.k.k_index})")
    print(f"  Start: {pre_bi.start.type}(k={pre_bi.start.k.index}) val={pre_bi.start.val}")
    print(f"  End:   {pre_bi.end.type}(k={pre_bi.end.k.index}) val={pre_bi.end.val}")
    print(f"  Hit count: {hit_count_triggered} at triplet t[{triggered_ti}]")

    # Show triplet
    print(f"\n  Triplet: ({triplet[0].type} k={triplet[0].k.index}, "
          f"{triplet[1].type} k={triplet[1].k.index}, "
          f"{triplet[2].type} k={triplet[2].k.index})")
    for fx in triplet:
        print(f"    {fx.type}(k={fx.k.index}, k_idx={fx.k.k_index}) "
              f"val={fx.val:.2f} high={fx.high(qj,qy):.2f} low={fx.low(qj,qy):.2f}")

    # Determine split1 type
    split1_type = "di" if pre_bi.type == "down" else "ding"
    split2_type = "ding" if pre_bi.type == "down" else "di"

    # Get split1 candidates from triplet
    candidates = [fx for fx in triplet if fx.type == split1_type]

    print(f"\n  Split1 candidates ({split1_type}) from triplet:")
    for fx in candidates:
        cl_gap = fx.k.index - pre_bi.start.k.index
        k_gap = fx.k.k_index - pre_bi.start.k.k_index

        # Check _split_gap_ok
        gap_ok = k_gap >= 4
        
        # Check full _bi_fx_valid
        bi_fx_valid = cl_nosplit._bi_fx_valid(pre_bi.start, fx)
        
        # Also show strict check details
        strict_result = "N/A"
        if k_gap < 13:  # fx_check_k_nums
            if pre_bi.start.type == "ding" and fx.type == "di":
                check1 = pre_bi.start.low(qj, qy) < fx.low(qj, qy)
                check2 = fx.high(qj, qy) > pre_bi.start.high(qj, qy)
                if check1:
                    strict_result = f"FAIL: start.low({pre_bi.start.low(qj,qy):.2f}) < end.low({fx.low(qj,qy):.2f})"
                elif check2:
                    strict_result = f"FAIL: end.high({fx.high(qj,qy):.2f}) > start.high({pre_bi.start.high(qj,qy):.2f})"
                else:
                    strict_result = f"PASS (start.low={pre_bi.start.low(qj,qy):.2f} >= end.low={fx.low(qj,qy):.2f}, end.high={fx.high(qj,qy):.2f} <= start.high={pre_bi.start.high(qj,qy):.2f})"
            elif pre_bi.start.type == "di" and fx.type == "ding":
                check1 = pre_bi.start.high(qj, qy) > fx.high(qj, qy)
                check2 = fx.low(qj, qy) < pre_bi.start.low(qj, qy)
                if check1:
                    strict_result = f"FAIL: start.high({pre_bi.start.high(qj,qy):.2f}) > end.high({fx.high(qj,qy):.2f})"
                elif check2:
                    strict_result = f"FAIL: end.low({fx.low(qj,qy):.2f}) < start.low({pre_bi.start.low(qj,qy):.2f})"
                else:
                    strict_result = f"PASS"
        else:
            strict_result = f"SKIP (k_gap={k_gap} >= 13)"

        print(f"    {fx.type}(k={fx.k.index}, k_idx={fx.k.k_index}) val={fx.val:.2f}")
        print(f"      cl_gap={cl_gap}, k_gap={k_gap}")
        print(f"      _split_gap_ok: {gap_ok}")
        print(f"      _bi_fx_valid:  {bi_fx_valid}")
        print(f"      strict check:  {strict_result}")

    # What cl_open currently picks (most extreme val)
    if split1_type == "di":
        sorted_by_val = sorted(candidates, key=lambda f: f.val)
    else:
        sorted_by_val = sorted(candidates, key=lambda f: f.val, reverse=True)
    cl_open_pick = None
    for fx in sorted_by_val:
        if cl_nosplit._bi_fx_valid(pre_bi.start, fx) or (fx.k.k_index - pre_bi.start.k.k_index) >= 4:
            cl_open_pick = fx
            break
    if cl_open_pick is None and sorted_by_val:
        cl_open_pick = sorted_by_val[0]

    # What "first by position with _bi_fx_valid" would pick
    sorted_by_pos = sorted(candidates, key=lambda f: f.k.index)
    hypothesis_pick = None
    for fx in sorted_by_pos:
        if cl_nosplit._bi_fx_valid(pre_bi.start, fx):
            hypothesis_pick = fx
            break
    # Fallback: first with gap_ok
    if hypothesis_pick is None:
        for fx in sorted_by_pos:
            if (fx.k.k_index - pre_bi.start.k.k_index) >= 4:
                hypothesis_pick = fx
                break

    # What pyarmor actually has
    # Find pyarmor bi that starts at pre_bi.start
    pyarmor_split1 = None
    for pbi in cl_p.bis:
        if pbi.start.k.index == pre_bi.start.k.index and pbi.end.k.index != pre_bi.end.k.index:
            pyarmor_split1 = pbi.end
            break

    print(f"\n  SELECTION COMPARISON:")
    print(f"    Current cl_open (extreme val):       {split1_type}(k={cl_open_pick.k.index}) val={cl_open_pick.val:.2f}" if cl_open_pick else "    Current cl_open: None")
    print(f"    Hypothesis (pos + _bi_fx_valid):     {split1_type}(k={hypothesis_pick.k.index}) val={hypothesis_pick.val:.2f}" if hypothesis_pick else "    Hypothesis: None")
    if pyarmor_split1:
        print(f"    Pyarmor actual split1:               {pyarmor_split1.type}(k={pyarmor_split1.k.index}) val={pyarmor_split1.val:.2f}")
    else:
        print(f"    Pyarmor actual split1:               (not found / no split)")

    match_current = cl_open_pick and pyarmor_split1 and cl_open_pick.k.index == pyarmor_split1.k.index
    match_hypothesis = hypothesis_pick and pyarmor_split1 and hypothesis_pick.k.index == pyarmor_split1.k.index
    print(f"\n    Current approach matches pyarmor:  {'✅' if match_current else '❌'}")
    print(f"    Hypothesis matches pyarmor:        {'✅' if match_hypothesis else '❌'}")

    # Also show what pyarmor's full split looks like
    print(f"\n  Pyarmor split result for this bi region:")
    for pbi in cl_p.bis:
        if pbi.start.k.index >= pre_bi.start.k.index and pbi.end.k.index <= pre_bi.end.k.index:
            print(f"    bi[{pbi.index}]: {pbi.type} {pbi.start.type}(k={pbi.start.k.index})→{pbi.end.type}(k={pbi.end.k.index})")

    print(f"\n  cl_open split result for this bi region:")
    for obi in cl_o.bis:
        if obi.start.k.index >= pre_bi.start.k.index and obi.end.k.index <= pre_bi.end.k.index:
            print(f"    bi[{obi.index}]: {obi.type} {obi.start.type}(k={obi.start.k.index})→{obi.end.type}(k={obi.end.k.index})")

print("\n" + "=" * 80)
print("Also checking ETH60 for additional data points...")
print("=" * 80)

# Load ETH60 data
eth_data_path = os.path.join(os.path.dirname(__file__), 'test_data', 'ETH_USDT_60m_1000.parquet')
eth_klines = pd.read_parquet(eth_data_path)

cl_o_eth = CL_open("test", "test", config=CL_CONFIG)
cl_o_eth.process_klines(eth_klines)
cl_p_eth = CL_pyarmor("test", "test", config=CL_CONFIG)
cl_p_eth.process_klines(eth_klines)
cl_nosplit_eth = CL_open("test", "test", config=CL_CONFIG_NOSPLIT)
cl_nosplit_eth.process_klines(eth_klines)

for pre_bi in cl_nosplit_eth.bis:
    start_idx = pre_bi.start.k.index
    end_idx = pre_bi.end.k.index

    internal_fxs = [fx for fx in cl_nosplit_eth.fxs
                    if start_idx < fx.k.index < end_idx]
    if len(internal_fxs) < 3:
        continue

    triggered_ti = -1
    hit_count_triggered = 0
    end_ki = pre_bi.end.k.k_index

    for ti in range(len(internal_fxs) - 2):
        fx1 = internal_fxs[ti]
        fx2 = internal_fxs[ti + 1]
        fx3 = internal_fxs[ti + 2]

        h1, l1 = fx1.high(qj, qy), fx1.low(qj, qy)
        h2, l2 = fx2.high(qj, qy), fx2.low(qj, qy)
        h3, l3 = fx3.high(qj, qy), fx3.low(qj, qy)

        hit_count = 0
        miss_count = 0
        for ki in range(fx1.k.k_index, end_ki):
            k = cl_nosplit_eth.src_klines[ki]
            if (k.h >= l1 and k.l <= h1
                    and k.h >= l2 and k.l <= h2
                    and k.h >= l3 and k.l <= h3):
                hit_count += 1
                miss_count = 0
            else:
                miss_count += 1
            if miss_count > tolerance:
                break

        if hit_count >= threshold:
            triggered_ti = ti
            hit_count_triggered = hit_count
            break

    if triggered_ti < 0:
        continue

    triplet = (internal_fxs[triggered_ti],
               internal_fxs[triggered_ti + 1],
               internal_fxs[triggered_ti + 2])

    split1_type = "di" if pre_bi.type == "down" else "ding"

    candidates = [fx for fx in triplet if fx.type == split1_type]
    if not candidates:
        continue

    print(f"\n{'='*60}")
    print(f"ETH60 Pre-split BI[{pre_bi.index}]: {pre_bi.type} "
          f"k={pre_bi.start.k.index}→{pre_bi.end.k.index}")
    print(f"  Hit count: {hit_count_triggered} at triplet t[{triggered_ti}]")
    print(f"  Triplet: ({triplet[0].type} k={triplet[0].k.index}, "
          f"{triplet[1].type} k={triplet[1].k.index}, "
          f"{triplet[2].type} k={triplet[2].k.index})")

    for fx in candidates:
        cl_gap = fx.k.index - pre_bi.start.k.index
        k_gap = fx.k.k_index - pre_bi.start.k.k_index
        gap_ok = k_gap >= 4
        bi_fx_valid = cl_nosplit_eth._bi_fx_valid(pre_bi.start, fx)

        strict_result = "N/A"
        if k_gap < 13:
            if pre_bi.start.type == "ding" and fx.type == "di":
                check1 = pre_bi.start.low(qj, qy) < fx.low(qj, qy)
                check2 = fx.high(qj, qy) > pre_bi.start.high(qj, qy)
                if check1:
                    strict_result = f"FAIL: start.low < end.low"
                elif check2:
                    strict_result = f"FAIL: end.high > start.high"
                else:
                    strict_result = "PASS"
            elif pre_bi.start.type == "di" and fx.type == "ding":
                check1 = pre_bi.start.high(qj, qy) > fx.high(qj, qy)
                check2 = fx.low(qj, qy) < pre_bi.start.low(qj, qy)
                if check1:
                    strict_result = f"FAIL: start.high > end.high"
                elif check2:
                    strict_result = f"FAIL: end.low < start.low"
                else:
                    strict_result = "PASS"
        else:
            strict_result = f"SKIP (k_gap={k_gap} >= 13)"

        print(f"    {fx.type}(k={fx.k.index}, k_idx={fx.k.k_index}) val={fx.val:.2f} "
              f"cl_gap={cl_gap} k_gap={k_gap} gap_ok={gap_ok} "
              f"_bi_fx_valid={bi_fx_valid} strict={strict_result}")

    # picks
    sorted_by_pos = sorted(candidates, key=lambda f: f.k.index)
    hypothesis_pick = None
    for fx in sorted_by_pos:
        if cl_nosplit_eth._bi_fx_valid(pre_bi.start, fx):
            hypothesis_pick = fx
            break
    if hypothesis_pick is None:
        for fx in sorted_by_pos:
            if (fx.k.k_index - pre_bi.start.k.k_index) >= 4:
                hypothesis_pick = fx
                break

    pyarmor_split1 = None
    for pbi in cl_p_eth.bis:
        if pbi.start.k.index == pre_bi.start.k.index and pbi.end.k.index != pre_bi.end.k.index:
            pyarmor_split1 = pbi.end
            break

    print(f"  Hypothesis pick: {split1_type}(k={hypothesis_pick.k.index})" if hypothesis_pick else "  Hypothesis: None")
    if pyarmor_split1:
        print(f"  Pyarmor split1:  {pyarmor_split1.type}(k={pyarmor_split1.k.index})")
    match = hypothesis_pick and pyarmor_split1 and hypothesis_pick.k.index == pyarmor_split1.k.index
    print(f"  Hypothesis matches: {'✅' if match else '❌'}")
