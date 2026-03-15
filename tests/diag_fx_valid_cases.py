"""
诊断两个 FX-valid 分歧案例:
  BTC5m: cl_open accepts (875→879), pyarmor rejects → cl_open too lenient
  ETH5m: pyarmor accepts (333→337), cl_open rejects → cl_open too strict
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


def analyze_fx_pair(name, start_kidx, end_kidx, data_path, start_type, end_type):
    df = pd.read_parquet(data_path)
    co = CL_O("test", "test", config=CL_CONFIG)
    co.process_klines(df)

    fxs = co.get_fxs()
    qj, qy = co.fx_qj, co.fx_qy

    # Find the FXs
    start_fx = None
    end_fx = None
    for fx in fxs:
        if fx.k.k_index == start_kidx and fx.type == start_type:
            start_fx = fx
        if fx.k.k_index == end_kidx and fx.type == end_type:
            end_fx = fx

    if start_fx is None or end_fx is None:
        print(f"{name}: start or end FX not found!")
        return

    cl_gap = end_fx.k.index - start_fx.k.index
    k_gap = end_fx.k.k_index - start_fx.k.k_index
    valid = co._bi_fx_valid(start_fx, end_fx)

    print(f"\n{'='*60}")
    print(f"{name}: {start_type}(k={start_kidx}) → {end_type}(k={end_kidx})")
    print(f"  cl_gap = {cl_gap}, k_gap = {k_gap}")
    print(f"  _bi_fx_valid = {valid}")

    print(f"\n  start FX ({start_type} k={start_kidx}):")
    print(f"    val = {start_fx.val:.2f}")
    print(f"    high({qj},{qy}) = {start_fx.high(qj, qy):.2f}")
    print(f"    low({qj},{qy})  = {start_fx.low(qj, qy):.2f}")
    print(f"    CLK index = {start_fx.k.index}")
    print(f"    klines: ", end="")
    for kl in start_fx.klines:
        if kl:
            print(f"[k_idx={kl.k_index} h={kl.h:.2f} l={kl.l:.2f}] ", end="")
    print()

    print(f"\n  end FX ({end_type} k={end_kidx}):")
    print(f"    val = {end_fx.val:.2f}")
    print(f"    high({qj},{qy}) = {end_fx.high(qj, qy):.2f}")
    print(f"    low({qj},{qy})  = {end_fx.low(qj, qy):.2f}")
    print(f"    CLK index = {end_fx.k.index}")
    print(f"    klines: ", end="")
    for kl in end_fx.klines:
        if kl:
            print(f"[k_idx={kl.k_index} h={kl.h:.2f} l={kl.l:.2f}] ", end="")
    print()

    # Detailed strict check
    if k_gap < co.fx_check_k_nums and co.allow_bi_fx_strict:
        print(f"\n  STRICT CHECK (k_gap={k_gap} < {co.fx_check_k_nums}):")
        if start_fx.type == "di" and end_fx.type == "ding":
            s_h = start_fx.high(qj, qy)
            e_h = end_fx.high(qj, qy)
            e_l = end_fx.low(qj, qy)
            s_l = start_fx.low(qj, qy)
            print(f"    上升笔:")
            print(f"    check1: start.high({s_h:.2f}) > end.high({e_h:.2f}) → {s_h > e_h}")
            print(f"    check2: end.low({e_l:.2f}) < start.low({s_l:.2f}) → {e_l < s_l}")
            # 区间包含检查 (potential missing check)
            print(f"    包含？ start包含end: {s_l <= e_l and s_h >= e_h}")
            print(f"    包含？ end包含start: {e_l <= s_l and e_h >= s_h}")
            # 区间重叠
            overlap = min(s_h, e_h) - max(s_l, e_l)
            print(f"    区间重叠: {overlap:.2f} (start=[{s_l:.2f},{s_h:.2f}] end=[{e_l:.2f},{e_h:.2f}])")
        elif start_fx.type == "ding" and end_fx.type == "di":
            s_h = start_fx.high(qj, qy)
            e_h = end_fx.high(qj, qy)
            e_l = end_fx.low(qj, qy)
            s_l = start_fx.low(qj, qy)
            print(f"    下降笔:")
            print(f"    check1: start.low({s_l:.2f}) < end.low({e_l:.2f}) → {s_l < e_l}")
            print(f"    check2: end.high({e_h:.2f}) > start.high({s_h:.2f}) → {e_h > s_h}")
            # 区间包含检查
            print(f"    包含？ start包含end: {s_l <= e_l and s_h >= e_h}")
            print(f"    包含？ end包含start: {e_l <= s_l and e_h >= s_h}")
            overlap = min(s_h, e_h) - max(s_l, e_l)
            print(f"    区间重叠: {overlap:.2f} (start=[{s_l:.2f},{s_h:.2f}] end=[{e_l:.2f},{e_h:.2f}])")
    else:
        print(f"\n  NO STRICT CHECK (k_gap={k_gap} >= {co.fx_check_k_nums})")

    # Check: do klines of start and end FX share any CLKlines?
    start_kl_indices = set()
    for kl in start_fx.klines:
        if kl:
            start_kl_indices.add(kl.k_index)
    end_kl_indices = set()
    for kl in end_fx.klines:
        if kl:
            end_kl_indices.add(kl.k_index)
    shared = start_kl_indices & end_kl_indices
    print(f"\n  共享缠论K线: {shared if shared else '无'}")
    print(f"  start kline indices: {sorted(start_kl_indices)}")
    print(f"  end kline indices: {sorted(end_kl_indices)}")


# Case 1: BTC5m - cl_open accepts (875→879), pyarmor rejects
# up bi: di(875) → ding(879)
analyze_fx_pair("BTC5m (cl_open too lenient)",
                875, 879, 'tests/test_data/BTC_USDT_5m_1000.parquet',
                "di", "ding")

# Case 2: ETH5m - pyarmor accepts (333→337), cl_open rejects  
# down bi: ding(333) → di(337)
# Actually I need to verify the types - let me check
analyze_fx_pair("ETH5m (cl_open too strict?)",
                333, 337, 'tests/test_data/ETH_USDT_5m_1000.parquet',
                "ding", "di")

# Also check the next valid FX that pyarmor uses for BTC5m
# pyarmor bi = 875→889, so let me check (875, 889)
analyze_fx_pair("BTC5m pyarmor endpoint",
                875, 889, 'tests/test_data/BTC_USDT_5m_1000.parquet',
                "di", "ding")
