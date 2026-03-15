"""
Trace _build_bis state machine for ETH5m at divergence point:
- cl_open bi: down k=333→349
- pyarmor bi: down k=333→337
Both accept (333→337) via _bi_fx_valid, so the difference is in confirmation logic.
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


def trace_state_machine(name, dataset, start_kidx, start_type, co, fxs):
    print(f"\n{'='*70}")
    print(f"{name}: trace from {start_type}(k={start_kidx})")
    
    qj, qy = co.fx_qj, co.fx_qy
    
    # Find start_fx
    start_fx_idx = None
    for idx, fx in enumerate(fxs):
        if fx.k.k_index == start_kidx and fx.type == start_type:
            start_fx_idx = idx
            break
    
    if start_fx_idx is None:
        print(f"  start FX not found!")
        return
    
    start_fx = fxs[start_fx_idx]
    end_fx = None
    end_idx = -1
    
    print(f"  start_fx: {start_fx.type} k={start_fx.k.k_index} val={start_fx.val:.2f} CLK={start_fx.k.index}")
    
    for i in range(start_fx_idx + 1, min(start_fx_idx + 25, len(fxs))):
        fx = fxs[i]
        
        if end_fx is None:
            if fx.type == start_fx.type:
                print(f"  [{i}] {fx.type:>4} k={fx.k.k_index:>4} val={fx.val:.2f} → same dir, skip (has_confirmed_bi=True)")
            else:
                valid = co._bi_fx_valid(start_fx, fx)
                cl_gap = fx.k.index - start_fx.k.index
                k_gap = fx.k.k_index - start_fx.k.k_index
                if valid:
                    end_fx = fx
                    end_idx = i
                    print(f"  [{i}] {fx.type:>4} k={fx.k.k_index:>4} val={fx.val:.2f} → valid={valid} cl_gap={cl_gap} k_gap={k_gap} → SET end_fx")
                else:
                    # Detail why invalid
                    detail = ""
                    if cl_gap < 4:
                        detail = f"cl_gap={cl_gap}<4"
                    elif k_gap < co.fx_check_k_nums and co.allow_bi_fx_strict:
                        if start_fx.type == "ding" and fx.type == "di":
                            s_l = start_fx.low(qj, qy)
                            e_l = fx.low(qj, qy)
                            e_h = fx.high(qj, qy)
                            s_h = start_fx.high(qj, qy)
                            if s_l < e_l: detail = f"s.low({s_l:.2f})<e.low({e_l:.2f})"
                            elif e_h > s_h: detail = f"e.high({e_h:.2f})>s.high({s_h:.2f})"
                        elif start_fx.type == "di" and fx.type == "ding":
                            s_h = start_fx.high(qj, qy)
                            e_h = fx.high(qj, qy)
                            e_l = fx.low(qj, qy)
                            s_l = start_fx.low(qj, qy)
                            if s_h > e_h: detail = f"s.high({s_h:.2f})>e.high({e_h:.2f})"
                            elif e_l < s_l: detail = f"e.low({e_l:.2f})<s.low({s_l:.2f})"
                    print(f"  [{i}] {fx.type:>4} k={fx.k.k_index:>4} val={fx.val:.2f} → valid={valid} cl_gap={cl_gap} k_gap={k_gap} {detail}")
        else:
            # Have end_fx candidate
            if fx.type == end_fx.type:
                # Same as end_fx - check extension
                extend = False
                if end_fx.type == "di" and fx.val <= end_fx.val:
                    if co._bi_fx_valid(start_fx, fx):
                        extend = True
                elif end_fx.type == "ding" and fx.val >= end_fx.val:
                    if co._bi_fx_valid(start_fx, fx):
                        extend = True
                if extend:
                    print(f"  [{i}] {fx.type:>4} k={fx.k.k_index:>4} val={fx.val:.2f} → EXTEND end_fx (better val)")
                    end_fx = fx
                    end_idx = i
                else:
                    print(f"  [{i}] {fx.type:>4} k={fx.k.k_index:>4} val={fx.val:.2f} → skip (not better or invalid)")
            else:
                # Reverse of end_fx - try confirmation
                confirm = co._bi_fx_valid(end_fx, fx)
                cl_gap = fx.k.index - end_fx.k.index
                k_gap = fx.k.k_index - end_fx.k.k_index
                detail = ""
                if not confirm:
                    if cl_gap < 4:
                        detail = f"cl_gap={cl_gap}<4"
                    elif k_gap < co.fx_check_k_nums and co.allow_bi_fx_strict:
                        if end_fx.type == "di" and fx.type == "ding":
                            s_h = end_fx.high(qj, qy)
                            e_h = fx.high(qj, qy)
                            e_l = fx.low(qj, qy)
                            s_l = end_fx.low(qj, qy)
                            if s_h > e_h: detail = f"e_fx.high({s_h:.2f})>c_fx.high({e_h:.2f})"
                            elif e_l < s_l: detail = f"c_fx.low({e_l:.2f})<e_fx.low({s_l:.2f})"
                        elif end_fx.type == "ding" and fx.type == "di":
                            s_l = end_fx.low(qj, qy)
                            e_l = fx.low(qj, qy)
                            e_h = fx.high(qj, qy)
                            s_h = end_fx.high(qj, qy)
                            if s_l < e_l: detail = f"e_fx.low({s_l:.2f})<c_fx.low({e_l:.2f})"
                            elif e_h > s_h: detail = f"c_fx.high({e_h:.2f})>e_fx.high({s_h:.2f})"
                if confirm:
                    print(f"  [{i}] {fx.type:>4} k={fx.k.k_index:>4} val={fx.val:.2f} → CONFIRM bi: {start_fx.k.k_index}→{end_fx.k.k_index} (cl_gap={cl_gap} k_gap={k_gap})")
                    print(f"  → BI confirmed: {start_fx.type}({start_fx.k.k_index}) → {end_fx.type}({end_fx.k.k_index})")
                    return end_fx.k.k_index
                else:
                    print(f"  [{i}] {fx.type:>4} k={fx.k.k_index:>4} val={fx.val:.2f} → confirm_FAIL cl_gap={cl_gap} k_gap={k_gap} {detail}")
    
    if end_fx:
        print(f"  → END (unconfirmed): {start_fx.type}({start_fx.k.k_index}) → {end_fx.type}({end_fx.k.k_index})")
    return None


df_eth5m = pd.read_parquet('tests/test_data/ETH_USDT_5m_1000.parquet')
co = CL_O("test", "test", config=CL_CONFIG)
co.process_klines(df_eth5m)

fxs = co.get_fxs()
trace_state_machine("ETH5m down bi from ding(333)", "ETH5m", 333, "ding", co, fxs)

# Also do BTC5m for comparison
df_btc5m = pd.read_parquet('tests/test_data/BTC_USDT_5m_1000.parquet')
co2 = CL_O("test", "test", config=CL_CONFIG)
co2.process_klines(df_btc5m)

fxs2 = co2.get_fxs()
trace_state_machine("BTC5m up bi from di(875)", "BTC5m", 875, "di", co2, fxs2)
