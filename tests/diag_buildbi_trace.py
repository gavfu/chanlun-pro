"""
Investigate remaining _build_bis divergence for BTC5m and ETH5m.
BTC5m bi[59]: cl_open up 875→879, pyarmor up 875→889  
ETH5m bi[34]: cl_open down 468→484, pyarmor down 468→492
"""
import sys, os
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

CL_CONFIG = {
    'bi_type': 'bi_type_old', 'fx_qj': 'fx_qj_k', 'fx_qy': 'fx_qy_three',
    'bi_fx_cgd': 'bi_fx_cgd_yes', 'fx_check_k_nums': 13,
    'bi_split_k_cross_nums': '0', 'xd_bzh': 'xd_bzh_no',
}

from chanlun.cl_open import CL as CL_open
from chanlun.cl_pyarmor import CL as CL_pyarmor

qj, qy = 'fx_qj_k', 'fx_qy_three'

def trace_state_machine(dataset_name, data_path, bi_idx, context=5):
    """Trace the state machine around the divergent bi"""
    df = pd.read_parquet(data_path)
    co = CL_open("test", "test", config=CL_CONFIG)
    co.process_klines(df)
    cp = CL_pyarmor("test", "test", config=CL_CONFIG)
    cp.process_klines(df)

    # Get the divergent bi
    o_bi = co.bis[bi_idx]
    p_bi = cp.bis[bi_idx]
    
    print(f"\n{'='*80}")
    print(f"INVESTIGATING {dataset_name} bi[{bi_idx}]")
    print(f"{'='*80}")
    print(f"  cl_open:   {o_bi.type} {o_bi.start.type}(k={o_bi.start.k.index})→{o_bi.end.type}(k={o_bi.end.k.index})")
    print(f"  cl_pyarmor:{p_bi.type} {p_bi.start.type}(k={p_bi.start.k.index})→{p_bi.end.type}(k={p_bi.end.k.index})")
    
    # The start is the same, so the divergence is in which end_fx is chosen
    start_k = o_bi.start.k.index
    o_end_k = o_bi.end.k.index
    p_end_k = p_bi.end.k.index
    
    # Show FXs in the relevant region
    region_start = start_k - 2
    region_end = max(o_end_k, p_end_k) + 10
    
    print(f"\n  FXs from k={region_start} to k={region_end}:")
    fxs_region = [fx for fx in co.fxs if region_start <= fx.k.index <= region_end]
    
    for fx in fxs_region:
        marker = ""
        if fx.k.index == start_k:
            marker = " ← BI START"
        elif fx.k.index == o_end_k:
            marker += " ← cl_open END"
        elif fx.k.index == p_end_k:
            marker += " ← pyarmor END"
        
        print(f"    {fx.type}(k={fx.k.index}, k_idx={fx.k.k_index}) "
              f"val={fx.val:.2f} high={fx.high(qj,qy):.2f} low={fx.low(qj,qy):.2f}"
              f"{marker}")
    
    # Find start_fx in fxs
    start_fx = None
    for fx in co.fxs:
        if fx.k.index == start_k:
            start_fx = fx
            break
    
    # Check _bi_fx_valid for the FXs between start and both endpoints
    print(f"\n  Validity checks from start({start_fx.type} k={start_k}):")
    for fx in fxs_region:
        if fx.k.index <= start_k:
            continue
        if fx.type == start_fx.type:
            continue  # Same type, can't form bi
        
        cl_gap = fx.k.index - start_fx.k.index
        k_gap = fx.k.k_index - start_fx.k.k_index
        valid = co._bi_fx_valid(start_fx, fx)
        
        # Detailed strict check
        strict_detail = ""
        if k_gap < 13:
            if start_fx.type == "ding" and fx.type == "di":
                if start_fx.low(qj, qy) < fx.low(qj, qy):
                    strict_detail = f"STRICT FAIL: start.low({start_fx.low(qj,qy):.2f}) < end.low({fx.low(qj,qy):.2f})"
                elif fx.high(qj, qy) > start_fx.high(qj, qy):
                    strict_detail = f"STRICT FAIL: end.high({fx.high(qj,qy):.2f}) > start.high({start_fx.high(qj,qy):.2f})"
                else:
                    strict_detail = "STRICT PASS"
            elif start_fx.type == "di" and fx.type == "ding":
                if start_fx.high(qj, qy) > fx.high(qj, qy):
                    strict_detail = f"STRICT FAIL: start.high({start_fx.high(qj,qy):.2f}) > end.high({fx.high(qj,qy):.2f})"
                elif fx.low(qj, qy) < start_fx.low(qj, qy):
                    strict_detail = f"STRICT FAIL: end.low({fx.low(qj,qy):.2f}) < start.low({start_fx.low(qj,qy):.2f})"
                else:
                    strict_detail = "STRICT PASS"
        else:
            strict_detail = f"STRICT SKIPPED (k_gap={k_gap} >= 13)"
        
        marker = ""
        if fx.k.index == o_end_k:
            marker = " ← cl_open picks this"
        elif fx.k.index == p_end_k:
            marker = " ← pyarmor picks this"
        
        print(f"    → {fx.type}(k={fx.k.index}) cl_gap={cl_gap} k_gap={k_gap} "
              f"valid={valid} {strict_detail}{marker}")
    
    # Now trace what the state machine does
    print(f"\n  State machine trace:")
    fxs = co.fxs
    start_idx = None
    for idx, fx in enumerate(fxs):
        if fx.k.index == start_k:
            start_idx = idx
            break
    
    if start_idx is None:
        print("    ERROR: start_fx not found in fxs!")
        return
    
    end_fx = None
    end_idx = -1
    
    for i in range(start_idx + 1, min(start_idx + 30, len(fxs))):
        cur_fx = fxs[i]
        
        if end_fx is None:
            if cur_fx.type != start_fx.type:
                valid = co._bi_fx_valid(start_fx, cur_fx)
                print(f"    [State1] i={i} {cur_fx.type}(k={cur_fx.k.index}) val={cur_fx.val:.2f} "
                      f"→ _bi_fx_valid={valid}", end="")
                if valid:
                    end_fx = cur_fx
                    end_idx = i
                    print(f" → SET end_fx")
                else:
                    print(f" → skip")
            else:
                print(f"    [State1] i={i} {cur_fx.type}(k={cur_fx.k.index}) val={cur_fx.val:.2f} "
                      f"→ same type, skip")
        else:
            if cur_fx.type == end_fx.type:
                # Same as end_fx → check extension
                if end_fx.type == "di" and cur_fx.val < end_fx.val:
                    valid = co._bi_fx_valid(start_fx, cur_fx)
                    print(f"    [State2-extend] i={i} {cur_fx.type}(k={cur_fx.k.index}) "
                          f"val={cur_fx.val:.2f} < end_fx.val={end_fx.val:.2f} "
                          f"→ _bi_fx_valid={valid}", end="")
                    if valid:
                        end_fx = cur_fx
                        end_idx = i
                        print(f" → EXTEND end_fx")
                    else:
                        print(f" → skip (invalid)")
                elif end_fx.type == "ding" and cur_fx.val > end_fx.val:
                    valid = co._bi_fx_valid(start_fx, cur_fx)
                    print(f"    [State2-extend] i={i} {cur_fx.type}(k={cur_fx.k.index}) "
                          f"val={cur_fx.val:.2f} > end_fx.val={end_fx.val:.2f} "
                          f"→ _bi_fx_valid={valid}", end="")
                    if valid:
                        end_fx = cur_fx
                        end_idx = i
                        print(f" → EXTEND end_fx")
                    else:
                        print(f" → skip (invalid)")
                else:
                    print(f"    [State2-same] i={i} {cur_fx.type}(k={cur_fx.k.index}) "
                          f"val={cur_fx.val:.2f} not better than end_fx.val={end_fx.val:.2f} → ignore")
            else:
                # Opposite type → check confirmation
                valid = co._bi_fx_valid(end_fx, cur_fx)
                print(f"    [State2-confirm] i={i} {cur_fx.type}(k={cur_fx.k.index}) "
                      f"val={cur_fx.val:.2f} → _bi_fx_valid(end_fx,cur)={valid}", end="")
                if valid:
                    print(f" → CONFIRM bi {start_fx.type}(k={start_fx.k.index})→"
                          f"{end_fx.type}(k={end_fx.k.index})")
                    # Check if this is the divergent bi
                    if end_fx.k.index == o_end_k:
                        print(f"    *** cl_open correctly ends bi here (k={o_end_k})")
                    elif end_fx.k.index == p_end_k:
                        print(f"    *** This matches pyarmor endpoint (k={p_end_k})")
                    break
                else:
                    print(f" → not confirmed, continue")
        
        if i - start_idx > 25:
            print("    ... (truncated)")
            break
    
    # Now show what happens in pyarmor's bi sequence around this area
    print(f"\n  Pyarmor bis around divergence:")
    for bi in cp.bis:
        if bi.start.k.index >= start_k - 5 and bi.end.k.index <= max(o_end_k, p_end_k) + 15:
            marker = ""
            if bi.index == bi_idx:
                marker = " ← DIVERGENT"
            print(f"    bi[{bi.index}]: {bi.type} k={bi.start.k.index}→{bi.end.k.index}{marker}")

# Run BTC5m
tdir = os.path.join(os.path.dirname(__file__), 'test_data')
trace_state_machine("BTC5m", os.path.join(tdir, 'BTC_USDT_5m_1000.parquet'), 59)

# Run ETH5m  
trace_state_machine("ETH5m", os.path.join(tdir, 'ETH_USDT_5m_1000.parquet'), 34)
