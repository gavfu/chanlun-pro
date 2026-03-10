"""Test theory: pyarmor confirmation may require confirm_fx.val to be more extreme
than start_fx.val in the BI direction.

For UP BI (di→ding): confirm_fx is di. Require confirm.val <= start.val (lower bottom).
For DOWN BI (ding→di): confirm_fx is ding. Require confirm.val >= start.val (higher top).

This ensures the confirmation creates a "deeper reversal" than the start."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

CL_CONFIG = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
}

DATASETS = [
    ("BTCd", "tests/test_data/BTC_USDT_d_500.parquet"),
    ("ETH60", "tests/test_data/ETH_USDT_60m_1000.parquet"),
    ("BTC60", "tests/test_data/BTC_USDT_60m_1000.parquet"),
    ("BTC5m", "tests/test_data/BTC_USDT_5m_1000.parquet"),
    ("ETH5m", "tests/test_data/ETH_USDT_5m_1000.parquet"),
]

for name, path in DATASETS:
    df = pd.read_parquet(path)
    cd_o = CL_O("test", "test", config=CL_CONFIG)
    cd_o.process_klines(df)
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    
    bis_o = cd_o.get_bis()
    bis_p = cd_p.get_bis()
    fxs = cd_o.get_fxs()
    
    # Check how many BIs would be affected if we add the confirm.val constraint
    # We need to replay _build_bis with the extra check
    # For simplicity, check each existing open BI's confirmation:
    # A confirmed BI means: the NEXT BI's start was the confirmation point
    # Actually, let's check: for each open BI, does the confirmation FX (= start of next next BI or next BI's end)
    # have val more extreme than start?
    
    # Easier: count and compare
    print(f"\n{'='*60}")
    print(f"=== {name}: Open={len(bis_o)} Pyarmor={len(bis_p)} ===")
    
    # Check each BI's confirmation point
    # When BI[i] is confirmed, the confirming FX becomes the start of the scan for the next end_fx
    # after BI[i]. The confirming FX is the first FX after end_fx that passes _bi_fx_valid.
    # 
    # Actually, we need to re-simulate to find confirmation points.
    # Let's re-implement _build_bis with the extra check and count BIs.
    
    # Standard count
    standard_bis = cd_o._build_bis(fxs)
    
    # Modified: add confirm.val check
    # For UP BI (di→ding): confirm is di. Require confirm.val <= start.val.
    # For DOWN BI (ding→di): confirm is ding. Require confirm.val >= start.val.
    
    modified_bis = []
    start_fx = fxs[0]
    start_idx = 0
    end_fx = None
    end_idx = -1
    has_confirmed = False
    
    i = 1
    while i < len(fxs):
        cur_fx = fxs[i]
        
        if end_fx is None:
            if cur_fx.type == start_fx.type:
                if not has_confirmed:
                    if start_fx.type == "ding" and cur_fx.val > start_fx.val:
                        start_fx = cur_fx
                        start_idx = i
                    elif start_fx.type == "di" and cur_fx.val < start_fx.val:
                        start_fx = cur_fx
                        start_idx = i
            else:
                if cd_o._bi_fx_valid(start_fx, cur_fx):
                    end_fx = cur_fx
                    end_idx = i
            i += 1
        else:
            if cur_fx.type == end_fx.type:
                if end_fx.type == "di" and cur_fx.val <= end_fx.val:
                    if cd_o._bi_fx_valid(start_fx, cur_fx):
                        end_fx = cur_fx
                        end_idx = i
                elif end_fx.type == "ding" and cur_fx.val >= end_fx.val:
                    if cd_o._bi_fx_valid(start_fx, cur_fx):
                        end_fx = cur_fx
                        end_idx = i
                i += 1
            else:
                confirm = cd_o._bi_fx_valid(end_fx, cur_fx)
                
                # EXTRA CHECK: confirm.val must be more extreme than start.val
                if confirm:
                    if start_fx.type == "di":
                        # UP BI: confirm is di, must be <= start.val
                        if cur_fx.val > start_fx.val:
                            confirm = False
                    else:
                        # DOWN BI: confirm is ding, must be >= start.val
                        if cur_fx.val < start_fx.val:
                            confirm = False
                
                if confirm:
                    from chanlun.cl_interface import BI
                    bi_type = "down" if start_fx.type == "ding" else "up"
                    bi = BI(start=start_fx, end=end_fx, _type=bi_type,
                            index=len(modified_bis), default_zs_type="zs_type_bz")
                    modified_bis.append(bi)
                    has_confirmed = True
                    start_fx = end_fx
                    start_idx = end_idx
                    end_fx = None
                    end_idx = -1
                    i = start_idx + 1
                else:
                    i += 1
    
    if end_fx is not None:
        from chanlun.cl_interface import BI
        bi_type = "down" if start_fx.type == "ding" else "up"
        bi = BI(start=start_fx, end=end_fx, _type=bi_type,
                index=len(modified_bis), default_zs_type="zs_type_bz")
        modified_bis.append(bi)
    
    print(f"  Standard: {len(standard_bis)} BIs")
    print(f"  Modified (confirm.val check): {len(modified_bis)} BIs")
    print(f"  Pyarmor:  {len(bis_p)} BIs")
    
    # Show boundary comparison
    if len(modified_bis) == len(bis_p):
        match = 0
        for bo, bp in zip(modified_bis, bis_p):
            if (bo.start.k.k_index == bp.start.k.k_index and 
                bo.end.k.k_index == bp.end.k.k_index):
                match += 1
        print(f"  Boundary match: {match}/{len(bis_p)}")
    
    # Show first few differences
    min_len = min(len(modified_bis), len(bis_p))
    diffs = 0
    for j in range(min_len):
        bo = modified_bis[j]
        bp = bis_p[j]
        if bo.start.k.k_index != bp.start.k.k_index or bo.end.k.k_index != bp.end.k.k_index:
            if diffs < 5:
                print(f"  DIFF bi[{j}]: mod={bo.type} {bo.start.k.k_index}→{bo.end.k.k_index} "
                      f"vs pya={bp.type} {bp.start.k.k_index}→{bp.end.k.k_index}")
            diffs += 1
    if diffs > 5:
        print(f"  ... and {diffs - 5} more diffs")
    print(f"  Total boundary diffs: {diffs}")
