"""Check BTC60 remaining split divergence at bi[8-10].
open: down 187→192, up 192→197, down 197→215
pyarmor: down 187→199, up 199→204, down 204→215
Both split down 187→215 differently."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O

CL_CONFIG = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
}

df = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cd = CL_O("test", "test", config=CL_CONFIG)
cd.bi_split_k_cross_nums = 0
cd.process_klines(df)
bis_raw = cd.get_bis()
fxs = cd.get_fxs()

# Find the raw BI down 187→215
for j, b in enumerate(bis_raw):
    if b.start.k.k_index == 187 and b.end.k.k_index == 215:
        bi = b
        print(f"raw bi[{j}]: {bi.type} {bi.start.k.k_index}→{bi.end.k.k_index}")
        start_idx = bi.start.k.index
        end_idx = bi.end.k.index
        
        internal = [fx for fx in fxs if start_idx < fx.k.index < end_idx]
        print(f"\nInternal FXes ({len(internal)}):")
        for fx in internal:
            print(f"  {fx.type}@{fx.k.k_index} (cl_idx={fx.k.index}) val={fx.val:.2f}")
        
        # Check triplet hit counts
        qj = "fx_qj_k"; qy = "fx_qy_three"
        triggered_ti = -1
        for ti in range(len(internal) - 2):
            fx1, fx2, fx3 = internal[ti], internal[ti+1], internal[ti+2]
            h1, l1 = fx1.high(qj, qy), fx1.low(qj, qy)
            h2, l2 = fx2.high(qj, qy), fx2.low(qj, qy)
            h3, l3 = fx3.high(qj, qy), fx3.low(qj, qy)
            hit, miss = 0, 0
            for ki in range(fx1.k.k_index, bi.end.k.k_index):
                k = cd.src_klines[ki]
                if (k.h>=l1 and k.l<=h1 and k.h>=l2 and k.l<=h2 and k.h>=l3 and k.l<=h3):
                    hit += 1; miss = 0
                else:
                    miss += 1
                if miss > 1: break
            mark = "*** TRIGGER ***" if hit >= 20 else ""
            print(f"  triplet[{ti}]: {fx1.type}@{fx1.k.k_index},{fx2.type}@{fx2.k.k_index},{fx3.type}@{fx3.k.k_index} → hit={hit} {mark}")
            if hit >= 20 and triggered_ti < 0:
                triggered_ti = ti
        
        if triggered_ti >= 0:
            trip = (internal[triggered_ti], internal[triggered_ti+1], internal[triggered_ti+2])
            print(f"\nTriggered at triplet[{triggered_ti}]: {trip[0].type}@{trip[0].k.k_index}, "
                  f"{trip[1].type}@{trip[1].k.k_index}, {trip[2].type}@{trip[2].k.k_index}")
            
            # DOWN: split1_type = "ding", but wait: DOWN bi split selects di first.
            # Actually: _select_split_down: split1=di, split2=ding
            print(f"\nDOWN BI split: start(ding)→di(split1)→ding(split2)→end(di)")
            
            # Triplet-only di candidates
            cands = [fx for fx in trip if fx.type == "di"]
            print(f"\nTriplet-only di candidates: {[(f.k.k_index, f.val) for f in cands]}")
            for f in cands:
                k = f.k.k_index - bi.start.k.k_index
                cl = f.k.index - bi.start.k.index
                print(f"  di@{f.k.k_index}: cl_gap={cl} k_gap={k}")
            
            # With new k_gap check
            gap_ok_cands = [f for f in cands if (f.k.k_index - bi.start.k.k_index) >= 4]
            if gap_ok_cands:
                split1 = gap_ok_cands[0]
            else:
                split1 = cands[-1]
            print(f"\n  → split1 = di@{split1.k.k_index}")
            
            # split2: ding after split1, val > split1.val
            cands2 = [fx for fx in internal if fx.type == "ding" and fx.k.index > split1.k.index
                      and fx.val > split1.val]
            print(f"\n  ding after split1: {[(f.k.k_index, f.val) for f in cands2]}")
            gap_ok2 = [f for f in cands2 if (f.k.k_index - split1.k.k_index) >= 4]
            if gap_ok2:
                split2 = max(gap_ok2, key=lambda f: f.val)
            elif cands2:
                split2 = cands2[0]
            else:
                split2 = None
            print(f"  → split2 = {'ding@' + str(split2.k.k_index) if split2 else 'None'}")
            
            if split2:
                print(f"\n  Open produces: {bi.start.k.k_index}→{split1.k.k_index}→{split2.k.k_index}→{bi.end.k.k_index}")
                print(f"  Pyarmor:       {bi.start.k.k_index}→199→204→{bi.end.k.k_index}")
        break
