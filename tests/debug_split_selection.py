"""Debug split point selection for both divergent BIs.
raw[13]: up 299→330 — open picks 308/315, pyarmor picks 304/309
raw[48]: down 911→992 — open picks 918/926, pyarmor picks 957/967"""
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

df = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")

# Run both without split to get raw BIs + fxs
cd_o = CL_O("test", "test", config=CL_CONFIG)
cd_o.bi_split_k_cross_nums = 0
cd_o.process_klines(df)
bis_raw = cd_o.get_bis()
fxs = cd_o.get_fxs()

print("=" * 70)
print("CASE 1: raw bi[13] = up 299→330 (di→ding)")
print("        open:   299→308, 308→315, 315→330")
print("        pyarmor: 299→304, 304→309, 309→330")
print("=" * 70)

bi = bis_raw[13]
start_idx = bi.start.k.index
end_idx = bi.end.k.index

internal_fxs = [fx for fx in fxs if start_idx < fx.k.index < end_idx]
print(f"\nInternal FXes ({len(internal_fxs)}):")
for fx in internal_fxs:
    print(f"  {fx.type}@{fx.k.k_index} (cl_idx={fx.k.index}) val={fx.val:.2f}")

# For UP bi: split1=ding, split2=di  (start(di)→ding→di→end(ding))
# Triggered triplet: triplet[0] = (ding@304, di@306, ding@308)
triplet = (internal_fxs[0], internal_fxs[1], internal_fxs[2])
print(f"\nTriggered triplet: {triplet[0].type}@{triplet[0].k.k_index}, "
      f"{triplet[1].type}@{triplet[1].k.k_index}, "
      f"{triplet[2].type}@{triplet[2].k.k_index}")

# _find_split1_from_triplet for UP: split1_type = "ding"
print(f"\n--- Split1 selection (type=ding) ---")
# Candidates = ding FXes from triplet
candidates = [fx for fx in triplet if fx.type == "ding"]
# Also FXes before triplet with correct type
prev_idx = triplet[0].k.index
for fx in fxs:
    if fx.k.index < prev_idx and fx.k.index > start_idx and fx.type == "ding":
        candidates.append(fx)
candidates.sort(key=lambda f: f.k.index)
print(f"Candidates: {[(f.type, f.k.k_index, f.k.index) for f in candidates]}")

# Check _split_gap_ok for each
for fx in candidates:
    cl = fx.k.index - bi.start.k.index
    k = fx.k.k_index - bi.start.k.k_index
    gap_ok = cl >= 4  # bi_type_old: cl_gap >= 4
    print(f"  {fx.type}@{fx.k.k_index}: cl_gap={cl}, k_gap={k}, gap_ok={gap_ok}")

# Result: first with gap_ok=True, or last if none
gap_ok_list = [fx for fx in candidates if (fx.k.index - bi.start.k.index) >= 4]
if gap_ok_list:
    split1 = gap_ok_list[0]
    print(f"  → Selected: {split1.type}@{split1.k.k_index} (first gap_ok)")
else:
    split1 = candidates[-1]
    print(f"  → Selected: {split1.type}@{split1.k.k_index} (last candidate)")

# _find_split2 for UP: split2_type = "di" (after split1)
print(f"\n--- Split2 selection (type=di, after ding@{split1.k.k_index}) ---")
candidates2 = [fx for fx in internal_fxs
               if fx.type == "di" and fx.k.index > split1.k.index]
print(f"All di after split1: {[(f.type, f.k.k_index, f.val) for f in candidates2]}")

# Valid: split1.val > di.val (ding > di)
valid = [fx for fx in candidates2 if split1.val > fx.val]
print(f"Valid (split1.val={split1.val:.2f} > di.val): "
      f"{[(f.k.k_index, f.val) for f in valid]}")

# Gap-ok (cl_gap >= 4 from split1)
gap_ok = [fx for fx in valid if (fx.k.index - split1.k.index) >= 4]
print(f"Gap-ok (cl_gap>=4): {[(f.k.k_index, f.val) for f in gap_ok]}")

if gap_ok:
    split2 = min(gap_ok, key=lambda f: f.val)
    print(f"  → Selected: di@{split2.k.k_index} (min val among gap_ok)")
else:
    split2 = valid[0]
    print(f"  → Selected: di@{split2.k.k_index} (first valid)")

print(f"\n*** Open's split: {bi.start.k.k_index}→{split1.k.k_index}→{split2.k.k_index}→{bi.end.k.k_index}")
print(f"*** Pyarmor:      {bi.start.k.k_index}→304→309→{bi.end.k.k_index}")

# For pyarmor result (299→304→309→330):
# split1=ding@304 needs cl_gap=1 from start → gap_ok = False!
# So pyarmor either: (a) uses k_gap check instead of cl_gap, or (b) different split selection
print(f"\n--- If pyarmor uses k_gap for split gap check ---")
for fx in candidates:
    cl = fx.k.index - bi.start.k.index
    k = fx.k.k_index - bi.start.k.k_index
    k_gap_ok = k >= 4  # k_gap check
    print(f"  {fx.type}@{fx.k.k_index}: cl_gap={cl}, k_gap={k}, k_gap_ok={k_gap_ok}")

# With k_gap: ding@304 has k_gap=5 → True → selected first
# Then split2 after ding@304: di candidates...
candidates2_k = [fx for fx in internal_fxs
                 if fx.type == "di" and fx.k.index > 216]  # ding@304 is cl_idx=216
valid_k = [fx for fx in candidates2_k if 70000.0 > fx.val]  # ding@304.val=70000
print(f"\nWith split1=ding@304:")
print(f"  di candidates: {[(f.k.k_index, f.val) for f in valid_k]}")
gap_ok_k = [fx for fx in valid_k if (fx.k.k_index - 304) >= 4]
print(f"  k_gap>=4 from 304: {[(f.k.k_index, f.val, f.k.k_index-304) for f in gap_ok_k]}")
if gap_ok_k:
    split2_k = min(gap_ok_k, key=lambda f: f.val)
    print(f"  → di@{split2_k.k.k_index} val={split2_k.val:.2f}")

print(f"\n\n{'='*70}")
print(f"CASE 2: raw bi[48] = down 911→992 (ding→di)")
print(f"        open:   911→918, 918→926, 926→992")
print(f"        pyarmor: 911→957, 957→967, 967→992")
print(f"{'='*70}")

bi2 = bis_raw[48]
start_idx2 = bi2.start.k.index
end_idx2 = bi2.end.k.index

internal_fxs2 = [fx for fx in fxs if start_idx2 < fx.k.index < end_idx2]
print(f"\nInternal FXes ({len(internal_fxs2)}):")
for fx in internal_fxs2:
    print(f"  {fx.type}@{fx.k.k_index} (cl_idx={fx.k.index}) val={fx.val:.2f}")

# Check all triplets for hit counts
print(f"\nTriplet hit counts:")
qj = "fx_qj_k"; qy = "fx_qy_three"
for ti in range(len(internal_fxs2) - 2):
    fx1 = internal_fxs2[ti]
    fx2 = internal_fxs2[ti + 1]
    fx3 = internal_fxs2[ti + 2]
    
    h1, l1 = fx1.high(qj, qy), fx1.low(qj, qy)
    h2, l2 = fx2.high(qj, qy), fx2.low(qj, qy)
    h3, l3 = fx3.high(qj, qy), fx3.low(qj, qy)
    
    hit_count = 0
    miss_count = 0
    for ki in range(fx1.k.k_index, bi2.end.k.k_index):
        k = cd_o.src_klines[ki]
        if (k.h >= l1 and k.l <= h1
                and k.h >= l2 and k.l <= h2
                and k.h >= l3 and k.l <= h3):
            hit_count += 1
            miss_count = 0
        else:
            miss_count += 1
        if miss_count > 1:
            break
    
    print(f"  [{ti}]: {fx1.type}@{fx1.k.k_index}, {fx2.type}@{fx2.k.k_index}, "
          f"{fx3.type}@{fx3.k.k_index} → hit={hit_count} "
          f"{'*** TRIGGER ***' if hit_count >= 20 else ''}")

# For DOWN bi: split1=di, split2=ding (start(ding)→di→ding→end(di))
# Find triggered triplet
triggered_ti = -1
for ti in range(len(internal_fxs2) - 2):
    fx1 = internal_fxs2[ti]
    fx2 = internal_fxs2[ti + 1]
    fx3 = internal_fxs2[ti + 2]
    h1, l1 = fx1.high(qj, qy), fx1.low(qj, qy)
    h2, l2 = fx2.high(qj, qy), fx2.low(qj, qy)
    h3, l3 = fx3.high(qj, qy), fx3.low(qj, qy)
    hit_count = 0
    miss_count = 0
    for ki in range(fx1.k.k_index, bi2.end.k.k_index):
        k = cd_o.src_klines[ki]
        if (k.h >= l1 and k.l <= h1
                and k.h >= l2 and k.l <= h2
                and k.h >= l3 and k.l <= h3):
            hit_count += 1
            miss_count = 0
        else:
            miss_count += 1
        if miss_count > 1:
            break
    if hit_count >= 20:
        triggered_ti = ti
        break

triplet2 = (internal_fxs2[triggered_ti], internal_fxs2[triggered_ti + 1], 
            internal_fxs2[triggered_ti + 2])
print(f"\nTriggered triplet: {triplet2[0].type}@{triplet2[0].k.k_index}, "
      f"{triplet2[1].type}@{triplet2[1].k.k_index}, "
      f"{triplet2[2].type}@{triplet2[2].k.k_index}")

# For DOWN: split1_type = "di"
print(f"\n--- Split1 selection (type=di) ---")
cands = [fx for fx in triplet2 if fx.type == "di"]
prev_idx2 = triplet2[0].k.index
for fx in fxs:
    if fx.k.index < prev_idx2 and fx.k.index > start_idx2 and fx.type == "di":
        cands.append(fx)
cands.sort(key=lambda f: f.k.index)
print(f"Candidates: {[(f.type, f.k.k_index, f.k.index) for f in cands]}")

for fx in cands:
    cl = fx.k.index - bi2.start.k.index
    k = fx.k.k_index - bi2.start.k.k_index
    cl_gap_ok = cl >= 4
    k_gap_ok = k >= 4
    print(f"  {fx.type}@{fx.k.k_index}: cl_gap={cl} cl_ok={cl_gap_ok}, k_gap={k} k_ok={k_gap_ok}")
