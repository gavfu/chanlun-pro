"""Compare the FX structures of di@142 (ETH60) and di@337 (ETH5m) to understand
what makes one case legitimate for strict check and the other not."""
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

# ETH60 di@142
print("=== ETH60 di@142 ===")
df = pd.read_parquet("tests/test_data/ETH_USDT_60m_1000.parquet")
cd = CL_O("test", "test", config=CL_CONFIG)
cd.process_klines(df)
fxs = cd.get_fxs()
for fx in fxs:
    if fx.k.k_index == 142 and fx.type == "di":
        for i, ck in enumerate(fx.klines):
            if ck is not None:
                print(f"  klines[{i}]: ck_idx={ck.index} k_index={ck.k_index} h={ck.h:.2f} l={ck.l:.2f}")
                for sk in ck.klines:
                    print(f"    raw: idx={sk.index} h={sk.h:.2f} l={sk.l:.2f}")
        # Also show the FX before this
        print(f"  FX val={fx.val:.2f}")
        break

# What FX is before di@142?
print("\nFXes around 138-148:")
for fx in fxs:
    if 138 <= fx.k.k_index <= 148:
        print(f"  {fx.type}@{fx.k.k_index} val={fx.val:.2f} ck={fx.k.index}")

# ETH5m di@337
print("\n=== ETH5m di@337 ===")
df = pd.read_parquet("tests/test_data/ETH_USDT_5m_1000.parquet")
cd2 = CL_O("test", "test", config=CL_CONFIG)
cd2.process_klines(df)
fxs2 = cd2.get_fxs()
for fx in fxs2:
    if fx.k.k_index == 337 and fx.type == "di":
        for i, ck in enumerate(fx.klines):
            if ck is not None:
                print(f"  klines[{i}]: ck_idx={ck.index} k_index={ck.k_index} h={ck.h:.2f} l={ck.l:.2f}")
                for sk in ck.klines:
                    print(f"    raw: idx={sk.index} h={sk.h:.2f} l={sk.l:.2f}")
        print(f"  FX val={fx.val:.2f}")
        break

print("\nFXes around 333-345:")
for fx in fxs2:
    if 333 <= fx.k.k_index <= 345:
        print(f"  {fx.type}@{fx.k.k_index} val={fx.val:.2f} ck={fx.k.index}")

# Key question: what's special about di@337 klines[0] vs di@142 klines[0]?
# In di@142: klines[0] has MUCH higher h than center (2494 vs 2432)
# In di@337: klines[0] has higher h than center (2076 vs 2069)
# Both have klines[0] with higher h. But the PRECEDING FX context might differ.

# In ETH60: the FX before di@142 should be ding@???
# In ETH5m: the FX before di@337 is ding@333

# Let me check: is klines[0] of di SHARED with the previous ding FX?
print("\n=== Checking FX overlap ===")
# ETH60: di@142
for fx in fxs:
    if fx.k.k_index == 142 and fx.type == "di":
        di_left_ck = fx.klines[0]
        print(f"ETH60 di@142 klines[0] ck_idx={di_left_ck.index}")
        # Check if the previous ding has this as its klines[2]
        for pfx in fxs:
            if pfx.type == "ding" and pfx.k.k_index < 142:
                last_ding = pfx
        print(f"  Previous ding: ding@{last_ding.k.k_index} ck={last_ding.k.index}")
        if last_ding.klines[2] is not None:
            print(f"  ding klines[2] ck_idx={last_ding.klines[2].index}")
            shared = last_ding.klines[2].index == di_left_ck.index
            print(f"  Shared with di klines[0]? {shared}")
        break

# ETH5m: di@337
for fx in fxs2:
    if fx.k.k_index == 337 and fx.type == "di":
        di_left_ck = fx.klines[0]
        print(f"\nETH5m di@337 klines[0] ck_idx={di_left_ck.index}")
        for pfx in fxs2:
            if pfx.type == "ding" and pfx.k.k_index < 337:
                last_ding = pfx
        print(f"  Previous ding: ding@{last_ding.k.k_index} ck={last_ding.k.index}")
        if last_ding.klines[2] is not None:
            print(f"  ding klines[2] ck_idx={last_ding.klines[2].index}")
            shared = last_ding.klines[2].index == di_left_ck.index
            print(f"  Shared with di klines[0]? {shared}")
        break
