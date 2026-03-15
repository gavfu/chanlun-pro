"""
Test hypothesis: cl_gap >= 5 (instead of >= 4) in _bi_fx_valid.

Check whether changing the threshold fixes BTC5m/ETH5m without breaking BTC60/ETH60/BTCd.
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

tdir = os.path.join(os.path.dirname(__file__), 'test_data')
datasets = [
    ('BTC60',  'BTC_USDT_60m_1000.parquet'),
    ('ETH60',  'ETH_USDT_60m_1000.parquet'),
    ('BTC5m',  'BTC_USDT_5m_1000.parquet'),
    ('ETH5m',  'ETH_USDT_5m_1000.parquet'),
    ('BTCd',   'BTC_USDT_d_500.parquet'),
]

# First: check what happens with current code (cl_gap >= 4)
print("=" * 80)
print("CURRENT CODE (cl_gap >= 4) - Pre-split comparison")
print("=" * 80)
for name, fname in datasets:
    df = pd.read_parquet(os.path.join(tdir, fname))
    co = CL_open("test", "test", config=CL_CONFIG)
    co.process_klines(df)
    cp = CL_pyarmor("test", "test", config=CL_CONFIG)
    cp.process_klines(df)
    
    diffs = 0
    for idx in range(min(len(co.bis), len(cp.bis))):
        ob, pb = co.bis[idx], cp.bis[idx]
        if ob.start.k.index != pb.start.k.index or ob.end.k.index != pb.end.k.index:
            diffs += 1
            if diffs <= 2:
                print(f"  {name} bi[{idx}]: cl_open {ob.start.k.index}→{ob.end.k.index} "
                      f"pyarmor {pb.start.k.index}→{pb.end.k.index}")
            break
    cnt_diff = abs(len(co.bis) - len(cp.bis))
    if diffs == 0 and cnt_diff == 0:
        print(f"  {name}: ✅ PERFECT ({len(co.bis)} bis)")
    else:
        print(f"  {name}: ❌ first diff at bi[{idx}], counts {len(co.bis)}/{len(cp.bis)}")

# Now: test with cl_gap >= 5
print("\n" + "=" * 80)
print("TEST: cl_gap >= 5 in _bi_fx_valid")
print("=" * 80)

# Monkey-patch _bi_fx_valid to use cl_gap >= 5
from chanlun.cl_interface import Config, FX

original_bi_fx_valid = CL_open._bi_fx_valid

def patched_bi_fx_valid(self, start_fx: FX, end_fx: FX) -> bool:
    if start_fx.type == end_fx.type:
        return False
    cl_gap = end_fx.k.index - start_fx.k.index
    k_gap = end_fx.k.k_index - start_fx.k.k_index
    if self.bi_type == Config.BI_TYPE_DD.value:
        if cl_gap < 1:
            return False
    elif self.bi_type == Config.BI_TYPE_JDB.value:
        if k_gap < 4:
            return False
    else:
        # CHANGED: cl_gap < 5 instead of cl_gap < 4
        if cl_gap < 5:
            return False
    if k_gap < self.fx_check_k_nums:
        if self.allow_bi_fx_strict:
            qj = self.fx_qj
            qy = self.fx_qy
            if start_fx.type == "ding" and end_fx.type == "di":
                if start_fx.low(qj, qy) < end_fx.low(qj, qy):
                    return False
                if end_fx.high(qj, qy) > start_fx.high(qj, qy):
                    return False
            elif start_fx.type == "di" and end_fx.type == "ding":
                if start_fx.high(qj, qy) > end_fx.high(qj, qy):
                    return False
                if end_fx.low(qj, qy) < start_fx.low(qj, qy):
                    return False
    return True

CL_open._bi_fx_valid = patched_bi_fx_valid

for name, fname in datasets:
    df = pd.read_parquet(os.path.join(tdir, fname))
    co = CL_open("test", "test", config=CL_CONFIG)
    co.process_klines(df)
    cp = CL_pyarmor("test", "test", config=CL_CONFIG)
    cp.process_klines(df)
    
    diffs = 0
    first_diff = -1
    for idx in range(min(len(co.bis), len(cp.bis))):
        ob, pb = co.bis[idx], cp.bis[idx]
        if ob.start.k.index != pb.start.k.index or ob.end.k.index != pb.end.k.index:
            diffs += 1
            if first_diff < 0:
                first_diff = idx
                print(f"  {name} bi[{idx}]: cl_open {ob.start.k.index}→{ob.end.k.index} "
                      f"pyarmor {pb.start.k.index}→{pb.end.k.index}")
    cnt_diff = abs(len(co.bis) - len(cp.bis))
    if diffs == 0 and cnt_diff == 0:
        print(f"  {name}: ✅ PERFECT ({len(co.bis)} bis)")
    else:
        print(f"  {name}: ❌ {diffs} diffs from bi[{first_diff}], counts {len(co.bis)}/{len(cp.bis)}")

# Restore original
CL_open._bi_fx_valid = original_bi_fx_valid

# Also: check what initial end_fx cl_gaps are in BTC60 to understand impact
print("\n" + "=" * 80)
print("BTC60 initial end_fx cl_gaps (with current code)")
print("=" * 80)
df = pd.read_parquet(os.path.join(tdir, 'BTC_USDT_60m_1000.parquet'))
co = CL_open("test", "test", config=CL_CONFIG)
co.process_klines(df)

# Trace _build_bis to find initial end_fx cl_gaps
min_cl_gap_initial = 999
min_cl_gap_confirm = 999
fxs = co.fxs
start_fx = fxs[0]
end_fx = None
has_confirmed = False
i = 1
while i < len(fxs):
    cur_fx = fxs[i]
    if end_fx is None:
        if cur_fx.type != start_fx.type:
            if co._bi_fx_valid(start_fx, cur_fx):
                cl_gap = cur_fx.k.index - start_fx.k.index
                min_cl_gap_initial = min(min_cl_gap_initial, cl_gap)
                end_fx = cur_fx
                end_idx = i
        else:
            if not has_confirmed:
                if start_fx.type == "ding" and cur_fx.val > start_fx.val:
                    start_fx = cur_fx
                elif start_fx.type == "di" and cur_fx.val < start_fx.val:
                    start_fx = cur_fx
        i += 1
    else:
        if cur_fx.type == end_fx.type:
            if (end_fx.type == "di" and cur_fx.val < end_fx.val) or \
               (end_fx.type == "ding" and cur_fx.val > end_fx.val):
                if co._bi_fx_valid(start_fx, cur_fx):
                    end_fx = cur_fx
                    end_idx = i
            i += 1
        else:
            confirm = co._bi_fx_valid(end_fx, cur_fx)
            if confirm:
                cl_gap_c = cur_fx.k.index - end_fx.k.index
                min_cl_gap_confirm = min(min_cl_gap_confirm, cl_gap_c)
                has_confirmed = True
                start_fx = end_fx
                end_fx = None
                i = end_idx + 1
            else:
                i += 1

print(f"  Min cl_gap for initial end_fx setting: {min_cl_gap_initial}")
print(f"  Min cl_gap for confirmation: {min_cl_gap_confirm}")

# Do the same for ETH60
df = pd.read_parquet(os.path.join(tdir, 'ETH_USDT_60m_1000.parquet'))
co = CL_open("test", "test", config=CL_CONFIG)
co.process_klines(df)

min_cl_gap_initial = 999
min_cl_gap_confirm = 999
fxs = co.fxs
start_fx = fxs[0]
end_fx = None
has_confirmed = False
end_idx = -1
i = 1
while i < len(fxs):
    cur_fx = fxs[i]
    if end_fx is None:
        if cur_fx.type != start_fx.type:
            if co._bi_fx_valid(start_fx, cur_fx):
                cl_gap = cur_fx.k.index - start_fx.k.index
                min_cl_gap_initial = min(min_cl_gap_initial, cl_gap)
                end_fx = cur_fx
                end_idx = i
        else:
            if not has_confirmed:
                if start_fx.type == "ding" and cur_fx.val > start_fx.val:
                    start_fx = cur_fx
                elif start_fx.type == "di" and cur_fx.val < start_fx.val:
                    start_fx = cur_fx
        i += 1
    else:
        if cur_fx.type == end_fx.type:
            if (end_fx.type == "di" and cur_fx.val < end_fx.val) or \
               (end_fx.type == "ding" and cur_fx.val > end_fx.val):
                if co._bi_fx_valid(start_fx, cur_fx):
                    end_fx = cur_fx
                    end_idx = i
            i += 1
        else:
            confirm = co._bi_fx_valid(end_fx, cur_fx)
            if confirm:
                cl_gap_c = cur_fx.k.index - end_fx.k.index
                min_cl_gap_confirm = min(min_cl_gap_confirm, cl_gap_c)
                has_confirmed = True
                start_fx = end_fx
                end_fx = None
                i = end_idx + 1
            else:
                i += 1

print(f"\n  ETH60 Min cl_gap for initial: {min_cl_gap_initial}")
print(f"  ETH60 Min cl_gap for confirmation: {min_cl_gap_confirm}")

# BTCd
df = pd.read_parquet(os.path.join(tdir, 'BTC_USDT_d_1000.parquet'))
co = CL_open("test", "test", config=CL_CONFIG)
co.process_klines(df)

min_cl_gap_initial = 999
min_cl_gap_confirm = 999
fxs = co.fxs
start_fx = fxs[0]
end_fx = None
has_confirmed = False
end_idx = -1
i = 1
while i < len(fxs):
    cur_fx = fxs[i]
    if end_fx is None:
        if cur_fx.type != start_fx.type:
            if co._bi_fx_valid(start_fx, cur_fx):
                cl_gap = cur_fx.k.index - start_fx.k.index
                min_cl_gap_initial = min(min_cl_gap_initial, cl_gap)
                end_fx = cur_fx
                end_idx = i
        else:
            if not has_confirmed:
                if start_fx.type == "ding" and cur_fx.val > start_fx.val:
                    start_fx = cur_fx
                elif start_fx.type == "di" and cur_fx.val < start_fx.val:
                    start_fx = cur_fx
        i += 1
    else:
        if cur_fx.type == end_fx.type:
            if (end_fx.type == "di" and cur_fx.val < end_fx.val) or \
               (end_fx.type == "ding" and cur_fx.val > end_fx.val):
                if co._bi_fx_valid(start_fx, cur_fx):
                    end_fx = cur_fx
                    end_idx = i
            i += 1
        else:
            confirm = co._bi_fx_valid(end_fx, cur_fx)
            if confirm:
                cl_gap_c = cur_fx.k.index - end_fx.k.index
                min_cl_gap_confirm = min(min_cl_gap_confirm, cl_gap_c)
                has_confirmed = True
                start_fx = end_fx
                end_fx = None
                i = end_idx + 1
            else:
                i += 1

print(f"\n  BTCd Min cl_gap for initial: {min_cl_gap_initial}")
print(f"  BTCd Min cl_gap for confirmation: {min_cl_gap_confirm}")
