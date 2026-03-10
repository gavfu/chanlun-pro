"""
Trace calls to pyarmor's _xd_cal_line_xlfx specifically filtering for 
calls related to the down[28] segment (lines starting around bi[29]).
Also look at the pattern: when does pyarmor call with bh?
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_pyarmor import CL as CL_Pyarmor

config = {
    "bi_type": "bi_type_old", "fx_qj": "fx_qj_k", "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes", "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1", "xd_bzh": "xd_bzh_no",
    "xd_zs_max_lines_split": 11, "xd_allow_split_no_highlow": 1,
    "xd_allow_split_zs_kz": 0, "xd_allow_split_zs_more_line": 1,
    "xd_allow_split_zs_no_direction": 1,
}

original_xlfx = CL_Pyarmor._xd_cal_line_xlfx

call_log = []

def traced_xlfx(self, lines, fx_type='ding', bh_type='no_bh', *args, **kwargs):
    result = original_xlfx(self, lines, fx_type, bh_type, *args, **kwargs)
    lines_idx = [l.index for l in lines]
    n_xlfx = 0
    fx_info = []
    if result is not None:
        n_xlfx = len(result[1])
        for fx in result[1]:
            fx_lines = ",".join(str(l.index) for l in fx.xl.lines)
            fx_info.append(f"bi[{fx_lines}]bad={fx.is_line_bad}")
    
    call_log.append({
        'lines': lines_idx,
        'fx_type': fx_type,
        'bh_type': bh_type,
        'n_lines': len(lines),
        'n_xlfx': n_xlfx,
        'fx_info': fx_info,
    })
    return result

CL_Pyarmor._xd_cal_line_xlfx = traced_xlfx

# Run BTC60
df60 = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
cl60 = CL_Pyarmor("test", "60m", config)
cl60.process_klines(df60)

print(f"Total calls: {len(call_log)}\n")

# 1. Show call pattern: when does pyarmor call with bh vs no_bh?
print("=== Call pattern (bh vs no_bh) ===")
nobh_count = sum(1 for c in call_log if c['bh_type'] == 'no_bh')
bh_count = sum(1 for c in call_log if c['bh_type'] == 'bh')
print(f"  no_bh calls: {nobh_count}")
print(f"  bh calls: {bh_count}")

# 2. Look for consecutive call pairs (no_bh then bh with same params)
print("\n=== Consecutive no_bh/bh pairs ===")
pairs = []
for i in range(len(call_log) - 1):
    c1, c2 = call_log[i], call_log[i+1]
    if (c1['bh_type'] == 'no_bh' and c2['bh_type'] == 'bh' 
        and c1['lines'] == c2['lines'] and c1['fx_type'] == c2['fx_type']):
        pairs.append((i, c1, c2))

print(f"  Found {len(pairs)} consecutive no_bh→bh pairs")
for idx, c1, c2 in pairs[:30]:
    first = c1['lines'][0] if c1['lines'] else '?'
    last = c1['lines'][-1] if c1['lines'] else '?'
    nobh_fx = "; ".join(c1['fx_info']) if c1['fx_info'] else "none"
    bh_fx = "; ".join(c2['fx_info']) if c2['fx_info'] else "none"
    print(f"  [{idx}] {c1['fx_type']} n={c1['n_lines']:2d} lines[{first}..{last}]  "
          f"no_bh:{nobh_fx}  bh:{bh_fx}")

# 3. Filter for calls that include lines starting near bi[29] (down from bi[28])
print("\n=== Calls involving down[28] lines (starting around bi[29]) ===")
for i, c in enumerate(call_log):
    if c['lines'] and c['lines'][0] in [29, 30, 31] and c['fx_type'] == 'di':
        first = c['lines'][0]
        last = c['lines'][-1]
        fx_str = "; ".join(c['fx_info']) if c['fx_info'] else "none"
        print(f"  [{i:4d}] {c['bh_type']:5s} {c['fx_type']:4s} n={c['n_lines']:2d} "
              f"lines[{first}..{last}] → {c['n_xlfx']} FXs: {fx_str}")

# 4. Filter for calls that include lines starting near bi[40] (up from bi[39])
print("\n=== Calls involving up[39] lines (starting around bi[40]) ===")
for i, c in enumerate(call_log):
    if c['lines'] and c['lines'][0] in [40, 41, 42] and c['fx_type'] == 'ding':
        first = c['lines'][0]
        last = c['lines'][-1]
        fx_str = "; ".join(c['fx_info']) if c['fx_info'] else "none"
        print(f"  [{i:4d}] {c['bh_type']:5s} {c['fx_type']:4s} n={c['n_lines']:2d} "
              f"lines[{first}..{last}] → {c['n_xlfx']} FXs: {fx_str}")

# 5. For the pairs, look at what happens when no_bh finds a bad FX
print("\n=== Pairs where no_bh has bad FX ===")
for idx, c1, c2 in pairs:
    has_bad = any('bad=True' in fi for fi in c1['fx_info'])
    if has_bad:
        first = c1['lines'][0] if c1['lines'] else '?'
        last = c1['lines'][-1] if c1['lines'] else '?'
        nobh_fx = "; ".join(c1['fx_info']) if c1['fx_info'] else "none"
        bh_fx = "; ".join(c2['fx_info']) if c2['fx_info'] else "none"
        print(f"  [{idx}] {c1['fx_type']} n={c1['n_lines']:2d} lines[{first}..{last}]  "
              f"no_bh:{nobh_fx}  bh:{bh_fx}")
