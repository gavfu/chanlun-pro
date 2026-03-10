"""
Check bh vs no_bh for BTC5m to understand the impact of dual-mode.
Focus on xd[9] which is down bi[54→56] (ours) vs down bi[54→64] (pyarmor).
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_open import CL
from chanlun.cl_interface import TZXL

df = pd.read_parquet("tests/test_data/BTC_USDT_5m_1000.parquet")
config = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
    "xd_zs_max_lines_split": 11,
    "xd_allow_split_no_highlow": 1,
    "xd_allow_split_zs_kz": 0,
    "xd_allow_split_zs_more_line": 1,
    "xd_allow_split_zs_no_direction": 1,
}

cl = CL("BTC5m", "5m", config)
cl.process_klines(df)
bis = cl.get_bis()

def build_tzxls(bis, start_bi_idx, xd_type, mode):
    """Build TZXLs in bh or no_bh mode"""
    tzxl_bi_type = "down" if xd_type == "up" else "up"
    bh_direction = "up" if xd_type == "up" else "down"
    
    tzxl_bis = [b for b in bis[start_bi_idx:] if b.type == tzxl_bi_type]
    
    tzxls = []
    for bi in tzxl_bis:
        pre_line = bis[bi.index - 1] if bi.index > 0 else bi
        new_tzxl = TZXL(
            bh_direction=bh_direction,
            line=bi,
            pre_line=pre_line,
            line_bad=False,
            done=bi.is_done(),
        )
        
        if len(tzxls) == 0:
            tzxls.append(new_tzxl)
            continue
        
        last_tzxl = tzxls[-1]
        old_contains_new = last_tzxl.max >= new_tzxl.max and last_tzxl.min <= new_tzxl.min
        new_contains_old = new_tzxl.max >= last_tzxl.max and new_tzxl.min <= last_tzxl.min
        
        if old_contains_new:
            last_tzxl.lines.append(bi)
            last_tzxl.done = bi.is_done()
            last_tzxl.line_bad = False
            last_tzxl.update_maxmin()
        elif new_contains_old:
            if mode == 'bh':
                # bh mode: merge too
                last_tzxl.lines.append(bi)
                last_tzxl.done = bi.is_done()
                last_tzxl.line_bad = False
                last_tzxl.update_maxmin()
            else:
                # no_bh mode: separate with bad=True
                new_tzxl.line_bad = True
                tzxls.append(new_tzxl)
        else:
            tzxls.append(new_tzxl)
    
    return tzxls

def find_first_fx(tzxls, target_fx_type, bis, start_bi_idx, xd_type, cl):
    """Find first valid FX in TZXLs"""
    if len(tzxls) < 3:
        return None
    
    first_bad_result = None
    first_bad_extreme = None
    
    for i in range(1, len(tzxls) - 1):
        curr = tzxls[i]
        prev_xl = tzxls[i - 1]
        next_xl = tzxls[i + 1]
        
        if target_fx_type == "di":
            is_fx = curr.min < prev_xl.min and curr.min < next_xl.min
        else:
            is_fx = curr.max > prev_xl.max and curr.max > next_xl.max
        
        if is_fx:
            pohuai = cl._check_xd_bi_pohuai(bis, start_bi_idx, curr, xd_type)
            if not pohuai:
                if target_fx_type == "di":
                    end_bi = min(curr.lines, key=lambda l: l.low)
                else:
                    end_bi = max(curr.lines, key=lambda l: l.high)
                end_bi_idx = end_bi.index
                if xd_type == "down" and bis[end_bi_idx].type == "up":
                    end_bi_idx -= 1
                elif xd_type == "up" and bis[end_bi_idx].type == "down":
                    end_bi_idx -= 1
                
                if end_bi_idx - start_bi_idx >= 2:
                    lines = [l.index for l in curr.lines]
                    
                    if curr.line_bad:
                        if first_bad_result is None:
                            first_bad_result = (end_bi_idx, curr.min if target_fx_type == "di" else curr.max, lines, True)
                            first_bad_extreme = curr.min if target_fx_type == "di" else curr.max
                        continue
                    
                    # Non-bad found
                    if first_bad_result is not None:
                        if target_fx_type == "di":
                            is_more = curr.min < first_bad_extreme
                        else:
                            is_more = curr.max > first_bad_extreme
                        if is_more:
                            return (end_bi_idx, curr.min if target_fx_type == "di" else curr.max, lines, False)
                        else:
                            return first_bad_result
                    
                    return (end_bi_idx, curr.min if target_fx_type == "di" else curr.max, lines, False)
    
    return first_bad_result

# Test for BTC5m down bi[54]
print("=== BTC5m down bi[54] ===")
print(f"Pyarmor: end=64, Our: end=56")

for mode in ['no_bh', 'bh']:
    tzxls = build_tzxls(bis, 54, "down", mode)
    print(f"\n  {mode} TZXLs ({len(tzxls)}):")
    for i, t in enumerate(tzxls[:12]):
        lines = [l.index for l in t.lines]
        print(f"    [{i}] max={t.max:.1f} min={t.min:.1f} bad={t.line_bad} lines={lines}")
    
    fx = find_first_fx(tzxls, "di", bis, 54, "down", cl)
    print(f"  First FX: {fx}")

# Also check BTCd xd[1] down bi[22→24] (ours) vs down bi[22→28] (pyarmor)
print("\n\n=== BTCd ===")
df_d = pd.read_parquet("tests/test_data/BTC_USDT_d_500.parquet")
cl_d = CL("BTCd", "d", config)
cl_d.process_klines(df_d)
bis_d = cl_d.get_bis()

print("BTCd down bi[22]: Our end=24, Pyarmor end=28")
for mode in ['no_bh', 'bh']:
    tzxls = build_tzxls(bis_d, 22, "down", mode)
    print(f"\n  {mode} TZXLs ({len(tzxls)}):")
    for i, t in enumerate(tzxls[:12]):
        lines = [l.index for l in t.lines]
        print(f"    [{i}] max={t.max:.1f} min={t.min:.1f} bad={t.line_bad} lines={lines}")
    
    fx = find_first_fx(tzxls, "di", bis_d, 22, "down", cl_d)
    print(f"  First FX: {fx}")

# ETH60 xd[10] down bi[52→57] (ours) vs down bi[52→56] (pyarmor)
print("\n\n=== ETH60 ===")
df_eth = pd.read_parquet("tests/test_data/ETH_USDT_60m_1000.parquet")
cl_eth = CL("ETH60", "60m", config)
cl_eth.process_klines(df_eth)
bis_eth = cl_eth.get_bis()

print("ETH60 down bi[52]: Our end=57, Pyarmor end=56")
for mode in ['no_bh', 'bh']:
    tzxls = build_tzxls(bis_eth, 52, "down", mode)
    print(f"\n  {mode} TZXLs ({len(tzxls)}):")
    for i, t in enumerate(tzxls[:12]):
        lines = [l.index for l in t.lines]
        print(f"    [{i}] max={t.max:.1f} min={t.min:.1f} bad={t.line_bad} lines={lines}")
    
    fx = find_first_fx(tzxls, "di", bis_eth, 52, "down", cl_eth)
    print(f"  First FX: {fx}")
