"""
Debug why our code gives down bi[28→44] instead of down bi[28→38] for BTC60.
Trace _find_xd_end step by step.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_open import CL

df = pd.read_parquet("tests/test_data/BTC_USDT_60m_1000.parquet")
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

cl = CL("BTC60", "60m", config)
cl.process_klines(df)
bis = cl.get_bis()

print("=== BTC60: calling _find_xd_end(bis, 28, 'down') ===")
print(f"  Total BIs: {len(bis)}")

# Reproduce _find_xd_end logic manually
start_bi_idx = 28
xd_type = "down"
tzxl_bi_type = "up"  # for down segment, use UP BIs
bh_direction = "down"
target_fx_type = "di"

# Collect UP BIs starting from bi[28]
tzxl_bis = []
for i in range(start_bi_idx, len(bis)):
    if bis[i].type == tzxl_bi_type:
        tzxl_bis.append(bis[i])

print(f"\n  UP BIs from bi[28]: {[b.index for b in tzxl_bis]}")
for b in tzxl_bis:
    print(f"    bi[{b.index}]: high={b.high}, low={b.low}")

# Build TZXLs with containment
from chanlun.cl_interface import TZXL
tzxls = []
for bi in tzxl_bis:
    pre_line = bis[bi.index - 1] if bi.index > 0 else bi
    done = bi.is_done()
    new_tzxl = TZXL(
        bh_direction=bh_direction,
        line=bi,
        pre_line=pre_line,
        line_bad=False,
        done=done,
    )
    
    if len(tzxls) == 0:
        tzxls.append(new_tzxl)
        continue
    
    last_tzxl = tzxls[-1]
    old_contains_new = last_tzxl.max >= new_tzxl.max and last_tzxl.min <= new_tzxl.min
    new_contains_old = new_tzxl.max >= last_tzxl.max and new_tzxl.min <= last_tzxl.min
    
    if old_contains_new:
        print(f"    → OLD⊃NEW: merge bi[{bi.index}] into TZXL lines={[l.index for l in last_tzxl.lines]}")
        last_tzxl.lines.append(bi)
        last_tzxl.done = done
        last_tzxl.line_bad = False
        last_tzxl.update_maxmin()
    elif new_contains_old:
        print(f"    → NEW⊃OLD: bi[{bi.index}] separate, bad=True")
        new_tzxl.line_bad = True
        tzxls.append(new_tzxl)
    else:
        tzxls.append(new_tzxl)

print(f"\n  Built {len(tzxls)} TZXLs:")
for i, t in enumerate(tzxls):
    lines = [l.index for l in t.lines]
    print(f"    TZXL[{i}]: max={t.max:.1f} min={t.min:.1f} bad={t.line_bad} lines={lines}")

# Search for DI FX
print(f"\n  Searching for DI FX:")
first_bad_result = None
first_bad_extreme = None

for i in range(1, len(tzxls) - 1):
    curr_xl = tzxls[i]
    prev_xl = tzxls[i - 1]
    next_xl = tzxls[i + 1]
    
    is_fx = curr_xl.min < prev_xl.min and curr_xl.min < next_xl.min
    
    if is_fx:
        print(f"    FX at TZXL[{i}] lines={[l.index for l in curr_xl.lines]} min={curr_xl.min:.1f} bad={curr_xl.line_bad}")
        
        # Check bi_pohuai
        pohuai = cl._check_xd_bi_pohuai(bis, start_bi_idx, curr_xl, xd_type)
        print(f"      bi_pohuai = {pohuai}")
        
        if not pohuai:
            # Build result
            from chanlun.cl_interface import XLFX
            end_bi = min(curr_xl.lines, key=lambda l: l.low)
            end_bi_idx = end_bi.index
            print(f"      end_bi = bi[{end_bi_idx}] (low={end_bi.low:.1f})")
            if bis[end_bi_idx].type == "up" and end_bi_idx > 0:
                end_bi_idx -= 1
                print(f"      → adjusted to bi[{end_bi_idx}] (up→down)")
            
            if end_bi_idx - start_bi_idx >= 2:
                print(f"      → VALID result: end={end_bi_idx}")
                
                if curr_xl.line_bad:
                    print(f"      → BAD FX, storing as first_bad")
                    if first_bad_result is None:
                        first_bad_result = f"end={end_bi_idx}"
                        first_bad_extreme = curr_xl.min
                    # Continue looking for non-bad
                else:
                    print(f"      → NON-BAD FX")
                    if first_bad_result is not None:
                        is_more_extreme = curr_xl.min < first_bad_extreme
                        print(f"      → Compare: non-bad min={curr_xl.min:.1f} vs bad min={first_bad_extreme:.1f}")
                        print(f"      → non-bad more extreme: {is_more_extreme}")
                        if is_more_extreme:
                            print(f"      → USING NON-BAD")
                        else:
                            print(f"      → USING BAD (first_bad_result={first_bad_result})")
                    else:
                        print(f"      → USING NON-BAD (no prior bad)")
                    break
            else:
                print(f"      → INVALID: end-start={end_bi_idx - start_bi_idx} < 2")

if first_bad_result is not None and not any(not tzxls[i].line_bad for i in range(1, len(tzxls) - 1) if tzxls[i].min < tzxls[i-1].min and tzxls[i].min < tzxls[i+1].min):
    print(f"\n  No non-bad FX found, falling back to first_bad: {first_bad_result}")

# Now call the actual method
print(f"\n=== Actual _find_xd_end result ===")
result = cl._find_xd_end(bis, 28, "down")
if result:
    end_bi_idx, ding_fx, di_fx, tzxls = result
    print(f"  end_bi_idx={end_bi_idx}")
    print(f"  di_fx lines={[l.index for l in di_fx.xls[1].lines]}")
    print(f"  di_fx bad={di_fx.is_line_bad}")
else:
    print(f"  None!")
