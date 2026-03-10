"""
Check bi_pohuai for ETH60 down bi[34] FX candidates.
If FX at bi[37] is rejected by bi_pohuai, that explains why pyarmor skips it.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_pyarmor import CL

df = pd.read_parquet("tests/test_data/ETH_USDT_60m_1000.parquet")
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

cl = CL("ETH60", "60m", config)
cl.process_klines(df)
bis = cl.get_bis()

from chanlun.cl_interface import TZXL

# Build TZXL for bi[37] and check bi_pohuai
# For down bi[34], TZXL uses UP BIs with bh_direction="down"
bi37 = bis[37]
bi35 = bis[35]
bi39 = bis[39]

tzxl_37 = TZXL(
    bh_direction="down",
    line=bi37,
    pre_line=bis[36],
    line_bad=False,
    done=bi37.is_done(),
)

print("=== ETH60 down bi[34]: FX at bi[37] ===")
print(f"  bi[34]: type={bis[34].type}, high={bis[34].high}, low={bis[34].low}")
print(f"  bi[37]: type={bi37.type}, high={bi37.high}, low={bi37.low}")
print(f"  bi[38]: type={bis[38].type}, high={bis[38].high}, low={bis[38].low}")
print(f"  bi[39]: type={bi39.type}, high={bi39.high}, low={bi39.low}")
print(f"  bi[40]: type={bis[40].type}, high={bis[40].high}, low={bis[40].low}")
print(f"  bi[41]: type={bis[41].type}, high={bis[41].high}, low={bis[41].low}")

# Bi pohuai check:
# For a DOWN segment starting at bi[34], if DI FX is at TZXL containing bi[37]:
# Check if bi[38] (the next bi after bi[37]) breaks above bi[34]'s high
print(f"\n  bi_pohuai check for DI FX at bi[37]:")
print(f"    Checking BIs after bi[37] (last line in TZXL):")
print(f"    bi[34].high = {bis[34].high}")
# For DOWN segment, bi_pohuai checks if any BI AFTER the FX's last line
# has a UP bi with high > start's high
for bi_idx in range(38, min(50, len(bis))):
    bi = bis[bi_idx]
    if bi.type == "up":
        if bi.high > bis[34].high:
            print(f"    bi[{bi_idx}] (up): high={bi.high} > {bis[34].high} → BREAKS!")
        else:
            print(f"    bi[{bi_idx}] (up): high={bi.high} <= {bis[34].high}")
    # Only check ONE bi after
    break

# Actually check our _check_xd_bi_pohuai
from chanlun.cl_open import CL as CL_Open
cl_open = CL_Open("test", "60m", config)
cl_open.process_klines(df)
our_bis = cl_open.get_bis()

print("\n  Using our _check_xd_bi_pohuai:")
# Need to build a TZXL from our BIs
bi37_our = our_bis[37]
tzxl_check = TZXL(
    bh_direction="down",
    line=bi37_our,
    pre_line=our_bis[36],
    line_bad=False,
    done=bi37_our.is_done(),
)
result = cl_open._check_xd_bi_pohuai(our_bis, 34, tzxl_check, "down")
print(f"    bi_pohuai(down bi[34], FX at bi[37]) = {result}")

# Also build TZXL for bi[39,41] (bh mode) 
tzxl_39_41 = TZXL(
    bh_direction="down",
    line=our_bis[39],
    pre_line=our_bis[38],
    line_bad=False,
    done=our_bis[41].is_done(),
)
tzxl_39_41.lines.append(our_bis[41])
tzxl_39_41.update_maxmin()
result2 = cl_open._check_xd_bi_pohuai(our_bis, 34, tzxl_39_41, "down")
print(f"    bi_pohuai(down bi[34], FX at bi[39,41]) = {result2}")

# Now check _check_xd_bi_pohuai logic in detail
print(f"\n  Detailed bi_pohuai check for bi[37]:")
print(f"    start_bi = bi[34], type={our_bis[34].type}")
last_line_idx = 37  # last line in TZXL
for bi_idx in range(last_line_idx + 1, min(last_line_idx + 5, len(our_bis))):
    bi = our_bis[bi_idx]
    print(f"    bi[{bi_idx}]: type={bi.type}, high={bi.high}, low={bi.low}")
    if bi.type == "down":
        if bi.low < our_bis[34].low:
            print(f"      → DOWN bi breaks below start ({bi.low} < {our_bis[34].low})")

# For DOWN segment, bi_pohuai checks: after the FX's last line, 
# is there a bi that violates the segment direction?
# Looking at the actual code:
print("\n  Re-reading _check_xd_bi_pohuai code...")
