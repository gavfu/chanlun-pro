"""Debug BI 'bi' BC calculation differences.

bi[3] UP: pyarmor has 'bi', we have none.
- bi[3].high=69033 < bi[1].high=70110.9 → NOT new extreme
- So ours requires 'new extreme' but pyarmor might not

bi[2] DOWN: we have 'bi', pyarmor has none.
- bi[2].low=67785.4 > bi[0].low=68112.2 → NOT new extreme
- So... our algorithm should also not give bi[2] 'bi'
  unless we have a bug where we check wrongly

Let me trace through bi type BC calculations.
"""
import sys; sys.path.insert(0, 'src')
import pandas as pd
from chanlun.cl_pyarmor import CL as CL_P
from chanlun.cl_open import CL as CL_O
from chanlun.cl_interface import BC, compare_ld_beichi

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')
cd_p = CL_P('BTC/USDT', '60m')
cd_p.process_klines(df)
cd_o = CL_O('BTC/USDT', '60m')
cd_o.process_klines(df)

print("=== BC type 'bi' for all BIs (pyarmor) ===")
for bi in cd_p.bis:
    bi_bc = [bc for bc in bi.bcs if bc.type == 'bi']
    if bi_bc:
        # Find the compare line
        compare_line = bi_bc[0].compare_line
        if compare_line:
            print(f"  bi[{bi.index}] {bi.type}: compare with bi[{compare_line.index}]")
            # Check if new extreme
            if bi.type == 'up':
                new_extreme = bi.high > compare_line.high
            else:
                new_extreme = bi.low < compare_line.low
            print(f"    new_extreme={new_extreme} bi.high={bi.high:.1f} vs bi[{compare_line.index}].high={compare_line.high:.1f}")
        else:
            print(f"  bi[{bi.index}] {bi.type}: compare_line=None")

print("\n=== BC type 'bi' for all BIs (open) ===")
for bi in cd_o.bis:
    bi_bc = [bc for bc in bi.bcs if bc.type == 'bi']
    if bi_bc:
        compare_line = bi_bc[0].compare_line
        if compare_line:
            print(f"  bi[{bi.index}] {bi.type}: compare with bi[{compare_line.index}]")
            if bi.type == 'up':
                new_extreme = bi.high > compare_line.high
            else:
                new_extreme = bi.low < compare_line.low
            print(f"    new_extreme={new_extreme}")
        else:
            print(f"  bi[{bi.index}] {bi.type}: compare_line=None")

# Check: does pyarmor require new extreme for bi BC?
print("\n=== Checking new_extreme requirement ===")
for bi in cd_p.bis:
    bi_bc = [bc for bc in bi.bcs if bc.type == 'bi']
    if bi_bc:
        compare_line = bi_bc[0].compare_line
        if compare_line:
            if bi.type == 'up':
                new_extreme = bi.high > compare_line.high
            else:
                new_extreme = bi.low < compare_line.low
            if not new_extreme:
                print(f"  bi[{bi.index}] has bi-BC WITHOUT new extreme! (not new high/low)")
