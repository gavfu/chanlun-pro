"""Diagnose BI MMD differences: bi[8] missing 1buy, bi[10] missing 2buy, bi[13] extra 3sell"""
import sys
sys.path.insert(0, 'src')
from chanlun.cl_open import CL
from chanlun.cl_pyarmor import CL as CLP
import pandas as pd

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')
config = {}
cd_open = CL('BTC/USDT', '60m', config=config)
cd_pyarmor = CLP('BTC/USDT', '60m', config=config)
cd_open.process_klines(df)
cd_pyarmor.process_klines(df)

bis_o = cd_open.get_bis()
bis_p = cd_pyarmor.get_bis()
bi_zss_o = cd_open.get_bi_zss()
bi_zss_p = cd_pyarmor.get_bi_zss()

# Show ZS info
print("=== ZS boundaries ===")
for i, (zo, zp) in enumerate(zip(bi_zss_o, bi_zss_p)):
    print(f"  ZS[{i}]: open start_fx@{zo.start.k.index}  end_fx@{zo.end.k.index}  lines=[{zo.lines[0].index}..{zo.lines[-1].index}]  level={zo.level}  real={zo.real}  done={zo.done}")
    print(f"         pyar start_fx@{zp.start.k.index}  end_fx@{zp.end.k.index}  lines=[{zp.lines[0].index}..{zp.lines[-1].index}]  level={zp.level}  real={zp.real}  done={zp.done}")

# Focus on bi[8] - should get 1buy
print("\n=== bi[8] (down, should get '1buy') ===")
bo8 = bis_o[8]
bp8 = bis_p[8]
print(f"  open:   {bo8.type} {bo8.start.type}@{bo8.start.k.index} -> {bo8.end.type}@{bo8.end.k.index}  high={bo8.high:.1f} low={bo8.low:.1f}")
print(f"  pyarm:  {bp8.type} {bp8.start.type}@{bp8.start.k.index} -> {bp8.end.type}@{bp8.end.k.index}  high={bp8.high:.1f} low={bp8.low:.1f}")
print(f"  open mmds: {bo8.line_mmds('|')}")
print(f"  pyar mmds: {bp8.line_mmds('|')}")
print(f"  open bcs: {bo8.line_bcs('|')}")
print(f"  pyar bcs: {bp8.line_bcs('|')}")

# Check beichi_qs for bi[8]
from chanlun.cl_open import CL as CLOpen_
zss_o = [zs for zs in bi_zss_o if zs.zs_type == "bi"]
bc_qs, compare_lines = cd_open.beichi_qs(bis_o, zss_o, bo8)
print(f"  open beichi_qs: {bc_qs}, compare_lines: {[l.index for l in compare_lines]}")

zss_p = [zs for zs in bi_zss_p if zs.zs_type == "bi"]
bc_qs_p, compare_lines_p = cd_pyarmor.beichi_qs(bis_p, zss_p, bp8)
print(f"  pyar beichi_qs: {bc_qs_p}, compare_lines: {[l.index for l in compare_lines_p]}")

# zss_is_qs check
if len(zss_o) >= 2:
    qs_dir = cd_open.zss_is_qs(zss_o[-2], zss_o[-1])
    print(f"  open zss_is_qs(ZS[-2], ZS[-1]) = {qs_dir}")
if len(zss_p) >= 2:
    qs_dir_p = cd_pyarmor.zss_is_qs(zss_p[-2], zss_p[-1])
    print(f"  pyar zss_is_qs(ZS[-2], ZS[-1]) = {qs_dir_p}")

# Focus on bi[10] - should get 2buy
print("\n=== bi[10] (down, should get '2buy') ===")
bo10 = bis_o[10]
bp10 = bis_p[10]
print(f"  open mmds: {bo10.line_mmds('|')}")
print(f"  pyar mmds: {bp10.line_mmds('|')}")
print(f"  open bcs: {bo10.line_bcs('|')}")
# Check bi[8] for 1buy (needed for 2buy chain)
print(f"  bi[8] open mmds: {bis_o[8].line_mmds('|')}")
print(f"  bi[8] pyar mmds: {bis_p[8].line_mmds('|')}")
# Check condition: bi[10].low > bi[8].low
print(f"  bi[10].low={bo10.low:.1f} > bi[8].low={bis_o[8].low:.1f}? {bo10.low > bis_o[8].low}")

# Focus on bi[13] - should NOT get 3sell
print("\n=== bi[13] (up, should NOT get '3sell') ===")
bo13 = bis_o[13]
bp13 = bis_p[13]
print(f"  open:   {bo13.type} {bo13.start.type}@{bo13.start.k.index} -> {bo13.end.type}@{bo13.end.k.index}  high={bo13.high:.1f}")
print(f"  open mmds: {bo13.line_mmds('|')}")
print(f"  pyar mmds: {bp13.line_mmds('|')}")
# 3sell condition: up line, after last_zs_line, high < zs.zd
# Which ZS is this vs?
for i, zs in enumerate(bi_zss_o):
    last_idx = zs.lines[-1].index
    if bo13.index > last_idx and bo13.type == "up":
        print(f"  ZS[{i}]: last_line.index={last_idx}, bo13.high={bo13.high:.1f} < zs.zd={zs.zd:.1f}? {bo13.high < zs.zd}")
        print(f"         real={zs.real} done={zs.done}")
