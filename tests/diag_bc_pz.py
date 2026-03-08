"""Diagnose extra 'pz' BC in open vs pyarmor"""
import sys
sys.path.insert(0, 'src')
from chanlun.cl_open import CL
from chanlun.cl_pyarmor import CL as CLP
from chanlun.cl_interface import compare_ld_beichi
import numpy as np
import pandas as pd

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')
klines = df

config = {}
cd_open = CL('BTC/USDT', '60m', config=config)
cd_pyarmor = CLP('BTC/USDT', '60m', config=config)
cd_open.process_klines(klines)
cd_pyarmor.process_klines(klines)

bis_o = cd_open.get_bis()
bis_p = cd_pyarmor.get_bis()
bi_zss_o = cd_open.get_bi_zss()
bi_zss_p = cd_pyarmor.get_bi_zss()

print("=== All BC diffs (using line_bcs) ===")
for i, (bo, bp) in enumerate(zip(bis_o, bis_p)):
    o_bcs = sorted(bo.line_bcs("|"))
    p_bcs = sorted(bp.line_bcs("|"))
    if o_bcs != p_bcs:
        print(f"\n  bi[{i}] open_bc={o_bcs}  pyarmor_bc={p_bcs}")
        print(f"    bi: {bo.type} from {bo.start.type}@{bo.start.k.index} to {bo.end.type}@{bo.end.k.index}")
        # Find related ZS and compute enter_line
        for zs in reversed(bi_zss_o):
            zs_lines_idxs = [l.index for l in zs.lines]
            if zs.lines and bo.index > zs.lines[-1].index:
                related_zs = zs
                break
            if bo.index in zs_lines_idxs:
                related_zs = zs
                break
        else:
            related_zs = None
        if related_zs:
            enter_line = None
            for line in related_zs.lines:
                if line.type == bo.type:
                    enter_line = line
                    break
            enter_ld = enter_line.get_ld(cd_open) if enter_line else None
            now_ld = bo.get_ld(cd_open)
            bc = compare_ld_beichi(enter_ld, now_ld, bo.type) if enter_ld else None
            print(f"    open related_zs: lines idx=[{related_zs.lines[0].index}..{related_zs.lines[-1].index}]")
            if enter_line:
                print(f"    open enter_line: bi[{enter_line.index}] {enter_line.type} k_range=[{enter_line.start.k.k_index},{enter_line.end.k.k_index}]")
                print(f"    open now_line:   bi[{bo.index}] k_range=[{bo.start.k.k_index},{bo.end.k.k_index}]")
                if enter_ld and now_ld:
                    enter_hist = enter_ld['macd']['hist']
                    now_hist = now_ld['macd']['hist']
                    print(f"    open enter_ld: up={enter_hist['up_sum']:.2f} down={enter_hist['down_sum']:.2f}")
                    print(f"    open now_ld:   up={now_hist['up_sum']:.2f} down={now_hist['down_sum']:.2f}")
                print(f"    open pz_bc = {bc}")
        # Show pyarmor raw bcs
        for bc in bp.bcs:
            zs_info = f"zs.real={bc.zs.real}" if bc.zs else "zs=None"
            print(f"    pyarm BC: type={bc.type} bc={bc.bc} {zs_info}")

# Show ZS info for BIs with extra pz
print("\n=== ZS info ===")
bi_zss = cd_open.get_bi_zss()
bi_zss_p = cd_pyarmor.get_bi_zss()

print(f"open ZSs: {len(bi_zss)}, pyarmor ZSs: {len(bi_zss_p)}")
for i, zs in enumerate(bi_zss):
    print(f"  ZS[{i}]: lines idx=[{zs.lines[0].index}..{zs.lines[-1].index}]  start_fx@{zs.start.k.index}  end_fx@{zs.end.k.index}  ZG={zs.zg:.1f}  ZD={zs.zd:.1f}")

print()
for i, zs in enumerate(bi_zss_p):
    print(f"  PY_ZS[{i}]: lines idx=[{zs.lines[0].index}..{zs.lines[-1].index}]  start_fx@{zs.start.k.index}  end_fx@{zs.end.k.index}  ZG={zs.zg:.1f}  ZD={zs.zd:.1f}")

