"""Check MMD details: ZS ZG/ZD, bi[6] mmds, bi[8] bc, bi[13] ZS trigger"""
import sys; sys.path.insert(0, 'src')
from chanlun.cl_open import CL
from chanlun.cl_pyarmor import CL as CLP
import pandas as pd

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')
config = {}
cd = CL('BTC/USDT', '60m', config=config); cd.process_klines(df)
cdp = CLP('BTC/USDT', '60m', config=config); cdp.process_klines(df)
bis = cd.get_bis(); bisp = cdp.get_bis()
bi_zss = cd.get_bi_zss(); bi_zssp = cdp.get_bi_zss()

# ZS ZG/ZD
for i, (zo, zp) in enumerate(zip(bi_zss, bi_zssp)):
    print(f'ZS[{i}] open: ZG={zo.zg:.1f} ZD={zo.zd:.1f} start={zo.start.k.index} end={zo.end.k.index} done={zo.done} line_num={zo.line_num}')
    print(f'        pyar: ZG={zp.zg:.1f} ZD={zp.zd:.1f} start={zp.start.k.index} end={zp.end.k.index} done={zp.done} line_num={zp.line_num}')

print()
# bi[6] mmds - needed for bi[8] and bi[7] cascade
for j in [4, 5, 6, 7]:
    print(f'bi[{j}] open {bis[j].type}: low={bis[j].low:.1f} high={bis[j].high:.1f} mmds={bis[j].line_mmds("|")} bcs={bis[j].line_bcs("|")}')
    print(f'bi[{j}] pyar {bisp[j].type}: low={bisp[j].low:.1f} high={bisp[j].high:.1f} mmds={bisp[j].line_mmds("|")} bcs={bisp[j].line_bcs("|")}')

print()
print('bi[8] info:')
print(f'  open: bc_exists={bis[8].bc_exists(["bi","pz","qs"], "|")} low={bis[8].low:.1f} < bi[6].low={bis[6].low:.1f}: {bis[8].low < bis[6].low}')
print(f'  bi[6] open has 3sell: {"3sell" in bis[6].line_mmds("|")}')
print(f'  bi[6] pyar has 3sell: {"3sell" in bisp[6].line_mmds("|")}')

print()
print('bi[13] 3sell check:')
for i, zs in enumerate(bi_zss):
    last_idx = zs.lines[-1].index
    if bis[13].index > last_idx and bis[13].type == 'up':
        print(f'  open ZS[{i}]: last={last_idx}, hi={bis[13].high:.1f} < zd={zs.zd:.1f}? {bis[13].high < zs.zd}, done={zs.done}, real={zs.real}, line_num={zs.line_num}')
print('pyar:')
for i, zs in enumerate(bi_zssp):
    last_idx = zs.lines[-1].index
    if bisp[13].index > last_idx and bisp[13].type == 'up':
        print(f'  pyar ZS[{i}]: last={last_idx}, hi={bisp[13].high:.1f} < zd={zs.zd:.1f}? {bisp[13].high < zs.zd}, done={zs.done}, real={zs.real}, line_num={zs.line_num}')
