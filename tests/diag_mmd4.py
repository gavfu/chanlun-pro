"""Show all BI MMDs in pyarmor vs open to understand 3sell logic"""
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

# Show all non-empty MMDs
print("=== All non-empty BI MMDs ===")
for j in range(len(bis)):
    o_mmd = bis[j].line_mmds('|')
    p_mmd = bisp[j].line_mmds('|')
    if o_mmd or p_mmd:
        marker = "  " if o_mmd == p_mmd else "!!"
        print(f"{marker} bi[{j:2d}] {bis[j].type:4s} fx@{bis[j].start.k.index:3d}->{bis[j].end.k.index:3d}: open={o_mmd}  pyar={p_mmd}")

# Focus on ZS[1] and bi[7] vs bi[13]:
# bi[7]: up, after ZS[0] (last=6), high=67299.4 < ZD(ZS[0])=68112.2? True -> 3sell
# bi[13]: up, after ZS[0] (last=6), high=66240.4 < ZD(ZS[0])=68112.2? True -> 3sell (open) but NOT pyar
# 
# What is different about bi[7] vs bi[13]?
# bi[7].index=7, bi[13].index=13
# ZS[1] lines=[6..23]: bi[7] IS in ZS[1], bi[13] IS in ZS[1]
# 
# BUT: maybe pyarmor only gives 3sell to the FIRST up stroke after ZS[0]?
# i.e., only the stroke right after the ZS's last line?
# bi[7].index = 7 = last_line.index+1 = 6+1 -> FIRST stroke after ZS[0]
# bi[13].index = 13 >> 6+1 -> NOT the first stroke after ZS[0]

print("\n=== Check if 'only first stroke' hypothesis holds ===")
zss = [zs for zs in bi_zss if zs.zs_type == "bi"]
for i, zs in enumerate(zss):
    last_idx = zs.lines[-1].index
    print(f"ZS[{i}]: last_line.index={last_idx}")
    for j in range(len(bisp)):
        bp = bisp[j]
        o = bis[j]
        p_3sell = '3sell' in bp.line_mmds('|')
        p_3buy = '3buy' in bp.line_mmds('|')
        if (p_3sell or p_3buy) and bp.index > last_idx:
            print(f"  pyar bi[{j:2d}] {bp.type:4s}: {bp.line_mmds('|')}  (index - last_idx = {bp.index - last_idx})")
