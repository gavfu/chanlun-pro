"""Check pyarmor bi[8] 1buy detail - what ZS and path was used?"""
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

# Show all MMDs for bi[8] in pyarmor with details
print("bi[8] pyarmor mmds detail:")
bp8 = bisp[8]
for mmd in bp8.mmds:
    zs_info = f"zs.lines=[{mmd.zs.lines[0].index}..{mmd.zs.lines[-1].index}]" if mmd.zs else "zs=None"
    print(f"  mmd: name={mmd.name} {zs_info}")

# For each zs_type_mmds in bp8
print("\nbp8.zs_type_mmds keys:", list(bp8.zs_type_mmds.keys()))
for zs_type_key, mmds in bp8.zs_type_mmds.items():
    for mmd in mmds:
        zs_info = f"zs.lines=[{mmd.zs.lines[0].index}..{mmd.zs.lines[-1].index}]" if mmd.zs else "zs=None"
        print(f"  key={zs_type_key} mmd: name={mmd.name} {zs_info}")

# All MMDs for bi 7-10
print("\n=== MMD cascade bi[7..12] ===")
for j in range(5, 15):
    boj = bis[j]
    bpj = bisp[j]
    o_mmds = boj.line_mmds('|')
    p_mmds = bpj.line_mmds('|')
    marker = "  " if o_mmds == p_mmds else "!!"
    print(f"{marker} bi[{j:2d}] {boj.type:4s}: open_mmd={o_mmds}  pyar_mmd={p_mmds}")

# Check if bi[7] has 3sell in pyarmor
print(f"\nbi[7] type: {bisp[7].type}")
print(f"bi[7] pyar mmds: {bisp[7].line_mmds('|')}")

# Let me check what prev_line (bi[6]) has in pyarmor 
# vs what triggers 1buy for bi[8] in pyarmor
# Maybe pyarmor looks at bi[idx-1] (opposite direction) for 3sell?
print(f"\nbi[6] pyar mmds: {bisp[6].line_mmds('|')}")
print(f"bi[7] pyar mmds: {bisp[7].line_mmds('|')}")
# bi[8] is down, bi[7] is up with 3sell
# If pyarmor checks "opposite direction prev line" for 3sell...
print(f"\nHypothesis: pyarmor looks at bi[7] (up, idx-1) which has 3sell -> bi[8] (down) gets 1buy")

# Check beichi_qs one more time with all params
zss_o = [zs for zs in bi_zss if zs.zs_type == "bi"]
zss_p = [zs for zs in bi_zssp if zs.zs_type == "bi"]
print(f"\nbeichi_qs for bi[8]:")
print(f"  open: {cd.beichi_qs(bis, zss_o, bis[8])}")
print(f"  pyar: {cdp.beichi_qs(bisp, zss_p, bisp[8])}")
print(f"zss_is_qs(ZS[0],ZS[1]):")
print(f"  open: {cd.zss_is_qs(zss_o[0], zss_o[1])}")
print(f"  pyar: {cdp.zss_is_qs(zss_p[0], zss_p[1])}")
