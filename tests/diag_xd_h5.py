"""Test Hypothesis 5 for all sequence fractals: pohuai = next bi exceeds fx max."""
import sys; sys.path.insert(0, 'src')
import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_interface import TZXL

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')
cd_o = CL_O('BTC/USDT', '60m')
cd_o.process_klines(df)
bis = cd_o.bis

start_bi_idx = 2
xd_type = "down"
bh_direction = "down"

# Build TZXL from bi[2]
tzxl_bis = [bi for bi in bis[start_bi_idx:] if bi.type == "up"]

tzxls = []
for bi in tzxl_bis:
    pre_line = bis[bi.index - 1] if bi.index > 0 else bi
    new_tzxl = TZXL(bh_direction=bh_direction, line=bi, pre_line=pre_line,
                    line_bad=False, done=bi.is_done())
    if not tzxls:
        tzxls.append(new_tzxl)
        continue
    last = tzxls[-1]
    if (last.max >= new_tzxl.max and last.min <= new_tzxl.min) or \
       (new_tzxl.max >= last.max and new_tzxl.min <= last.min):
        last.lines.append(bi)
        last.done = bi.is_done()
        last.update_maxmin()
    else:
        tzxls.append(new_tzxl)

print("Final TZXL:")
for i, xl in enumerate(tzxls):
    lines = [l.index for l in xl.lines]
    print(f"  [{i}] max={xl.max:.1f} min={xl.min:.1f} lines=bi{lines}")

print(f"\nSearching for 'di' sequence fractals with H5 pohuai check:")
print("(pohuai = next UP stroke exceeds fx.max)")
print()

for i in range(1, len(tzxls) - 1):
    prev = tzxls[i - 1]
    curr = tzxls[i]
    nxt = tzxls[i + 1]
    
    is_fx = curr.min < prev.min and curr.min < nxt.min
    
    if is_fx:
        last_line_idx = curr.lines[-1].index
        
        # H5: check if next stroke AFTER the fx goes above fx.max
        has_pohuai = False
        pohuai_bi = None
        for bi_idx in range(last_line_idx + 1, len(bis)):
            bi = bis[bi_idx]
            if bi.type == "up":
                if bi.high > curr.max:
                    has_pohuai = True
                    pohuai_bi = bi
                break
        
        end_bi = min(curr.lines, key=lambda l: l.low)
        end_bi_idx = end_bi.index
        if bis[end_bi_idx].type == "up" and end_bi_idx > 0:
            end_bi_idx -= 1
        
        status = "REJECTED (pohuai)" if has_pohuai else "ACCEPTED"
        print(f"  tzxl[{i}] DI fx: min={curr.min:.1f} max={curr.max:.1f} lines=bi{[l.index for l in curr.lines]}")
        print(f"    end_bi_idx={end_bi_idx}, span={end_bi_idx - start_bi_idx}")
        if has_pohuai:
            print(f"    pohuai: bi[{pohuai_bi.index}] h={pohuai_bi.high:.1f} > fx.max={curr.max:.1f}")
        print(f"    → {status}")
        if not has_pohuai and end_bi_idx - start_bi_idx >= 2:
            print(f"    → Final XD: down bi[{start_bi_idx}]→bi[{end_bi_idx}]")
            break
        print()
