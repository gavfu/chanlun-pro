"""Deep analysis: what's different between bi[9] (bad) and bi[15] (not bad)?"""
import sys; sys.path.insert(0, 'src')
import pandas as pd
from chanlun.cl_open import CL as CL_O

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')
cd_o = CL_O('BTC/USDT', '60m')
cd_o.process_klines(df)
bis = cd_o.bis

# In a DOWN XD, characteristic sequence uses UP strokes
# For DOWN inclusion, max = min([l.high]), min = min([l.low])

# From the pyarmor result, comparing adjacent TZXL pairs:
# bi[7] → bi[9]: bi[9] gets line_bad=True
# bi[13] → bi[15]: bi[15] does NOT get line_bad

# Let me look at ALL the relationships between consecutive UP strokes

up_bis = [(bi.index, bi) for bi in bis if bi.type == "up"]

print("=== Consecutive UP stroke pair analysis ===")
print("For DOWN XD: bh_direction=down, max=min([h]), min=min([l])")
print()

for j in range(len(up_bis) - 1):
    idx_a, a = up_bis[j]
    idx_b, b = up_bis[j + 1]
    
    # raw values
    a_h, a_l = a.high, a.low
    b_h, b_l = b.high, b.low
    
    # Check containment on raw h/l
    a_c_b = a_h >= b_h and a_l <= b_l
    b_c_a = b_h >= a_h and b_l <= a_l
    
    if a_c_b or b_c_a:
        direction = "OLD⊃NEW" if a_c_b else "NEW⊃OLD"
        
        # Check the PRE_LINE relationship
        # pre_line for b is bis[b.index - 1] = the DOWN stroke before b
        pre_b = bis[b.index - 1]
        
        # Does the pre_line (down stroke before b) have containment with b?
        pre_c_b = pre_b.high >= b.high and pre_b.low <= b.low
        b_c_pre = b.high >= pre_b.high and b.low <= pre_b.low
        
        # pre_line for a
        pre_a = bis[a.index - 1]
        
        print(f"bi[{idx_a}] vs bi[{idx_b}]: {direction}")
        print(f"  a: h={a_h:.1f} l={a_l:.1f}")
        print(f"  b: h={b_h:.1f} l={b_l:.1f}")
        print(f"  pre_b (bi[{pre_b.index}] {pre_b.type}): h={pre_b.high:.1f} l={pre_b.low:.1f}")
        print(f"  pre_b contains b? {pre_c_b}")
        print(f"  b contains pre_b? {b_c_pre}")
        
        # Down stroke BEFORE a
        down_a = bis[a.index - 1]
        # Down stroke BEFORE b
        down_b = bis[b.index - 1]
        
        # Does the DOWN stroke extend the range?
        # For DOWN XD: we look at reverse-direction strokes (UP). 
        # The "line" is the UP stroke, "pre_line" is the DOWN stroke before it.
        # "line_bad" might be: does the UP stroke (line) contain its pre_line (the DOWN stroke)?
        
        print(f"  Line (UP bi[{idx_b}]) contains pre_line (DOWN bi[{pre_b.index}])?  {b_c_pre}")
        print(f"  Line (UP bi[{idx_a}]) contains pre_line (DOWN bi[{pre_a.index}])?  h:{a_h >= pre_a.high} l:{a_l <= pre_a.low}")
        
        # More: what about the relationship of the pre_lines?
        # pre_a (down before a) and pre_b (down before b)
        print(f"  Down bi[{pre_a.index}]: h={pre_a.high:.1f} l={pre_a.low:.1f}")
        print(f"  Down bi[{pre_b.index}]: h={pre_b.high:.1f} l={pre_b.low:.1f}")
        
        # Does b's pre_line contain a's pre_line?
        pre_a_c_pre_b = pre_a.high >= pre_b.high and pre_a.low <= pre_b.low
        pre_b_c_pre_a = pre_b.high >= pre_a.high and pre_b.low <= pre_a.low
        print(f"  pre_a ⊃ pre_b? {pre_a_c_pre_b}")
        print(f"  pre_b ⊃ pre_a? {pre_b_c_pre_a}")
        
        print()
