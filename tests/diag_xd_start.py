"""Check how pyarmor determines the starting point for XD[0].
The first XD starts at bi[2], not bi[0]. Need to understand why."""
import sys; sys.path.insert(0, 'src')
import pandas as pd
from chanlun.cl_pyarmor import CL as CL_P

df = pd.read_parquet('tests/test_data/BTC_USDT_60m_500.parquet')
cd_p = CL_P('BTC/USDT', '60m')
cd_p.process_klines(df)
xds = cd_p.xds
bis = cd_p.bis

print("=== bi[0] to bi[2] details ===")
for i in range(min(5, len(bis))):
    bi = bis[i]
    print(f"  bi[{i}] {bi.type} h={bi.high:.1f} l={bi.low:.1f}")

print(f"\n=== XD[0] ===")
xd0 = xds[0]
print(f"  type={xd0.type} start=bi[{xd0.start_line.index}] end=bi[{xd0.end_line.index}]")
print(f"  start_line type={xd0.start_line.type}")

# The first XD is DOWN, starting at bi[2] which is a DOWN stroke
# bi[0] is UP (h=69230, l=68112.2→up to 70110.9)
# bi[1] is DOWN (not used for first XD start?)
# bi[2] is UP (third stroke)

# Wait - let me re-check. XD[0] is DOWN from bi[2] to bi[14].
# bi[2] is a DOWN stroke? No, let me verify.
print(f"\n  bi[{xd0.start_line.index}] type={xd0.start_line.type}")
print(f"  bi[{xd0.end_line.index}] type={xd0.end_line.type}")

# For a DOWN XD, start_line should be UP (the peak) or DOWN (the start)?
# Actually, DOWN XD goes from high to low.
# The start_line could be the UP stroke at the peak, or the first DOWN stroke.

# Let me check the TZXL[0] for XD[0]
print(f"\n=== XD[0] TZXL[0] ===")
t0 = xd0.tzxls[0]
print(f"  line=bi[{t0.line.index}] type={t0.line.type} h={t0.line.high:.1f} l={t0.line.low:.1f}")

# So TZXL[0] is bi[1] (UP stroke), which means the characteristic sequence starts from bi[1]
# And the XD starts at bi[2] (the DOWN stroke paired with bi[1])
# This means the starting point is bi[2] because bi[1] is the first element of the 
# characteristic sequence, and bi[2] is the corresponding DOWN stroke

# Actually, thinking about it differently:
# For a DOWN XD starting at bi[2]:
# - bi[2] is a DOWN stroke starting from the high of bi[1]
# - The characteristic sequence (UP strokes) starts at bi[1]
# - But wait, bi[1] is BEFORE bi[2]... 

# Let me check: does the DOWN XD start at the peak (high of bi[1]) or at bi[2]?
# In chan theory, a DOWN segment starts at the highest point.

# The issue might be about where the XD.start_line is set
print(f"\n=== XD start vs TZXL ===")
print(f"  XD start: bi[{xd0.start_line.index}] ({xd0.start_line.type})")
print(f"  TZXL first: bi[{xd0.tzxls[0].line.index}] ({xd0.tzxls[0].line.type})")

# So XD.start_line = bi[2] (DOWN stroke)
# But TZXL[0] = bi[1] (UP stroke - the one that forms the peak)
# The XD goes from the peak formed by bi[1] down to bi[14]
# In terms of bi indices: start_line=bi[2] means the XD starts counting from bi[2]
# But the peak price is at bi[1].high

# CRITICAL: The algorithm might start the first TZXL from bi[1] (the first reverse-direction stroke)
# Then bi[2] is a DOWN stroke (same direction as XD), so it's between TZXL elements
# So the starting bi for the XD search should be such that bi[1] is included

# Actually, let me check: in _build_xds, if we look at ALL strokes from bi[0]:
# bi[0] is DOWN, bi[1] is UP
# For first XD, if direction is DOWN:
#   - We need UP strokes for the characteristic sequence: bi[1], bi[3], bi[5], ...
#   - But bi[0] is already DOWN, so the XD starts at bi[0]?
# If direction is UP:
#   - We need DOWN strokes: bi[0], bi[2], bi[4], ...

# The key question: what determines the first XD direction?
print(f"\n=== First few BIs ===")
for i in range(4):
    print(f"  bi[{i}] {bis[i].type} h={bis[i].high:.1f} l={bis[i].low:.1f}")

# bi[0] is DOWN: h=69230.0 l=68112.2 
# bi[1] is UP: h=70110.9 l=68112.2
# So bi[1] goes HIGHER than bi[0]'s start.
# If we say the first direction is DOWN (following bi[0]):
#   Then characteristic sequence uses UP strokes starting from bi[1]
#   And the XD starts at bi[0] (the first same-direction stroke before bi[1])
# But pyarmor starts at bi[2]...

# Maybe the rule is: for a DOWN XD, start_line is the first DOWN stroke AFTER the first TZXL element?
# bi[1] is TZXL[0], so start_line = bi[2] (next DOWN stroke after bi[1])

# OR: maybe the first direction is determined differently.
# If bi[0] down to 68112, then bi[1] up to 70110 (new high) → 
# the real "top" is at bi[1], so DOWN segment starts from here
