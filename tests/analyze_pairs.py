"""Analyze the split validation patterns from trace data."""
# (di_clk, ding_clk, di_ki, ding_ki, passed_L2546, passed_L2547, result)
# For DOWN BI(191->213, start_ki=267, end_ki=298)
pairs = [
    # di  ding   di_ki ding_ki  L2543  L2546  L2547  result
    (208, 212,   291,  296,     True,  True,  False, "FAIL at L2547"),
    (208, 210,   291,  293,     True,  False, False, "FAIL at L2546"),
    (200, 203,   280,  283,     True,  False, False, "FAIL at L2546"),
    (200, 212,   280,  296,     True,  True,  False, "FAIL at L2547"),
    (200, 210,   280,  293,     True,  True,  True,  "FAIL (L2548 check)"),
    (200, 207,   280,  290,     True,  True,  True,  "FAIL (L2548 check)"),
    (206, 212,   288,  296,     True,  True,  False, "FAIL at L2547"),
    (206, 210,   288,  293,     True,  True,  True,  "FAIL (L2548 check)"),
    (206, 207,   288,  290,     True,  False, False, "FAIL at L2546"),
    (211, 212,   295,  296,     True,  False, False, "FAIL at L2546"),
    (195, 198,   271,  274,     True,  False, False, "FAIL at L2546"),
    (195, 203,   271,  283,     True,  True,  True,  "SUCCESS"),
]

start_ki = 267
end_ki = 298

print(f"{'di':>4} {'ding':>4}  {'di_ki':>5} {'dg_ki':>5}  "
      f"{'bi2_clk':>7} {'bi2_ki':>6} {'bi1_clk':>7} {'bi1_ki':>6} {'bi3_clk':>7} {'bi3_ki':>6}  "
      f"{'L2546':>5} {'L2547':>5}  result")
print("-" * 110)

for di, ding, di_ki, ding_ki, l2543, l2546, l2547, result in pairs:
    bi2_clk = ding - di          # middle sub-BI CLK distance
    bi2_ki = ding_ki - di_ki     # middle sub-BI raw K distance
    bi1_clk = di - 191           # first sub-BI CLK distance
    bi1_ki = di_ki - start_ki    # first sub-BI raw K distance
    bi3_clk = 213 - ding         # third sub-BI CLK distance
    bi3_ki = end_ki - ding_ki    # third sub-BI raw K distance
    
    l2546_s = "PASS" if l2546 else "FAIL"
    l2547_s = "PASS" if l2547 else "FAIL"
    
    print(f"{di:4d} {ding:4d}  {di_ki:5d} {ding_ki:5d}  "
          f"{bi2_clk:7d} {bi2_ki:6d} {bi1_clk:7d} {bi1_ki:6d} {bi3_clk:7d} {bi3_ki:6d}  "
          f"{l2546_s:>5} {l2547_s:>5}  {result}")

print()
print("L2546 pattern analysis (FAIL when bi2_clk <= 3):")
for di, ding, di_ki, ding_ki, l2543, l2546, l2547, result in pairs:
    bi2_clk = ding - di
    if not l2546:
        print(f"  FAIL: bi2_clk={bi2_clk} (di={di}, ding={ding})")
    else:
        print(f"  PASS: bi2_clk={bi2_clk} (di={di}, ding={ding})")

print()
print("L2547 pattern analysis (for those that passed L2546):")
for di, ding, di_ki, ding_ki, l2543, l2546, l2547, result in pairs:
    if not l2546:
        continue
    bi3_clk = 213 - ding
    bi3_ki = end_ki - ding_ki
    bi1_clk = di - 191
    bi1_ki = di_ki - start_ki
    if not l2547:
        print(f"  FAIL: bi3_clk={bi3_clk}, bi3_ki={bi3_ki}, bi1_clk={bi1_clk}, bi1_ki={bi1_ki} (di={di}, ding={ding})")
    else:
        print(f"  PASS: bi3_clk={bi3_clk}, bi3_ki={bi3_ki}, bi1_clk={bi1_clk}, bi1_ki={bi1_ki} (di={di}, ding={ding})")
