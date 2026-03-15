"""
测试修复2的影响（prev_xl.line_bad条件）：
检查在小数据集上是否破坏现有XD测试
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P

CL_CONFIG = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
}

# Count how many XD FX detections have prev_xl.line_bad=True in current passing tests
import unittest.mock as mock

prev_bad_hits = []
original_find_xd_end = CL_O._find_xd_end

def traced_find_xd_end(self, bis, start_bi_idx, xd_type):
    from chanlun.cl_interface import TZXL
    tzxl_bi_type = "down" if xd_type == "up" else "up"
    bh_direction = "up" if xd_type == "up" else "down"
    target_fx_type = "ding" if xd_type == "up" else "di"

    tzxl_bis = [b for b in bis[start_bi_idx:] if b.type == tzxl_bi_type]
    if len(tzxl_bis) < 3:
        return original_find_xd_end(self, bis, start_bi_idx, xd_type)

    tzxls = []
    for bi in tzxl_bis:
        pre_line = bis[bi.index - 1] if bi.index > 0 else bi
        new_tzxl = TZXL(bh_direction=bh_direction, line=bi, pre_line=pre_line,
                        line_bad=False, done=bi.is_done())
        if not tzxls:
            tzxls.append(new_tzxl)
            continue
        last = tzxls[-1]
        old_c_new = last.max >= new_tzxl.max and last.min <= new_tzxl.min
        new_c_old = new_tzxl.max >= last.max and new_tzxl.min <= last.min
        if old_c_new:
            last.lines.append(bi)
            last.done = bi.is_done()
            last.line_bad = False
            last.update_maxmin()
        elif new_c_old:
            new_tzxl.line_bad = True
            tzxls.append(new_tzxl)
        else:
            tzxls.append(new_tzxl)

    for i in range(1, len(tzxls)-1):
        curr = tzxls[i]
        prev = tzxls[i-1]
        nxt = tzxls[i+1]
        is_fx = (target_fx_type == "ding" and curr.max > prev.max and curr.max > nxt.max) or \
                (target_fx_type == "di" and curr.min < prev.min and curr.min < nxt.min)
        if is_fx:
            is_bad_old = curr.line_bad and i < 3
            is_bad_new = (curr.line_bad and i < 3) or prev.line_bad
            if is_bad_old != is_bad_new:
                prev_bad_hits.append({
                    'dataset': getattr(self, '_debug_dataset', '?'),
                    'start': start_bi_idx, 'xd_type': xd_type,
                    'i': i, 'prev_bad': prev.line_bad, 'curr_bad': curr.line_bad,
                    'old_is_bad': is_bad_old, 'new_is_bad': is_bad_new,
                })
    return original_find_xd_end(self, bis, start_bi_idx, xd_type)

SMALL_DATASETS = {
    "BTCd":  "tests/test_data/BTC_USDT_d_500.parquet",
    "ETH60": "tests/test_data/ETH_USDT_60m_1000.parquet",
    "BTC60": "tests/test_data/BTC_USDT_60m_1000.parquet",
    "BTC5m": "tests/test_data/BTC_USDT_5m_1000.parquet",
    "ETH5m": "tests/test_data/ETH_USDT_5m_1000.parquet",
}

# Temporarily patch
CL_O._find_xd_end = traced_find_xd_end

for name, path in SMALL_DATASETS.items():
    df = pd.read_parquet(path)
    cl_o = CL_O("test", "test", CL_CONFIG)
    cl_o._debug_dataset = name
    cl_p = CL_P("test", "test", CL_CONFIG)
    cl_o.process_klines(df)
    cl_p.process_klines(df)

CL_O._find_xd_end = original_find_xd_end

print(f"XD-level FX detections where prev_xl.line_bad=True: {len(prev_bad_hits)}")
for h in prev_bad_hits[:20]:
    print(f"  {h['dataset']} start={h['start']} {h['xd_type']} i={h['i']} "
          f"prev_bad={h['prev_bad']} curr_bad={h['curr_bad']} "
          f"old_bad={h['old_is_bad']} new_bad={h['new_is_bad']}")
