"""
Compare pre-split segments: current vs always-first for ETH60 and BTC5m.
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_interface import TZXL, BI

config = {
    "bi_type": "bi_type_old", "fx_qj": "fx_qj_k", "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes", "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1", "xd_bzh": "xd_bzh_no",
    "xd_zs_max_lines_split": 11, "xd_allow_split_no_highlow": 1,
    "xd_allow_split_zs_kz": 0, "xd_allow_split_zs_more_line": 1,
    "xd_allow_split_zs_no_direction": 1,
}

def patched_find_xd_end(self, bis, start_bi_idx, xd_type):
    """Same as original but with always-first rule"""
    tzxl_bi_type = "down" if xd_type == "up" else "up"
    bh_direction = "up" if xd_type == "up" else "down"
    target_fx_type = "ding" if xd_type == "up" else "di"

    tzxl_bis = []
    for i in range(start_bi_idx, len(bis)):
        if bis[i].type == tzxl_bi_type:
            tzxl_bis.append(bis[i])

    if len(tzxl_bis) < 3:
        return None

    tzxls = []
    for bi in tzxl_bis:
        pre_line = bis[bi.index - 1] if bi.index > 0 else bi
        done = bi.is_done()
        new_tzxl = TZXL(
            bh_direction=bh_direction, line=bi, pre_line=pre_line,
            line_bad=False, done=done,
        )
        if len(tzxls) == 0:
            tzxls.append(new_tzxl)
            continue
        last_tzxl = tzxls[-1]
        old_contains_new = last_tzxl.max >= new_tzxl.max and last_tzxl.min <= new_tzxl.min
        new_contains_old = new_tzxl.max >= last_tzxl.max and new_tzxl.min <= last_tzxl.min
        if old_contains_new:
            last_tzxl.lines.append(bi)
            last_tzxl.done = done
            last_tzxl.line_bad = False
            last_tzxl.update_maxmin()
        elif new_contains_old:
            new_tzxl.line_bad = True
            tzxls.append(new_tzxl)
        else:
            tzxls.append(new_tzxl)

    if len(tzxls) < 3:
        return None

    for i in range(1, len(tzxls) - 1):
        curr_xl = tzxls[i]
        prev_xl = tzxls[i - 1]
        next_xl = tzxls[i + 1]

        is_fx = False
        if target_fx_type == "ding":
            if curr_xl.max > prev_xl.max and curr_xl.max > next_xl.max:
                is_fx = True
        else:
            if curr_xl.min < prev_xl.min and curr_xl.min < next_xl.min:
                is_fx = True

        if is_fx:
            if not self._check_xd_bi_pohuai(bis, start_bi_idx, curr_xl, xd_type):
                result = self._build_xd_fx_result(
                    bis, start_bi_idx, xd_type, target_fx_type,
                    curr_xl, prev_xl, next_xl, tzxls,
                )
                if result is not None:
                    return result

    return None


# Capture pre-split segments by hooking _build_xds
original_build_xds = CL_Open._build_xds

def capturing_build_xds(self, bis, label=""):
    """Capture pre-split segments"""
    xds = original_build_xds(self, bis)
    if not hasattr(self, '_pre_split_xds'):
        self._pre_split_xds = []
    self._pre_split_xds = [(x.type, x.start_line.index, x.end_line.index) for x in xds]
    return xds


for name, file, freq in [("ETH60", "ETH_USDT_60m_1000.parquet", "60m"),
                           ("BTC5m", "BTC_USDT_5m_1000.parquet", "5m")]:
    df = pd.read_parquet(f"tests/test_data/{file}")
    
    # Current (unpatched)
    cl_curr = CL_Open(name+"_curr", freq, config)
    # Hook _split_xds to capture pre-split state
    orig_split = cl_curr._split_xds
    pre_split_curr = []
    def capture_pre_split_curr(xds, bis, cl=cl_curr):
        for x in xds:
            pre_split_curr.append((x.type, x.start_line.index, x.end_line.index))
        return orig_split(xds, bis)
    cl_curr._split_xds = capture_pre_split_curr
    cl_curr.process_klines(df)
    
    # Always-first (patched)
    cl_af = CL_Open(name+"_af", freq, config)
    cl_af._find_xd_end = lambda bis, start, xd_type, self=cl_af: patched_find_xd_end(self, bis, start, xd_type)
    orig_split_af = cl_af._split_xds
    pre_split_af = []
    def capture_pre_split_af(xds, bis, cl=cl_af):
        for x in xds:
            pre_split_af.append((x.type, x.start_line.index, x.end_line.index))
        return orig_split_af(xds, bis)
    cl_af._split_xds = capture_pre_split_af
    cl_af.process_klines(df)

    print(f"\n{'='*80}")
    print(f"  {name} PRE-SPLIT comparison")
    print(f"\n  Current pre-split ({len(pre_split_curr)} segments):")
    for i, (t, s, e) in enumerate(pre_split_curr):
        print(f"    [{i}] {t} bi[{s}→{e}]")
    print(f"\n  Always-first pre-split ({len(pre_split_af)} segments):")
    for i, (t, s, e) in enumerate(pre_split_af):
        print(f"    [{i}] {t} bi[{s}→{e}]")
