"""Test: use k_gap for CONFIRMATION gap check only, keep cl_gap for primary BI."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from chanlun.cl_open import CL as CL_O
from chanlun.cl_pyarmor import CL as CL_P
from chanlun.cl_interface import FX, BI, Config

CL_CONFIG = {
    "bi_type": "bi_type_old",
    "fx_qj": "fx_qj_k",
    "fx_qy": "fx_qy_three",
    "bi_fx_cgd": "bi_fx_cgd_yes",
    "fx_check_k_nums": 13,
    "bi_split_k_cross_nums": "20,1",
    "xd_bzh": "xd_bzh_no",
}

datasets = [
    ("BTCd", "tests/test_data/BTC_USDT_d_500.parquet"),
    ("ETH60", "tests/test_data/ETH_USDT_60m_1000.parquet"),
    ("BTC60", "tests/test_data/BTC_USDT_60m_1000.parquet"),
    ("BTC5m", "tests/test_data/BTC_USDT_5m_1000.parquet"),
    ("ETH5m", "tests/test_data/ETH_USDT_5m_1000.parquet"),
]

original_build_bis = CL_O._build_bis

def _bi_fx_valid_with_gap_type(self, start_fx, end_fx, use_k_gap=False):
    """Modified _bi_fx_valid that can use k_gap for gap check."""
    if start_fx.type == end_fx.type:
        return False
    cl_gap = end_fx.k.index - start_fx.k.index
    k_gap = end_fx.k.k_index - start_fx.k.k_index
    
    if self.bi_type == Config.BI_TYPE_DD.value:
        if cl_gap < 1: return False
    elif self.bi_type == Config.BI_TYPE_JDB.value:
        if k_gap < 4: return False
    else:
        if use_k_gap:
            if k_gap < 4: return False
        else:
            if cl_gap < 4: return False
    
    if k_gap < self.fx_check_k_nums:
        if self.allow_bi_fx_strict:
            qj = self.fx_qj
            qy = self.fx_qy
            if start_fx.type == "ding" and end_fx.type == "di":
                if start_fx.low(qj, qy) < end_fx.low(qj, qy): return False
                if end_fx.high(qj, qy) > start_fx.high(qj, qy): return False
            elif start_fx.type == "di" and end_fx.type == "ding":
                if start_fx.high(qj, qy) > end_fx.high(qj, qy): return False
                if end_fx.low(qj, qy) < start_fx.low(qj, qy): return False
    return True

def build_bis_confirm_k_gap(self, fxs):
    """_build_bis: primary uses cl_gap, confirmation uses k_gap."""
    bis = []
    if len(fxs) < 2: return bis
    start_fx = fxs[0]
    start_idx = 0
    end_fx = None
    end_idx = -1
    has_confirmed_bi = False
    i = 1
    while i < len(fxs):
        cur_fx = fxs[i]
        if end_fx is None:
            if cur_fx.type == start_fx.type:
                if not has_confirmed_bi:
                    if start_fx.type == "ding" and cur_fx.val > start_fx.val:
                        start_fx = cur_fx; start_idx = i
                    elif start_fx.type == "di" and cur_fx.val < start_fx.val:
                        start_fx = cur_fx; start_idx = i
            else:
                if _bi_fx_valid_with_gap_type(self, start_fx, cur_fx, use_k_gap=False):
                    end_fx = cur_fx; end_idx = i
            i += 1
        else:
            if cur_fx.type == end_fx.type:
                if end_fx.type == "di" and cur_fx.val <= end_fx.val:
                    if _bi_fx_valid_with_gap_type(self, start_fx, cur_fx, use_k_gap=False):
                        end_fx = cur_fx; end_idx = i
                elif end_fx.type == "ding" and cur_fx.val >= end_fx.val:
                    if _bi_fx_valid_with_gap_type(self, start_fx, cur_fx, use_k_gap=False):
                        end_fx = cur_fx; end_idx = i
                i += 1
            else:
                # CONFIRMATION: use k_gap for gap check
                confirm = _bi_fx_valid_with_gap_type(self, end_fx, cur_fx, use_k_gap=True)
                if confirm and self.bi_fx_cgd == Config.BI_FX_CHD_NO.value:
                    k_gap_confirm = cur_fx.k.k_index - end_fx.k.k_index
                    if k_gap_confirm < self.fx_check_k_nums:
                        for j in range(end_idx + 1, i):
                            mid_fx = fxs[j]
                            if mid_fx.type == cur_fx.type:
                                if cur_fx.type == "ding" and mid_fx.val > cur_fx.val:
                                    confirm = False; break
                                elif cur_fx.type == "di" and mid_fx.val < cur_fx.val:
                                    confirm = False; break
                if confirm:
                    bi_type = "down" if start_fx.type == "ding" else "up"
                    bi = BI(start=start_fx, end=end_fx, _type=bi_type,
                            index=len(bis), default_zs_type=self.default_bi_zs_type)
                    bis.append(bi)
                    has_confirmed_bi = True
                    start_fx = end_fx; start_idx = end_idx
                    end_fx = None; end_idx = -1
                    i = start_idx + 1
                else:
                    i += 1
    if end_fx is not None:
        bi_type = "down" if start_fx.type == "ding" else "up"
        bi = BI(start=start_fx, end=end_fx, _type=bi_type,
                index=len(bis), default_zs_type=self.default_bi_zs_type)
        bis.append(bi)
    return bis

print("=== Variant: confirmation uses k_gap, primary uses cl_gap ===")
for name, path in datasets:
    df = pd.read_parquet(path)
    cd_p = CL_P("test", "test", config=CL_CONFIG)
    cd_p.process_klines(df)
    n_p = len(cd_p.get_bis())
    
    CL_O._build_bis = build_bis_confirm_k_gap
    cd_o = CL_O("test", "test", config=CL_CONFIG)
    cd_o.process_klines(df)
    n_o = len(cd_o.get_bis())
    
    CL_O._build_bis = original_build_bis
    
    marker = "✅" if n_o == n_p else "❌"
    
    # Also check boundaries for matching-count cases
    if n_o == n_p:
        bis_o = cd_o.get_bis()
        bis_p2 = cd_p.get_bis()
        diffs = 0
        for bo, bp in zip(bis_o, bis_p2):
            if bo.start.k.k_index != bp.start.k.k_index or bo.end.k.k_index != bp.end.k.k_index:
                diffs += 1
        if diffs > 0:
            marker += f" ({diffs} boundary diffs)"
        else:
            marker += " PERFECT"
    
    print(f"  {name}: pyarmor={n_p}, open={n_o} {marker}")
