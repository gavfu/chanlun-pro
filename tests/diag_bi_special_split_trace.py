# -*- coding: utf-8 -*-
"""
诊断 pyarmor 特殊拆笔逻辑。

用途：
1. 复现 pyarmor `_bi_special_bi_split` 返回 True 的真实样本
2. 打印拆分前后的尾部笔列表
3. 抓取关键行上的局部变量，辅助还原 split 规则

运行：
    .venv/bin/python tests/diag_bi_special_split_trace.py
"""

import json
import os
import sys
from typing import Any, Dict, List, Tuple

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chanlun.cl_interface import BI, FX
from chanlun.cl_pyarmor import CL as CLPyarmor


CASES: List[Tuple[str, str, int, Tuple[int, int]]] = [
    ("ETH/USDT", "5m", 1000, (191, 195)),
    ("BTC/USDT", "60m", 1000, (254, 259)),
    ("BTC/USDT", "60m", 1000, (131, 139)),
]

SNAPSHOT_LINES = (2552,)


def summarize(value: Any) -> Any:
    if isinstance(value, FX):
        return ["FX", value.k.index, value.type, round(value.val, 4), value.k.k_index]
    if isinstance(value, BI):
        return ["BI", value.start.k.index, value.end.k.index, value.type]
    if isinstance(value, list):
        return ["list", len(value), [summarize(item) for item in value[:8]]]
    if isinstance(value, dict):
        return ["dict", len(value)]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return type(value).__name__


def run_case(symbol: str, frequency: str, limit: int, target: Tuple[int, int]) -> None:
    data_file = os.path.join(
        os.path.dirname(__file__),
        "test_data",
        f"{symbol.replace('/', '_')}_{frequency}_{limit}.parquet",
    )
    df = pd.read_parquet(data_file)

    original_special_split = CLPyarmor._bi_special_bi_split
    events: List[Dict[str, Any]] = []
    snapshots: Dict[int, Dict[str, Any]] = {}

    def tracer(frame, event, arg):
        if frame.f_code is not original_special_split.__code__:
            return tracer
        bi = frame.f_locals.get("bi")
        if bi is None or (bi.start.k.index, bi.end.k.index) != target:
            return tracer
        if event == "line" and frame.f_lineno in SNAPSHOT_LINES:
            snapshots[frame.f_lineno] = {
                key: summarize(val)
                for key, val in frame.f_locals.items()
                if key not in ("self", "__assert_armored__")
            }
        return tracer

    def wrapped_special_split(self, bi):
        before_tail = [(b.start.k.index, b.end.k.index) for b in self.bis[-8:]]
        result = original_special_split(self, bi)
        after_tail = [(b.start.k.index, b.end.k.index) for b in self.bis[-8:]]
        if bool(result) or (bi.start.k.index, bi.end.k.index) == target:
            events.append(
                {
                    "call": (bi.start.k.index, bi.end.k.index),
                    "result": bool(result),
                    "before_tail": before_tail,
                    "after_tail": after_tail,
                }
            )
        return result

    CLPyarmor._bi_special_bi_split = wrapped_special_split
    sys.settrace(tracer)
    try:
        cd = CLPyarmor(symbol, frequency)
        cd.process_klines(df)
    finally:
        sys.settrace(None)
        CLPyarmor._bi_special_bi_split = original_special_split

    print(f"\n=== {symbol} {frequency} target={target} ===")
    print("events:")
    print(json.dumps(events, ensure_ascii=False, indent=2))
    print("snapshots:")
    print(json.dumps(snapshots, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    for case in CASES:
        run_case(*case)