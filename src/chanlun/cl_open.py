# -*- coding: utf-8 -*-
"""
缠论核心计算 —— 开源实现 (cl_open.py)

替代 PyArmor 加密的 cl.py，实现 ICL 接口。
第一阶段：K线合并、分型、笔。
线段/中枢/买卖点/背驰 暂返回空列表。
"""
import datetime
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import talib

from chanlun.cl_interface import (
    BI,
    BC,
    CLKline,
    Config,
    FX,
    ICL,
    Kline,
    LINE,
    MMD,
    TZXL,
    XD,
    XLFX,
    ZS,
)


class CL(ICL):
    """
    缠论计算 —— 开源实现

    实现了：K线包含处理 → 分型识别 → 笔构建 → MACD 计算。
    线段 / 中枢 / 买卖点 / 背驰 暂未实现，返回空列表。
    """

    def __init__(
        self,
        code: str,
        frequency: str,
        config: Union[dict, None] = None,
        start_datetime: datetime.datetime = None,
    ):
        self.code = code
        self.frequency = frequency
        self.cl_config: dict = config if config is not None else {}
        self.start_datetime = start_datetime

        # ---- 读取配置 ----
        # K线类型
        self.kline_type = self.cl_config.get(
            "kline_type", Config.KLINE_TYPE_DEFAULT.value
        )
        # 分型区域
        self.fx_qy = self.cl_config.get("fx_qy", Config.FX_QY_THREE.value)
        # 分型区间
        self.fx_qj = self.cl_config.get("fx_qj", Config.FX_QJ_K.value)
        # 分型包含关系
        self.fx_bh = self.cl_config.get("fx_bh", Config.FX_BH_YES.value)
        # 笔类型
        self.bi_type = self.cl_config.get("bi_type", Config.BI_TYPE_OLD.value)
        # 笔标准化
        self.bi_bzh = self.cl_config.get("bi_bzh", Config.BI_BZH_YES.value)
        # 笔区间
        self.bi_qj = self.cl_config.get("bi_qj", Config.BI_QJ_DD.value)
        # 笔内分型次高低
        self.bi_fx_cgd = self.cl_config.get("bi_fx_cgd", Config.BI_FX_CHD_YES.value)
        # 分型检查K线数
        self.fx_check_k_nums: int = int(self.cl_config.get("fx_check_k_nums", 13))
        # 笔拆分K线跨越数
        bi_split_k_cross = self.cl_config.get("bi_split_k_cross_nums", "20,1")
        if isinstance(bi_split_k_cross, str):
            parts = bi_split_k_cross.split(",")
            self.bi_split_k_cross_nums = int(parts[0])
        else:
            self.bi_split_k_cross_nums = int(bi_split_k_cross)

        # MACD 参数
        self.macd_fast = int(self.cl_config.get("idx_macd_fast", 12))
        self.macd_slow = int(self.cl_config.get("idx_macd_slow", 26))
        self.macd_signal = int(self.cl_config.get("idx_macd_signal", 9))

        # 中枢默认类型 (目前只记录，不做计算)
        zs_bi_type = self.cl_config.get("zs_bi_type", [Config.ZS_TYPE_BZ.value])
        if isinstance(zs_bi_type, list):
            self.default_bi_zs_type = zs_bi_type[0] if zs_bi_type else Config.ZS_TYPE_BZ.value
        else:
            self.default_bi_zs_type = zs_bi_type

        # ---- 数据容器 ----
        self.src_klines: List[Kline] = []
        self.cl_klines: List[CLKline] = []
        self.fxs: List[FX] = []
        self.bis: List[BI] = []
        self.xds: List[XD] = []
        self.bi_zss: List[ZS] = []
        self.xd_zss: List[ZS] = []
        self.idx: dict = {"macd": {"dif": [], "dea": [], "hist": []}}

        # 内部追踪
        self._last_k_date = None  # 上一次处理到的最后 K 线日期

    # ----------------------------------------------------------------
    #  ICL 接口实现
    # ----------------------------------------------------------------

    def process_klines(self, klines: pd.DataFrame) -> "CL":
        if klines is None or len(klines) == 0:
            return self

        # ---------- 增量支持 ----------
        if self._last_k_date is not None:
            # 过滤掉已处理的 K 线 (保留最后一根以便更新)
            new_klines = klines[klines["date"] >= self._last_k_date]
            if len(new_klines) == 0:
                return self
            # 如果新数据的起始时间远大于已有数据末尾，说明时间不连续，全部重新计算
            if new_klines.iloc[0]["date"] > self._last_k_date:
                self._reset()
                new_klines = klines
        else:
            new_klines = klines

        # 是否为全量（首次）计算
        is_full = len(self.src_klines) == 0

        if is_full:
            self._full_compute(klines)
        else:
            self._incremental_compute(new_klines)

        return self

    def get_code(self) -> str:
        return self.code

    def get_frequency(self) -> str:
        return self.frequency

    def get_config(self) -> dict:
        return self.cl_config

    def get_src_klines(self) -> List[Kline]:
        return self.src_klines

    def get_klines(self) -> List[Kline]:
        """
        kline_type == kline_default  → 原始 K 线
        kline_type == kline_chanlun  → 缠论 K 线
        """
        if self.kline_type == Config.KLINE_TYPE_CHANLUN.value:
            return self.cl_klines  # type: ignore
        return self.src_klines

    def get_cl_klines(self) -> List[CLKline]:
        return self.cl_klines

    def get_idx(self) -> dict:
        return self.idx

    def get_fxs(self) -> List[FX]:
        return self.fxs

    def get_bis(self) -> List[BI]:
        return self.bis

    def get_xds(self) -> List[XD]:
        return self.xds

    def get_zsds(self) -> List[XD]:
        return []

    def get_qsds(self) -> List[XD]:
        return []

    def get_bi_zss(self, zs_type: str = None) -> List[ZS]:
        return self.bi_zss

    def get_xd_zss(self, zs_type: str = None) -> List[ZS]:
        return self.xd_zss

    def get_zsd_zss(self) -> List[ZS]:
        return []

    def get_qsd_zss(self) -> List[ZS]:
        return []

    def get_last_bi_zs(self) -> Union[ZS, None]:
        return self.bi_zss[-1] if self.bi_zss else None

    def get_last_xd_zs(self) -> Union[ZS, None]:
        return self.xd_zss[-1] if self.xd_zss else None

    def create_dn_zs(
        self,
        zs_type: str,
        lines: List[LINE],
        max_line_num: int = 999,
        zs_include_last_line=True,
    ) -> List[ZS]:
        """段内中枢 — 简易实现：取连续三段的重叠区域"""
        if len(lines) < 3:
            return []
        zss: List[ZS] = []
        i = 0
        while i + 2 < len(lines):
            highs = [l.high for l in lines[i : i + 3]]
            lows = [l.low for l in lines[i : i + 3]]
            zg = min(highs)
            zd = max(lows)
            if zg > zd:
                zs = ZS(
                    zs_type=zs_type or "bi",
                    start=lines[i].start,
                    end=lines[i + 2].end,
                    zg=zg,
                    zd=zd,
                    gg=max(highs),
                    dd=min(lows),
                    _type="up" if lines[i].type == "down" else "down",
                    index=len(zss),
                    line_num=3,
                    level=0,
                )
                zs.done = True
                zs.real = True
                for l in lines[i : i + 3]:
                    zs.add_line(l)
                # 尝试延伸
                j = i + 3
                while j < len(lines) and j - i < max_line_num:
                    if lines[j].high >= zd and lines[j].low <= zg:
                        zs.add_line(lines[j])
                        zs.end = lines[j].end
                        zs.line_num += 1
                        zs.gg = max(zs.gg, lines[j].high)
                        zs.dd = min(zs.dd, lines[j].low)
                        j += 1
                    else:
                        break
                zss.append(zs)
                i = j
            else:
                i += 1
        return zss

    def beichi_pz(self, zs: ZS, now_line: LINE) -> Tuple[bool, Union[LINE, None]]:
        """盘整背驰 — 简易实现"""
        if len(zs.lines) < 1:
            return False, None
        # 找到中枢进入段
        enter_line = zs.lines[0]
        if enter_line.type != now_line.type:
            if len(zs.lines) >= 2:
                enter_line = zs.lines[1]
            else:
                return False, None
        if enter_line.type != now_line.type:
            return False, None
        # 比较 MACD 力度
        enter_ld = now_line.get_ld(self)
        now_ld = now_line.get_ld(self)
        from chanlun.cl_interface import compare_ld_beichi

        bc = compare_ld_beichi(enter_ld, now_ld, now_line.type)
        return bc, enter_line

    def beichi_qs(
        self, lines: List[LINE], zss: List[ZS], now_line: LINE
    ) -> Tuple[bool, List[LINE]]:
        """趋势背驰 — 暂未实现"""
        return False, []

    def zss_is_qs(self, one_zs: ZS, two_zs: ZS) -> Union[str, None]:
        """判断两个中枢是否形成趋势"""
        if two_zs.dd > one_zs.gg:
            return "up"
        if two_zs.gg < one_zs.dd:
            return "down"
        return None

    # ----------------------------------------------------------------
    #  内部实现
    # ----------------------------------------------------------------

    def _reset(self):
        self.src_klines = []
        self.cl_klines = []
        self.fxs = []
        self.bis = []
        self.xds = []
        self.bi_zss = []
        self.xd_zss = []
        self.idx = {"macd": {"dif": [], "dea": [], "hist": []}}
        self._last_k_date = None

    def _full_compute(self, klines: pd.DataFrame):
        """全量计算"""
        self._reset()

        # 1) 构建原始 K 线
        self.src_klines = self._build_src_klines(klines)
        if len(self.src_klines) == 0:
            return

        # 2) K 线包含处理 → 缠论 K 线
        self.cl_klines = self._merge_klines(self.src_klines)

        # 3) 分型识别
        self.fxs = self._find_fxs(self.cl_klines)

        # 4) 笔构建
        self.bis = self._build_bis(self.fxs)

        # 5) 计算笔的高低点
        self._update_bi_highlow()

        # 6) 简易中枢
        self._build_bi_zss()

        # 7) 计算 MACD
        self._compute_macd()

        # 8) 记录最后 K 线日期
        self._last_k_date = self.src_klines[-1].date

    def _incremental_compute(self, new_klines: pd.DataFrame):
        """增量计算 — 简单实现：全量重算"""
        # 合并新旧 K 线数据，用 date 去重
        old_dates = {k.date for k in self.src_klines}
        new_rows = new_klines[~new_klines["date"].isin(old_dates)]
        if len(new_rows) == 0:
            # 可能只是最后一根更新
            if len(new_klines) > 0:
                last_new = new_klines.iloc[-1]
                last_new_date = last_new["date"]
                # 更新最后一根 K 线
                for k in reversed(self.src_klines):
                    if k.date == last_new_date:
                        k.h = float(last_new["high"])
                        k.l = float(last_new["low"])
                        k.o = float(last_new["open"])
                        k.c = float(last_new["close"])
                        k.a = float(last_new.get("volume", 0))
                        break

        # 简单策略：用所有已有 K 线 + 新 K 线重新全量计算
        all_dates_data = []
        for k in self.src_klines:
            all_dates_data.append(
                {
                    "date": k.date,
                    "open": k.o,
                    "high": k.h,
                    "low": k.l,
                    "close": k.c,
                    "volume": k.a,
                    "code": self.code,
                }
            )
        for _, row in new_rows.iterrows():
            all_dates_data.append(
                {
                    "date": row["date"],
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row.get("volume", 0),
                    "code": self.code,
                }
            )
        df = pd.DataFrame(all_dates_data)
        df = df.drop_duplicates(subset=["date"], keep="last")
        df = df.sort_values("date").reset_index(drop=True)
        self._full_compute(df)

    # ---- 构建原始 K 线 ----
    def _build_src_klines(self, klines: pd.DataFrame) -> List[Kline]:
        src = []
        for i, row in klines.iterrows():
            k = Kline(
                index=len(src),
                date=row["date"],
                h=float(row["high"]),
                l=float(row["low"]),
                o=float(row["open"]),
                c=float(row["close"]),
                a=float(row.get("volume", 0)),
            )
            src.append(k)
        return src

    # ---- K 线包含处理 ----
    def _merge_klines(self, src_klines: List[Kline]) -> List[CLKline]:
        """
        缠论 K 线合并（包含处理）

        规则：
        - 两根相邻 K 线，若一根的高低完全包含另一根 (h1>=h2 且 l1<=l2)，则合并。
        - 合并方向取决于之前的趋势：
          向上 → 取高高低高 (max(h), max(l))
          向下 → 取低高低低 (min(h), min(l))
        """
        if len(src_klines) == 0:
            return []

        merged: List[CLKline] = []

        for k in src_klines:
            ck = CLKline(
                k_index=k.index,
                date=k.date,
                h=k.h,
                l=k.l,
                o=k.o,
                c=k.c,
                a=k.a,
                klines=[k],
                index=len(merged),
                _n=1,
                _q=False,
            )

            if len(merged) == 0:
                merged.append(ck)
                continue

            last = merged[-1]

            # 判断包含关系
            if self._is_contain(last, ck):
                # 确定合并方向
                if len(merged) >= 2:
                    prev = merged[-2]
                    direction = "up" if last.h >= prev.h else "down"
                else:
                    direction = "up"

                # 合并
                if direction == "up":
                    new_h = max(last.h, ck.h)
                    new_l = max(last.l, ck.l)
                else:
                    new_h = min(last.h, ck.h)
                    new_l = min(last.l, ck.l)

                last.h = new_h
                last.l = new_l
                last.n += 1
                last.klines.extend(ck.klines)
                # 更新日期为最后一根原始K线的日期
                last.date = ck.date
                last.k_index = ck.k_index
            else:
                # 判断是否有缺口
                if ck.l > last.h or ck.h < last.l:
                    ck.q = True
                ck.index = len(merged)
                # 记录合并前趋势方向
                ck.up_qs = "up" if ck.h > last.h else "down"
                merged.append(ck)

        return merged

    def _is_contain(self, a: CLKline, b: CLKline) -> bool:
        """判断两根 K 线是否存在包含关系"""
        return (a.h >= b.h and a.l <= b.l) or (b.h >= a.h and b.l <= a.l)

    # ---- 分型识别 ----
    def _find_fxs(self, cl_klines: List[CLKline]) -> List[FX]:
        """
        在缠论 K 线序列中识别顶底分型

        顶分型：中间 K 线的高点是三根中最高的，低点也是三根中最高的。
        底分型：中间 K 线的低点是三根中最低的，高点也是三根中最低的。
        """
        fxs: List[FX] = []
        if len(cl_klines) < 3:
            return fxs

        for i in range(1, len(cl_klines) - 1):
            prev = cl_klines[i - 1]
            curr = cl_klines[i]
            nxt = cl_klines[i + 1]

            # 顶分型
            if curr.h > prev.h and curr.h > nxt.h and curr.l > prev.l and curr.l > nxt.l:
                fx = FX(
                    _type="ding",
                    k=curr,
                    klines=[prev, curr, nxt],
                    val=curr.h,
                    index=len(fxs),
                    done=True,
                )
                fxs.append(fx)
            # 底分型
            elif curr.l < prev.l and curr.l < nxt.l and curr.h < prev.h and curr.h < nxt.h:
                fx = FX(
                    _type="di",
                    k=curr,
                    klines=[prev, curr, nxt],
                    val=curr.l,
                    index=len(fxs),
                    done=True,
                )
                fxs.append(fx)

        # 处理最后一个未完成分型
        if len(cl_klines) >= 2:
            last_two = cl_klines[-2:]
            if len(fxs) > 0:
                last_fx = fxs[-1]
                last_ck = cl_klines[-1]
                second_last_ck = cl_klines[-2]
                # 检查是否形成未完成分型
                if last_fx.type == "di":
                    # 底分型后找顶
                    if second_last_ck.h > cl_klines[-3].h if len(cl_klines) >= 3 else False:
                        if last_ck.h < second_last_ck.h:
                            fx = FX(
                                _type="ding",
                                k=second_last_ck,
                                klines=[cl_klines[-3], second_last_ck, last_ck],
                                val=second_last_ck.h,
                                index=len(fxs),
                                done=True,
                            )
                            # 已经在主循环中处理，这里不重复添加

        # 过滤分型：确保顶底交替
        fxs = self._filter_fxs(fxs)

        return fxs

    def _filter_fxs(self, fxs: List[FX]) -> List[FX]:
        """
        过滤分型，确保顶底交替出现。

        连续同类型分型：
        - 连续顶分型保留最高的
        - 连续底分型保留最低的
        同时检查分型包含关系配置。
        """
        if len(fxs) <= 1:
            return fxs

        filtered: List[FX] = [fxs[0]]
        for fx in fxs[1:]:
            last = filtered[-1]

            if fx.type == last.type:
                # 同类型分型：保留极值更大/更小的
                if fx.type == "ding" and fx.val > last.val:
                    filtered[-1] = fx
                elif fx.type == "di" and fx.val < last.val:
                    filtered[-1] = fx
            else:
                # 顶底交替，但需要检查有效性
                if last.type == "ding" and fx.type == "di":
                    # 底不能高于顶
                    if fx.val >= last.val:
                        continue
                elif last.type == "di" and fx.type == "ding":
                    # 顶不能低于底
                    if fx.val <= last.val:
                        continue

                # 检查分型包含关系
                if not self._check_fx_bh(last, fx):
                    continue

                filtered.append(fx)

        # 重建索引
        for i, fx in enumerate(filtered):
            fx.index = i

        return filtered

    def _check_fx_bh(self, last_fx: FX, new_fx: FX) -> bool:
        """
        检查分型包含关系，根据 fx_bh 配置决定是否接受。
        """
        if self.fx_bh == Config.FX_BH_YES.value:
            return True  # 不判断，接受所有

        qj = self.fx_qj
        qy = self.fx_qy
        last_high = last_fx.high(qj, qy)
        last_low = last_fx.low(qj, qy)
        new_high = new_fx.high(qj, qy)
        new_low = new_fx.low(qj, qy)

        # 前包含后
        q_bao_h = last_high >= new_high and last_low <= new_low
        # 后包含前
        h_bao_q = new_high >= last_high and new_low <= last_low

        if self.fx_bh == Config.FX_BH_NO.value:
            return not q_bao_h and not h_bao_q
        elif self.fx_bh == Config.FX_BH_NO_QBH.value:
            return not q_bao_h
        elif self.fx_bh == Config.FX_BH_NO_HBQ.value:
            return not h_bao_q
        elif self.fx_bh == Config.FX_BH_DINGDI.value:
            # 顶不可在底中
            if last_fx.type == "ding" and new_fx.type == "di":
                return not h_bao_q
            return True
        elif self.fx_bh == Config.FX_BH_DIDING.value:
            # 底不可在顶中
            if last_fx.type == "di" and new_fx.type == "ding":
                return not h_bao_q
            return True

        return True

    # ---- 笔构建 ----
    def _build_bis(self, fxs: List[FX]) -> List[BI]:
        """
        根据分型序列构建笔。

        老笔规则（BI_TYPE_OLD）：
        - 顶底之间至少有一根独立 K 线（即顶底分型不共用 K 线，中间至少 1 根 K 线）。
        - 顶到底为向下笔，底到顶为向上笔。

        新笔规则（BI_TYPE_NEW）：
        - 顶底之间至少有一根独立 K 线。
        - 新的顶分型的最高点必须高于旧的顶分型，或者新的底分型的最低点低于旧的底分型。

        简单笔规则（BI_TYPE_JDB）：
        - 顶底交替即可成笔，不要求独立 K 线。

        顶底成笔（BI_TYPE_DD）：
        - 和老笔一样，但不要求独立K线只要顶底不共享同一根K线即可。
        """
        bis: List[BI] = []
        if len(fxs) < 2:
            return bis

        # 根据笔类型选择最小间隔
        if self.bi_type == Config.BI_TYPE_JDB.value:
            min_gap = 0  # 简单笔：无间隔要求
        elif self.bi_type == Config.BI_TYPE_DD.value:
            min_gap = 0  # 顶底成笔：顶底K线不同即可
        else:
            min_gap = 1  # 老笔/新笔：至少1根独立K线

        start_fx = fxs[0]
        for i in range(1, len(fxs)):
            end_fx = fxs[i]

            # 检查类型交替
            if start_fx.type == end_fx.type:
                # 同类型时保留更好的
                if start_fx.type == "ding" and end_fx.val > start_fx.val:
                    start_fx = end_fx
                elif start_fx.type == "di" and end_fx.val < start_fx.val:
                    start_fx = end_fx
                continue

            # 检查间隔（缠论K线数量）
            gap = end_fx.k.index - start_fx.k.index - 1
            if self.bi_type == Config.BI_TYPE_DD.value:
                # 顶底成笔：只要不是同一根缠论K线
                if end_fx.k.index <= start_fx.k.index:
                    continue
            elif gap < min_gap:
                continue

            # 老笔规则：顶底之间至少 4 根独立 K 线（含分型共用的）
            if self.bi_type in (Config.BI_TYPE_OLD.value, Config.BI_TYPE_NEW.value):
                k_count = end_fx.k.k_index - start_fx.k.k_index
                if k_count < 4:
                    continue

            # 创建笔
            bi_type = "down" if start_fx.type == "ding" else "up"
            bi = BI(
                start=start_fx,
                end=end_fx,
                _type=bi_type,
                index=len(bis),
                default_zs_type=self.default_bi_zs_type,
            )
            bis.append(bi)
            start_fx = end_fx

        return bis

    def _update_bi_highlow(self):
        """更新每笔的 high/low 值"""
        for bi in self.bis:
            if bi.type == "up":
                bi.high = bi.end.val
                bi.low = bi.start.val
            else:
                bi.high = bi.start.val
                bi.low = bi.end.val

            # 根据配置调整区间
            if self.bi_qj == Config.BI_QJ_CK.value:
                # 使用缠论K线最高最低
                ck_start = bi.start.k.index
                ck_end = bi.end.k.index
                for ck in self.cl_klines[ck_start : ck_end + 1]:
                    bi.high = max(bi.high, ck.h)
                    bi.low = min(bi.low, ck.l)
            elif self.bi_qj == Config.BI_QJ_K.value:
                # 使用原始K线最高最低
                k_start = bi.start.k.k_index
                k_end = bi.end.k.k_index
                for k in self.src_klines[k_start : k_end + 1]:
                    bi.high = max(bi.high, k.h)
                    bi.low = min(bi.low, k.l)

            # 笔标准化
            if self.bi_bzh == Config.BI_BZH_YES.value:
                if bi.type == "up":
                    bi.start.val = bi.low
                    bi.end.val = bi.high
                else:
                    bi.start.val = bi.high
                    bi.end.val = bi.low

    # ---- 中枢构建 ----
    def _build_bi_zss(self):
        """根据笔列表构建笔中枢"""
        self.bi_zss = self.create_dn_zs("bi", self.bis)

    # ---- MACD 计算 ----
    def _compute_macd(self):
        """使用 talib 计算 MACD"""
        if len(self.src_klines) == 0:
            return

        closes = np.array([k.c for k in self.src_klines], dtype=float)

        if len(closes) < self.macd_slow:
            # 数据不足，用 0 填充
            n = len(closes)
            self.idx = {
                "macd": {
                    "dif": [0.0] * n,
                    "dea": [0.0] * n,
                    "hist": [0.0] * n,
                }
            }
            return

        dif, dea, hist = talib.MACD(
            closes,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal,
        )

        # talib 返回 np.nan 的部分替换为 0
        dif = np.nan_to_num(dif, nan=0.0)
        dea = np.nan_to_num(dea, nan=0.0)
        hist = np.nan_to_num(hist, nan=0.0)

        self.idx = {
            "macd": {
                "dif": dif.tolist(),
                "dea": dea.tolist(),
                "hist": hist.tolist(),
            }
        }
