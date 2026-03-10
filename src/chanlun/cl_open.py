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
            self.bi_split_k_cross_tolerance = int(parts[1]) if len(parts) > 1 else 1
        else:
            self.bi_split_k_cross_nums = int(bi_split_k_cross)
            self.bi_split_k_cross_tolerance = 1

        # 分型严格处理
        self.allow_bi_fx_strict = int(self.cl_config.get("allow_bi_fx_strict", 1))
        # K线缺口
        self.kline_qk = self.cl_config.get("kline_qk", Config.KLINE_QK_NONE.value)

        # MACD 参数
        self.macd_fast = int(self.cl_config.get("idx_macd_fast", 12))
        self.macd_slow = int(self.cl_config.get("idx_macd_slow", 26))
        self.macd_signal = int(self.cl_config.get("idx_macd_signal", 9))

        # ---- 线段配置 ----
        self.xd_qj = self.cl_config.get("xd_qj", Config.XD_QJ_DD.value)
        self.xd_allow_bi_pohuai = self.cl_config.get(
            "xd_allow_bi_pohuai", Config.XD_BI_POHUAI_YES.value
        )
        self.xd_allow_split_no_highlow = int(
            self.cl_config.get("xd_allow_split_no_highlow", 1)
        )
        self.xd_allow_split_zs_kz = int(
            self.cl_config.get("xd_allow_split_zs_kz", 0)
        )
        self.xd_allow_split_zs_more_line = int(
            self.cl_config.get("xd_allow_split_zs_more_line", 1)
        )
        self.xd_allow_split_zs_no_direction = int(
            self.cl_config.get("xd_allow_split_zs_no_direction", 1)
        )
        self.xd_zs_max_lines_split = int(
            self.cl_config.get("xd_zs_max_lines_split", 11)
        )

        # ---- 走势段配置 ----
        self.zsd_qj = self.cl_config.get("zsd_qj", Config.ZSD_QJ_DD.value)

        # ---- 中枢配置 ----
        zs_bi_type = self.cl_config.get("zs_bi_type", [Config.ZS_TYPE_BZ.value])
        if isinstance(zs_bi_type, str):
            zs_bi_type = [zs_bi_type]
        self.zs_bi_type: List[str] = zs_bi_type
        self.default_bi_zs_type = self.zs_bi_type[0] if self.zs_bi_type else Config.ZS_TYPE_BZ.value

        zs_xd_type = self.cl_config.get("zs_xd_type", [Config.ZS_TYPE_BZ.value])
        if isinstance(zs_xd_type, str):
            zs_xd_type = [zs_xd_type]
        self.zs_xd_type: List[str] = zs_xd_type
        self.default_xd_zs_type = self.zs_xd_type[0] if self.zs_xd_type else Config.ZS_TYPE_BZ.value

        self.zs_qj = self.cl_config.get("zs_qj", Config.ZS_QJ_DD.value)
        self.zs_cd = self.cl_config.get("zs_cd", Config.ZS_CD_THREE.value)
        self.zs_wzgx = self.cl_config.get("zs_wzgx", Config.ZS_WZGX_ZGGDD.value)

        # ---- 买卖点开关 ----
        self.cl_mmd_cal_qs_1mmd = int(self.cl_config.get("cl_mmd_cal_qs_1mmd", 1))
        self.cl_mmd_cal_not_qs_3mmd_1mmd = int(self.cl_config.get("cl_mmd_cal_not_qs_3mmd_1mmd", 1))
        self.cl_mmd_cal_qs_3mmd_1mmd = int(self.cl_config.get("cl_mmd_cal_qs_3mmd_1mmd", 1))
        self.cl_mmd_cal_qs_not_lh_2mmd = int(self.cl_config.get("cl_mmd_cal_qs_not_lh_2mmd", 1))
        self.cl_mmd_cal_qs_bc_2mmd = int(self.cl_config.get("cl_mmd_cal_qs_bc_2mmd", 1))
        self.cl_mmd_cal_3mmd_not_lh_bc_2mmd = int(self.cl_config.get("cl_mmd_cal_3mmd_not_lh_bc_2mmd", 1))
        self.cl_mmd_cal_1mmd_not_lh_2mmd = int(self.cl_config.get("cl_mmd_cal_1mmd_not_lh_2mmd", 1))
        self.cl_mmd_cal_3mmd_xgxd_not_bc_2mmd = int(self.cl_config.get("cl_mmd_cal_3mmd_xgxd_not_bc_2mmd", 1))
        self.cl_mmd_cal_not_in_zs_3mmd = int(self.cl_config.get("cl_mmd_cal_not_in_zs_3mmd", 1))
        self.cl_mmd_cal_not_in_zs_gt_9_3mmd = int(self.cl_config.get("cl_mmd_cal_not_in_zs_gt_9_3mmd", 1))

        # ---- 数据容器 ----
        self.src_klines: List[Kline] = []
        self.cl_klines: List[CLKline] = []
        self.fxs: List[FX] = []
        self.bis: List[BI] = []
        self.xds: List[XD] = []
        self.zsds: List[XD] = []
        self.qsds: List[XD] = []
        self.bi_zss: List[ZS] = []
        self.xd_zss: List[ZS] = []
        self.zsd_zss: List[ZS] = []
        self.qsd_zss: List[ZS] = []
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
        return self.zsds

    def get_qsds(self) -> List[XD]:
        return self.qsds

    def get_bi_zss(self, zs_type: str = None) -> List[ZS]:
        if zs_type is None:
            return self.bi_zss
        return [zs for zs in self.bi_zss if zs.zs_type == zs_type]

    def get_xd_zss(self, zs_type: str = None) -> List[ZS]:
        if zs_type is None:
            return self.xd_zss
        return [zs for zs in self.xd_zss if zs.zs_type == zs_type]

    def get_zsd_zss(self) -> List[ZS]:
        return self.zsd_zss

    def get_qsd_zss(self) -> List[ZS]:
        return self.qsd_zss

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
        """盘整背驰：中枢进入段 vs 离开段力度比较"""
        if len(zs.lines) < 1:
            return False, None
        # 找到中枢进入段（与 now_line 方向相同的第一段）
        enter_line = None
        for line in zs.lines:
            if line.type == now_line.type:
                enter_line = line
                break
        if enter_line is None:
            return False, None
        # 比较 MACD 力度
        enter_ld = enter_line.get_ld(self)
        now_ld = now_line.get_ld(self)
        from chanlun.cl_interface import compare_ld_beichi

        bc = compare_ld_beichi(enter_ld, now_ld, now_line.type)
        return bc, enter_line

    def beichi_qs(
        self, lines: List[LINE], zss: List[ZS], now_line: LINE
    ) -> Tuple[bool, List[LINE]]:
        """趋势背驰：最后两个同向不重叠中枢，最后一段力度 vs 前一中枢后离开段"""
        if len(zss) < 2:
            return False, []
        # 取最后两个中枢，检查是否构成趋势
        last_zs = zss[-1]
        prev_zs = zss[-2]
        qs_dir = self.zss_is_qs(prev_zs, last_zs)
        if qs_dir is None:
            return False, []
        # 方向需要和 now_line 一致
        if now_line.type != qs_dir:
            return False, []
        # 找前一中枢的离开段（与 now_line 同方向）
        compare_line = None
        for line in reversed(lines):
            if line.index >= last_zs.lines[0].index:
                continue
            if line.type == now_line.type:
                compare_line = line
                break
        if compare_line is None:
            return False, []
        from chanlun.cl_interface import compare_ld_beichi

        bc = compare_ld_beichi(
            compare_line.get_ld(self), now_line.get_ld(self), now_line.type
        )
        return bc, [compare_line]

    def zss_is_qs(self, one_zs: ZS, two_zs: ZS) -> Union[str, None]:
        """判断两个中枢是否形成趋势，根据 zs_wzgx 配置"""
        if self.zs_wzgx == Config.ZS_WZGX_ZGD.value:
            # 宽松：用 zg/zd 比较
            if two_zs.zd > one_zs.zg:
                return "up"
            if two_zs.zg < one_zs.zd:
                return "down"
        elif self.zs_wzgx == Config.ZS_WZGX_ZGGDD.value:
            # 较为宽松：当前 zg/zd 与前一个 gg/dd 比较
            if two_zs.zd > one_zs.gg:
                return "up"
            if two_zs.zg < one_zs.dd:
                return "down"
        else:
            # 严格：用 gg/dd 比较 (默认 ZS_WZGX_GD)
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
        self.zsds = []
        self.qsds = []
        self.bi_zss = []
        self.xd_zss = []
        self.zsd_zss = []
        self.qsd_zss = []
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

        # 4.5) 笔拆分（重叠K线过多时拆分长笔）
        self.bis = self._bi_special_bi_split(self.bis)

        # 5) 计算笔的高低点
        self._update_bi_highlow()

        # 6) 计算 MACD（需要在背驰/买卖点计算之前）
        self._compute_macd()

        # 7) 笔中枢 + 笔背驰 + 笔买卖点
        self._build_bi_zss()
        self._calc_bi_bc()
        self._calc_bi_mmd()

        # 8) 线段构建
        self.xds = self._build_xds(self.bis)
        self._update_xd_highlow()

        # 9) 线段中枢 + 线段背驰 + 线段买卖点
        self._build_xd_zss()
        self._calc_xd_bc()
        self._calc_xd_mmd()

        # 10) 记录最后 K 线日期
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
                _n=0,
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
                    # 第一段包含时，用当前与上一根高点关系推断方向
                    direction = "up" if ck.h > last.h else "down"

                # 合并
                if direction == "up":
                    new_h = max(last.h, ck.h)
                    new_l = max(last.l, ck.l)
                    # k_index 取提供最高 h 的原始K线（与pyarmor一致）
                    if ck.h > last.h:
                        last.k_index = ck.k_index
                else:
                    new_h = min(last.h, ck.h)
                    new_l = min(last.l, ck.l)
                    # k_index 取提供最低 l 的原始K线（与pyarmor一致）
                    if ck.l < last.l:
                        last.k_index = ck.k_index

                last.h = new_h
                last.l = new_l
                last.n += 1
                last.klines.extend(ck.klines)
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

        # 处理最后一个未完成分型（pending fractal at end of data）
        # 如果最后一根缠论 K 线相对前一根继续朝远离上一个分型的方向运动，
        # 则在最后一根 K 线处创建一个 done=False 的未完成分型。
        if len(cl_klines) >= 2 and len(fxs) > 0:
            last_ck = cl_klines[-1]
            prev_ck = cl_klines[-2]
            last_fx = fxs[-1]
            if last_fx.type == "di":
                # 底分型后，最后K线高点 > 前一根高点 → 潜在顶分型
                if last_ck.h > prev_ck.h and last_ck.l > prev_ck.l:
                    fx = FX(
                        _type="ding",
                        k=last_ck,
                        klines=[prev_ck, last_ck, None],
                        val=last_ck.h,
                        index=len(fxs),
                        done=False,
                    )
                    fxs.append(fx)
            elif last_fx.type == "ding":
                # 顶分型后，最后K线低点 < 前一根低点 → 潜在底分型
                if last_ck.l < prev_ck.l and last_ck.h < prev_ck.h:
                    fx = FX(
                        _type="di",
                        k=last_ck,
                        klines=[prev_ck, last_ck, None],
                        val=last_ck.l,
                        index=len(fxs),
                        done=False,
                    )
                    fxs.append(fx)

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

        核心逻辑（老笔/新笔）：
        1. 从 start_fx 开始，找到第一个满足成笔条件的反向分型作为候选 end_fx
        2. 候选 end_fx 之后，继续扫描后续分型：
           a. 同类型且更优（更低底/更高顶） → 替换 end_fx（延伸当前笔）
           b. 反类型且与 end_fx 满足成笔条件 → 确认当前笔，end_fx 成为新 start_fx
           c. 反类型但不满足成笔条件 → 继续扫描（可能后面有更优的 end_fx）
           d. 同类型但不如当前 end_fx → 忽略
        3. 如果扫描结束都没确认，则以最后的 end_fx 作为当前笔的终点
        """
        bis: List[BI] = []
        if len(fxs) < 2:
            return bis

        start_fx = fxs[0]
        start_idx = 0  # start_fx 在 fxs 中的索引
        end_fx = None
        end_idx = -1
        has_confirmed_bi = False

        i = 1
        while i < len(fxs):
            cur_fx = fxs[i]

            if end_fx is None:
                # 还没有候选 end_fx
                if cur_fx.type == start_fx.type:
                    # 同类型：仅在首笔确认前更新 start_fx 为更优的
                    if not has_confirmed_bi:
                        if start_fx.type == "ding" and cur_fx.val > start_fx.val:
                            start_fx = cur_fx
                            start_idx = i
                        elif start_fx.type == "di" and cur_fx.val < start_fx.val:
                            start_fx = cur_fx
                            start_idx = i
                else:
                    # 反类型：检查成笔条件
                    if self._bi_fx_valid(start_fx, cur_fx):
                        end_fx = cur_fx
                        end_idx = i
                i += 1
            else:
                # 已有候选 end_fx
                if cur_fx.type == end_fx.type:
                    # 同类型分型（和 end_fx 同类）
                    if end_fx.type == "di" and cur_fx.val <= end_fx.val:
                        # 更低/等低的底 → 延伸 end_fx（如果满足成笔条件）
                        if self._bi_fx_valid(start_fx, cur_fx):
                            end_fx = cur_fx
                            end_idx = i
                    elif end_fx.type == "ding" and cur_fx.val >= end_fx.val:
                        # 更高/等高的顶 → 延伸 end_fx
                        if self._bi_fx_valid(start_fx, cur_fx):
                            end_fx = cur_fx
                            end_idx = i
                    i += 1
                else:
                    # 反类型分型（和 start_fx 同类）
                    # 检查能否从 end_fx 出发形成下一笔
                    confirm = self._bi_fx_valid(end_fx, cur_fx)
                    # 次高低检查
                    if confirm and self.bi_fx_cgd == Config.BI_FX_CHD_NO.value:
                        k_gap_confirm = cur_fx.k.k_index - end_fx.k.k_index
                        if k_gap_confirm < self.fx_check_k_nums:
                            # 确认分型(cur_fx)是否为 end_fx~cur_fx 之间的次高低
                            for j in range(end_idx + 1, i):
                                mid_fx = fxs[j]
                                if mid_fx.type == cur_fx.type:
                                    if cur_fx.type == "ding" and mid_fx.val > cur_fx.val:
                                        confirm = False
                                        break
                                    elif cur_fx.type == "di" and mid_fx.val < cur_fx.val:
                                        confirm = False
                                        break
                    if confirm:
                        # 能成笔 → 确认当前笔
                        bi_type = "down" if start_fx.type == "ding" else "up"
                        bi = BI(
                            start=start_fx,
                            end=end_fx,
                            _type=bi_type,
                            index=len(bis),
                            default_zs_type=self.default_bi_zs_type,
                        )
                        bis.append(bi)
                        has_confirmed_bi = True
                        # end_fx 成为新的 start_fx
                        start_fx = end_fx
                        start_idx = end_idx
                        end_fx = None
                        end_idx = -1
                        # 从 start_fx 的下一个分型继续
                        i = start_idx + 1
                    else:
                        # 不能成笔 → 忽略，继续扫描
                        i += 1

        # 处理最后一个未确认的笔
        if end_fx is not None:
            bi_type = "down" if start_fx.type == "ding" else "up"
            bi = BI(
                start=start_fx,
                end=end_fx,
                _type=bi_type,
                index=len(bis),
                default_zs_type=self.default_bi_zs_type,
            )
            bis.append(bi)

        return bis

    def _bi_fx_valid(self, start_fx: FX, end_fx: FX) -> bool:
        """检查两个分型之间是否能构成合法的笔"""
        # 类型必须交替
        if start_fx.type == end_fx.type:
            return False

        # 间隔检查
        cl_gap = end_fx.k.index - start_fx.k.index
        k_gap = end_fx.k.k_index - start_fx.k.k_index

        if self.bi_type == Config.BI_TYPE_DD.value:
            if cl_gap < 1:
                return False
        elif self.bi_type == Config.BI_TYPE_JDB.value:
            if k_gap < 4:
                return False
        else:
            # bi_type_old等旧笔规则：最小间隔按缠论K线间隔判断
            # 通过后继续走下方严格检查逻辑
            if cl_gap < 4:
                return False

        # 当原始K线间距未超过检查阈值时，应用严格检查
        if k_gap < self.fx_check_k_nums:
            # 分型严格处理（使用分型区间的高低点）
            if self.allow_bi_fx_strict:
                qj = self.fx_qj
                qy = self.fx_qy
                if start_fx.type == "ding" and end_fx.type == "di":
                    # 下降笔: 起始顶分型的低点不应低于终止底分型的低点
                    if start_fx.low(qj, qy) < end_fx.low(qj, qy):
                        return False
                    # 下降笔: 终止底分型的高点不应超过起始顶分型的高点
                    if end_fx.high(qj, qy) > start_fx.high(qj, qy):
                        return False
                elif start_fx.type == "di" and end_fx.type == "ding":
                    # 上升笔: 起始底分型的高点不应超过终止顶分型的高点
                    if start_fx.high(qj, qy) > end_fx.high(qj, qy):
                        return False
                    # 上升笔: 终止顶分型的低点不应低于起始底分型的低点
                    if end_fx.low(qj, qy) < start_fx.low(qj, qy):
                        return False

        return True

    def _bi_special_bi_split(self, bis: List[BI]) -> List[BI]:
        """
        笔拆分：当一笔内分型交叉（重叠K线）数量达到阈值时，将该笔拆分为三段子笔。

        算法：
        1. 取笔内部分型（不含起止），构成连续三元组
        2. 对每个三元组，遍历从第一个分型到笔终点之间的原始K线
        3. 若一根K线同时与三元组中所有三个分型的区间重叠，则计为一次交叉命中
        4. 允许最多 tolerance (默认1) 根连续未命中；超出则中断
        5. 命中数 ≥ threshold (默认20) 时触发拆分
        6. 拆分选点：按值排序遍历 (di, ding) 对，取第一个使三段子笔均合法的组合
        """
        if self.bi_split_k_cross_nums <= 0:
            return bis

        qj = self.fx_qj
        qy = self.fx_qy
        threshold = self.bi_split_k_cross_nums
        tolerance = self.bi_split_k_cross_tolerance

        result: List[BI] = []
        for bi in bis:
            split_bis = self._try_split_bi(bi, qj, qy, threshold, tolerance)
            result.extend(split_bis)

        # 重建索引
        for i, b in enumerate(result):
            b.index = i
        return result

    def _try_split_bi(self, bi: BI, qj: str, qy: str,
                      threshold: int, tolerance: int) -> List[BI]:
        """尝试拆分单笔，返回拆分后的笔列表（可能不变）"""
        start_idx = bi.start.k.index
        end_idx = bi.end.k.index

        # 获取笔内部的分型（不含起止分型）
        internal_fxs = [fx for fx in self.fxs
                        if start_idx < fx.k.index < end_idx]

        if len(internal_fxs) < 3:
            return [bi]

        # ---- Phase 1: 交叉计数 ----
        triggered_ti = -1
        end_ki = bi.end.k.k_index  # 原始K线终点索引

        for ti in range(len(internal_fxs) - 2):
            fx1 = internal_fxs[ti]
            fx2 = internal_fxs[ti + 1]
            fx3 = internal_fxs[ti + 2]

            h1, l1 = fx1.high(qj, qy), fx1.low(qj, qy)
            h2, l2 = fx2.high(qj, qy), fx2.low(qj, qy)
            h3, l3 = fx3.high(qj, qy), fx3.low(qj, qy)

            hit_count = 0
            miss_count = 0

            for ki in range(fx1.k.k_index, end_ki):
                k = self.src_klines[ki]
                # K线同时与三个分型区间重叠
                if (k.h >= l1 and k.l <= h1
                        and k.h >= l2 and k.l <= h2
                        and k.h >= l3 and k.l <= h3):
                    hit_count += 1
                    miss_count = 0
                else:
                    miss_count += 1
                if miss_count > tolerance:
                    break

            if hit_count >= threshold:
                triggered_ti = ti
                break

        if triggered_ti < 0:
            return [bi]

        # ---- Phase 2: 拆分选点 ----
        triplet = (internal_fxs[triggered_ti],
                   internal_fxs[triggered_ti + 1],
                   internal_fxs[triggered_ti + 2])
        if bi.type == "down":
            return self._select_split_down(bi, internal_fxs, triplet)
        else:
            return self._select_split_up(bi, internal_fxs, triplet)

    def _split_gap_ok(self, fx_a: FX, fx_b: FX) -> bool:
        """拆分子笔间隔检查（不含严格分型检查）"""
        cl_gap = fx_b.k.index - fx_a.k.index
        k_gap = fx_b.k.k_index - fx_a.k.k_index
        if self.bi_type == Config.BI_TYPE_DD.value:
            return cl_gap >= 1
        else:
            return k_gap >= 4

    def _find_split1_from_triplet(self, bi, triplet, split1_type):
        """从触发三元组中选取 split1 分型"""
        # 仅从三元组内选取 split1 类型的候选
        candidates = [fx for fx in triplet if fx.type == split1_type]
        # 按位置排序
        candidates.sort(key=lambda f: f.k.index)
        # 优先选取满足间隔条件的第一个
        for fx in candidates:
            if self._split_gap_ok(bi.start, fx):
                return fx
        # 都不满足则取最后一个
        return candidates[-1] if candidates else None

    def _find_split2(self, split1_fx, bi_end_fx, split2_type, internal_fxs):
        """在 split1 之后选取最优 split2 分型：优先最极值（带间隔），退而首个有效"""
        candidates = [fx for fx in internal_fxs
                      if fx.type == split2_type and fx.k.index > split1_fx.k.index]
        if not candidates:
            return None
        # 方向检查：down 的 split2 是 ding，需要 ding.val > di(split1).val
        # up 的 split2 是 di，需要 ding(split1).val > di.val
        if split2_type == "ding":
            valid = [fx for fx in candidates if fx.val > split1_fx.val]
        else:
            valid = [fx for fx in candidates if split1_fx.val > fx.val]
        if not valid:
            return None
        # 在满足间隔条件的候选中找最极值
        gap_ok = [fx for fx in valid
                  if self._split_gap_ok(split1_fx, fx)]
        if gap_ok:
            if split2_type == "ding":
                return max(gap_ok, key=lambda f: f.val)
            else:
                return min(gap_ok, key=lambda f: f.val)
        # 无 gap 合格候选，取首个有效
        return valid[0]

    def _select_split_down(self, bi: BI, internal_fxs: List[FX],
                           triplet: tuple) -> List[BI]:
        """下降笔拆分选点：start(ding)→di_fx→ding_fx→end(di)"""
        di_fx = self._find_split1_from_triplet(bi, triplet, "di")
        if di_fx is None:
            return [bi]
        ding_fx = self._find_split2(di_fx, bi.end, "ding", internal_fxs)
        if ding_fx is None:
            return [bi]
        bi1 = BI(start=bi.start, end=di_fx, _type="down",
                 index=0, default_zs_type=bi.default_zs_type)
        bi2 = BI(start=di_fx, end=ding_fx, _type="up",
                 index=1, default_zs_type=bi.default_zs_type)
        bi3 = BI(start=ding_fx, end=bi.end, _type="down",
                 index=2, default_zs_type=bi.default_zs_type)
        return [bi1, bi2, bi3]

    def _select_split_up(self, bi: BI, internal_fxs: List[FX],
                         triplet: tuple) -> List[BI]:
        """上升笔拆分选点：start(di)→ding_fx→di_fx→end(ding)"""
        ding_fx = self._find_split1_from_triplet(bi, triplet, "ding")
        if ding_fx is None:
            return [bi]
        di_fx = self._find_split2(ding_fx, bi.end, "di", internal_fxs)
        if di_fx is None:
            return [bi]
        bi1 = BI(start=bi.start, end=ding_fx, _type="up",
                 index=0, default_zs_type=bi.default_zs_type)
        bi2 = BI(start=ding_fx, end=di_fx, _type="down",
                 index=1, default_zs_type=bi.default_zs_type)
        bi3 = BI(start=di_fx, end=bi.end, _type="up",
                 index=2, default_zs_type=bi.default_zs_type)
        return [bi1, bi2, bi3]

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
        """根据笔列表构建笔中枢（多中枢类型）"""
        all_zss = []
        for zs_type in self.zs_bi_type:
            zss = self._build_zs_by_type(zs_type, self.bis, "bi")
            all_zss.extend(zss)
        self.bi_zss = all_zss

    def _build_xd_zss(self):
        """根据线段列表构建线段中枢（多中枢类型）"""
        all_zss = []
        for zs_type in self.zs_xd_type:
            zss = self._build_zs_by_type(zs_type, self.xds, "xd")
            all_zss.extend(zss)
        self.xd_zss = all_zss

    def _build_zs_by_type(
        self, zs_type: str, lines: List[LINE], line_type: str
    ) -> List[ZS]:
        """根据中枢类型构建中枢"""
        if zs_type == Config.ZS_TYPE_DN.value:
            return self.create_dn_zs(line_type, lines)
        elif zs_type == Config.ZS_TYPE_BZ.value:
            return self._build_zs_bz(lines, line_type)
        elif zs_type == Config.ZS_TYPE_FX.value:
            return self._build_zs_fx(lines, line_type)
        elif zs_type == Config.ZS_TYPE_FL.value:
            return self._build_zs_fl(lines, line_type)
        else:
            return self.create_dn_zs(line_type, lines)

    def _get_line_zs_highlow(self, line: LINE) -> Tuple[float, float]:
        """根据 zs_qj 配置获取线的高低点用于中枢计算"""
        if self.zs_qj == Config.ZS_QJ_DD.value:
            return line.high, line.low
        elif self.zs_qj == Config.ZS_QJ_CK.value:
            # 缠论K线高低
            ck_start = line.start.k.index
            ck_end = line.end.k.index
            h = max(ck.h for ck in self.cl_klines[ck_start : ck_end + 1])
            l = min(ck.l for ck in self.cl_klines[ck_start : ck_end + 1])
            return h, l
        elif self.zs_qj == Config.ZS_QJ_K.value:
            k_start = line.start.k.k_index
            k_end = line.end.k.k_index
            h = max(k.h for k in self.src_klines[k_start : k_end + 1])
            l = min(k.l for k in self.src_klines[k_start : k_end + 1])
            return h, l
        return line.high, line.low

    def _build_zs_bz(self, lines: List[LINE], line_type: str) -> List[ZS]:
        """标准中枢：第一段为进入段，接下来三段形成重叠区间，持续延伸"""
        if len(lines) < 4:
            return []
        zss: List[ZS] = []
        i = 0
        while i + 3 < len(lines):
            # lines[i] 是进入段，lines[i+1],[i+2],[i+3] 形成重叠区间
            h1, l1 = self._get_line_zs_highlow(lines[i + 1])
            h2, l2 = self._get_line_zs_highlow(lines[i + 2])
            h3, l3 = self._get_line_zs_highlow(lines[i + 3])
            zg = min(h1, h2, h3)
            zd = max(l1, l2, l3)
            if zg > zd:
                h0, l0 = self._get_line_zs_highlow(lines[i])
                gg = max(h0, h1, h2, h3)
                dd = min(l0, l1, l2, l3)
                zs = ZS(
                    zs_type=line_type,
                    start=lines[i].start,
                    end=lines[i + 3].end,
                    zg=zg,
                    zd=zd,
                    gg=gg,
                    dd=dd,
                    _type="up" if lines[i].type == "down" else "down",
                    index=len(zss),
                    line_num=4,
                    level=0,
                )
                zs.done = False
                zs.real = True
                for l in lines[i : i + 4]:
                    zs.add_line(l)
                # 延伸：后续段进入中枢区间则继续
                j = i + 4
                while j < len(lines):
                    hj, lj = self._get_line_zs_highlow(lines[j])
                    if hj >= zd and lj <= zg:
                        zs.add_line(lines[j])
                        zs.end = lines[j].end
                        zs.line_num += 1
                        zs.gg = max(zs.gg, hj)
                        zs.dd = min(zs.dd, lj)
                        # 根据 zs_cd 更新 zg/zd
                        if self.zs_cd == Config.ZS_CD_MORE.value:
                            zg = min(zg, hj)
                            zd = max(zd, lj)
                            zs.zg = zg
                            zs.zd = zd
                        j += 1
                    else:
                        break
                zs.done = True
                zss.append(zs)
                # 下一个 ZS 从当前 ZS 的最后一段开始（共享边界）
                i = j - 1
            else:
                i += 1
        return zss

    def _build_zs_fx(self, lines: List[LINE], line_type: str) -> List[ZS]:
        """方向中枢：进入段和离开段方向相反"""
        if len(lines) < 3:
            return []
        zss: List[ZS] = []
        i = 0
        while i + 2 < len(lines):
            # 方向中枢要求进入段和离开段方向一致（都是向上或向下），中间有反向
            enter_line = lines[i]
            h0, l0 = self._get_line_zs_highlow(lines[i])
            h1, l1 = self._get_line_zs_highlow(lines[i + 1])
            h2, l2 = self._get_line_zs_highlow(lines[i + 2])
            zg = min(h0, h1, h2)
            zd = max(l0, l1, l2)
            if zg > zd:
                # 确定中枢方向
                if enter_line.type == "up":
                    zs_type_dir = "up"
                else:
                    zs_type_dir = "down"
                zs = ZS(
                    zs_type=line_type,
                    start=lines[i].start,
                    end=lines[i + 2].end,
                    zg=zg,
                    zd=zd,
                    gg=max(h0, h1, h2),
                    dd=min(l0, l1, l2),
                    _type=zs_type_dir,
                    index=len(zss),
                    line_num=3,
                    level=0,
                )
                zs.done = True
                zs.real = True
                for l in lines[i : i + 3]:
                    zs.add_line(l)
                # 方向中枢不延伸
                zss.append(zs)
                i += 3
            else:
                i += 1
        return zss

    def _build_zs_fl(self, lines: List[LINE], line_type: str) -> List[ZS]:
        """分类中枢：类似段内中枢但允许跨段延续"""
        # 简化实现：与段内中枢相同
        return self.create_dn_zs(line_type, lines)

    # ---- 线段构建 ----
    def _build_xds(self, bis: List[BI]) -> List[XD]:
        """
        使用特征序列方法构建线段。

        算法：
        1. 通过特征序列分型确定初始线段方向和起始位置
        2. 取与线段方向相反的笔作为特征序列元素
        3. 对特征序列做方向感知的包含处理（OLD⊃NEW合并，NEW⊃OLD保留但标记line_bad）
        4. 在合并后的特征序列中寻找序列分型（包含line_bad的中间元素）
        5. 序列分型确认 → 线段结束
        """
        if len(bis) < 3:
            return []

        xds: List[XD] = []

        # 确定第一段方向和起始位置
        xd_type, start_bi_idx = self._find_first_xd_start(bis)

        while start_bi_idx < len(bis):
            result = self._find_xd_end(bis, start_bi_idx, xd_type)
            if result is None:
                # 无法找到线段结束点，把剩余笔作为最后一段（未完成）
                if start_bi_idx < len(bis) - 2:
                    end_bi_idx = len(bis) - 1
                    # 未完成线段的 end_bi 应与线段方向一致
                    if bis[end_bi_idx].type != xd_type:
                        end_bi_idx -= 1
                    xd = self._create_xd(
                        bis, start_bi_idx, end_bi_idx, xd_type, len(xds), done=False
                    )
                    if xd is not None:
                        xds.append(xd)
                break

            end_bi_idx, ding_fx, di_fx, tzxls = result

            xd = self._create_xd(
                bis, start_bi_idx, end_bi_idx, xd_type, len(xds),
                done=True, ding_fx=ding_fx, di_fx=di_fx, tzxls=tzxls,
            )
            if xd is not None:
                xds.append(xd)
                # 下一段从 end_bi_idx + 1 开始
                start_bi_idx = end_bi_idx + 1
                xd_type = "down" if xd_type == "up" else "up"
            else:
                start_bi_idx += 1

        # 线段拆分后处理
        xds = self._split_xds(xds, bis)

        return xds

    def _build_zs_in_range(self, bis: List[BI], start_idx: int, end_idx: int) -> list:
        """在一段 BI 范围内构建中枢列表"""
        seg_bis = bis[start_idx:end_idx + 1]
        zs_list = []
        i = 0
        while i < len(seg_bis) - 2:
            bi1 = seg_bis[i]
            bi2 = seg_bis[i + 1]
            bi3 = seg_bis[i + 2]
            zg = min(bi1.high, bi2.high, bi3.high)
            zd = max(bi1.low, bi2.low, bi3.low)
            if zg > zd:
                gg = max(bi1.high, bi2.high, bi3.high)
                dd = min(bi1.low, bi2.low, bi3.low)
                zs_type = "up" if bi1.type == "down" else "down"
                lines = [bi1, bi2, bi3]
                j = i + 3
                while j < len(seg_bis):
                    bj = seg_bis[j]
                    if bj.low < zg and bj.high > zd:
                        lines.append(bj)
                        gg = max(gg, bj.high)
                        dd = min(dd, bj.low)
                        j += 1
                    else:
                        break
                zs_list.append({
                    'start_bi': bi1.index,
                    'end_bi': lines[-1].index,
                    'zg': zg, 'zd': zd, 'gg': gg, 'dd': dd,
                    'type': zs_type,
                    'line_num': len(lines),
                    'start_list_idx': start_idx + i,
                    'end_list_idx': start_idx + i + len(lines) - 1,
                })
                i = j
            else:
                i += 1
        return zs_list

    def _split_xds(self, xds: List[XD], bis: List[BI]) -> List[XD]:
        """
        线段拆分后处理。
        检查每个线段内是否需要拆分，基于以下条件：
        1. 段内中枢线段超过阈值 (xd_allow_split_zs_more_line)
        2. 段内不同向中枢 (xd_allow_split_zs_no_direction)
        3. 笔破坏
        """
        result = []
        for xd in xds:
            splits = self._check_xd_split(xd, bis)
            if splits:
                result.extend(splits)
            else:
                result.append(xd)

        # 合并连续同向线段（拆分可能导致子段方向与下一段冲突）
        i = 0
        while i < len(result) - 1:
            if result[i].type == result[i + 1].type:
                result[i].end_line = result[i + 1].end_line
                result[i].high = max(result[i].high, result[i + 1].high)
                result[i].low = min(result[i].low, result[i + 1].low)
                result.pop(i + 1)
            else:
                i += 1

        # 重新编号
        for i, xd in enumerate(result):
            xd.index = i

        return result

    def _check_xd_split(self, xd: XD, bis: List[BI]) -> Union[List[XD], None]:
        """
        检查单个线段是否需要拆分，返回拆分后的线段列表或 None。
        """
        # 只拆分已完成的线段
        if not xd.done:
            return None

        start_idx = xd.start_line.index
        end_idx = xd.end_line.index
        bi_count = end_idx - start_idx + 1

        if bi_count < 5:
            return None

        # 构建段内中枢
        zs_list = self._build_zs_in_range(bis, start_idx, end_idx)
        if not zs_list:
            return None

        same_dir_zs = [zs for zs in zs_list if zs['type'] == xd.type]
        opp_dir_zs = [zs for zs in zs_list if zs['type'] != xd.type]

        # 检查条件1: 段内中枢线段超过阈值（任何方向）
        if self.xd_allow_split_zs_more_line:
            for zs in zs_list:
                if zs['line_num'] >= self.xd_zs_max_lines_split:
                    split_result = self._split_by_long_zs(xd, bis, zs_list, zs)
                    if split_result:
                        return split_result

        # 检查条件2: 段内不同向中枢（需要: 无同向中枢, 中枢不覆盖整段, 段内有笔破坏, 且段至少7笔）
        if self.xd_allow_split_zs_no_direction and opp_dir_zs and not same_dir_zs and bi_count >= 7:
            for zs in opp_dir_zs:
                # 中枢必须不覆盖整个线段
                if zs['start_list_idx'] > start_idx or zs['end_list_idx'] < end_idx:
                    # 还需要段内存在笔破坏
                    if self._has_bi_pohuai_in_range(bis, start_idx, end_idx, xd.type):
                        split_result = self._split_by_opposite_zs(xd, bis, zs)
                        if split_result:
                            return split_result

        return None

    def _has_bi_pohuai_in_range(self, bis: List[BI], start_idx: int, end_idx: int, xd_type: str) -> bool:
        """检查范围内是否存在笔破坏（反向笔超过前一反向笔）"""
        seg_bis = bis[start_idx:end_idx + 1]
        for i in range(2, len(seg_bis)):
            b = seg_bis[i]
            prev_same = None
            for k in range(i - 2, -1, -1):
                if seg_bis[k].type == b.type:
                    prev_same = seg_bis[k]
                    break
            if prev_same is None:
                continue
            if xd_type == "down" and b.type == "up" and b.high > prev_same.high:
                return True
            elif xd_type == "up" and b.type == "down" and b.low < prev_same.low:
                return True
        return False

    def _split_by_long_zs(self, xd: XD, bis: List[BI], zs_list: list, long_zs: dict) -> Union[List[XD], None]:
        """
        根据超长中枢拆分线段。
        策略：找到中枢内第一个笔破坏点作为拆分位置。
        """
        start_idx = xd.start_line.index
        end_idx = xd.end_line.index

        split_points = []

        zs_start = long_zs['start_list_idx']
        zs_end = long_zs['end_list_idx']

        # 如果有前置中枢，在前置中枢结束位置拆分
        for zs in zs_list:
            if zs is long_zs:
                continue
            if zs['end_list_idx'] < zs_start:
                split_bi = zs['end_list_idx']
                split_points.append(split_bi)
                break

        if not split_points:
            # 如果中枢方向与段方向相反且覆盖整段
            if long_zs['type'] != xd.type and zs_start == start_idx and zs_end == end_idx:
                # 检查是否有反向笔突破段起始值（UP 段检查 DOWN 笔低于起始 low，DOWN 段检查 UP 笔高于起始 high）
                seg_bis = bis[start_idx:end_idx + 1]
                start_bi = seg_bis[0]
                bi_pohuai_idx = None
                for i in range(2, len(seg_bis)):
                    b = seg_bis[i]
                    if xd.type == "up" and b.type == "down" and b.low < start_bi.low:
                        bi_pohuai_idx = start_idx + i - 1
                        break
                    elif xd.type == "down" and b.type == "up" and b.high > start_bi.high:
                        bi_pohuai_idx = start_idx + i - 1
                        break
                if bi_pohuai_idx is not None and bi_pohuai_idx > start_idx:
                    split_points.append(bi_pohuai_idx)
                else:
                    # 无反向笔突破段起始值，视为非标准线段，在首笔拆分
                    split_points.append(start_idx)
            else:
                # 在中枢范围内寻找第一个笔破坏点（相对前一同向笔）
                seg_bis = bis[zs_start:zs_end + 1]
                bi_pohuai_idx = None
                for i in range(2, len(seg_bis)):
                    b = seg_bis[i]
                    prev_same = None
                    for k in range(i - 2, -1, -1):
                        if seg_bis[k].type == b.type:
                            prev_same = seg_bis[k]
                            break
                    if prev_same is None:
                        continue
                    if xd.type == "up" and b.type == "down" and b.low < prev_same.low:
                        bi_pohuai_idx = zs_start + i - 1
                        break
                    elif xd.type == "down" and b.type == "up" and b.high > prev_same.high:
                        bi_pohuai_idx = zs_start + i - 1
                        break

                if bi_pohuai_idx is not None and bi_pohuai_idx > start_idx:
                    split_points.append(bi_pohuai_idx)
                elif zs_start > start_idx:
                    # 没有笔破坏但中枢不从段开头开始
                    split_points.append(zs_start - 1)
                else:
                    # 中枢从段开头开始且无笔破坏，使用第一笔拆分
                    split_points.append(start_idx)

        if not split_points:
            return None

        # 使用拆分点创建新线段
        result = self._create_split_segments(
            xd, bis, split_points, '段内中枢线段超过11'
        )

        # 对拆分后的段继续检查笔破坏
        if result:
            result = self._check_bi_pohuai_splits(result, bis)

        return result

    def _split_by_opposite_zs(self, xd: XD, bis: List[BI], opp_zs: dict) -> Union[List[XD], None]:
        """
        根据段内不同向中枢拆分线段。
        在中枢边界处拆分。
        """
        start_idx = xd.start_line.index
        end_idx = xd.end_line.index

        # 寻找笔破坏点作为拆分点
        seg_bis = bis[start_idx:end_idx + 1]
        split_bi_idx = None

        for i in range(2, len(seg_bis)):
            b = seg_bis[i]
            if i >= 2:
                # 检查笔破坏：同向笔超过前一同向笔
                prev_same = None
                for k in range(i - 2, -1, -1):
                    if seg_bis[k].type == b.type:
                        prev_same = seg_bis[k]
                        break
                if prev_same is not None:
                    if xd.type == "up" and b.type == "down" and b.low < prev_same.low:
                        # 下跌方向的笔破坏，在此之前拆分
                        split_bi_idx = start_idx + i - 1
                        break
                    elif xd.type == "down" and b.type == "up" and b.high > prev_same.high:
                        # 上涨方向的笔破坏，在此之前拆分
                        split_bi_idx = start_idx + i - 1
                        break

        if split_bi_idx is None:
            # 没有笔破坏点，在中枢开始处拆分
            zs_start = opp_zs['start_list_idx']
            if zs_start > start_idx:
                split_bi_idx = zs_start - 1
            else:
                # 中枢从线段开始就存在，使用第一个BI作为拆分
                split_bi_idx = start_idx

        split_points = [split_bi_idx]

        result = self._create_split_segments(
            xd, bis, split_points, '段内不同向中枢拆分'
        )

        # 对拆分后的段继续检查笔破坏
        if result:
            result = self._check_bi_pohuai_splits(result, bis)

        return result

    def _split_by_bi_pohuai(self, xd: XD, bis: List[BI]) -> Union[List[XD], None]:
        """
        纯笔破坏拆分：在第一个笔破坏点拆分。
        用于段内笔数较多但不满足中枢拆分条件的情况。
        最少需要 4 笔（2 个同向笔才能比较）。
        """
        start_idx = xd.start_line.index
        end_idx = xd.end_line.index
        seg_bis = bis[start_idx:end_idx + 1]

        # 寻找第一个笔破坏点
        for i in range(2, len(seg_bis)):
            b = seg_bis[i]
            prev_same = None
            for k in range(i - 2, -1, -1):
                if seg_bis[k].type == b.type:
                    prev_same = seg_bis[k]
                    break
            if prev_same is None:
                continue
            if xd.type == "down" and b.type == "up" and b.high > prev_same.high:
                split_bi_idx = start_idx + i - 1
                result = self._create_split_segments(xd, bis, [split_bi_idx], '笔破坏')
                if result:
                    result = self._check_bi_pohuai_splits(result, bis)
                return result
            elif xd.type == "up" and b.type == "down" and b.low < prev_same.low:
                split_bi_idx = start_idx + i - 1
                result = self._create_split_segments(xd, bis, [split_bi_idx], '笔破坏')
                if result:
                    result = self._check_bi_pohuai_splits(result, bis)
                return result

        return None

    def _check_bi_pohuai_splits(self, xds: List[XD], bis: List[BI]) -> List[XD]:
        """
        检查拆分后的线段内是否有笔破坏，继续拆分。
        """
        result = []
        for xd in xds:
            split_result = self._find_bi_pohuai_split(xd, bis)
            if split_result:
                result.extend(split_result)
            else:
                result.append(xd)
        return result

    def _find_bi_pohuai_split(self, xd: XD, bis: List[BI]) -> Union[List[XD], None]:
        """
        在线段内寻找笔破坏拆分点。
        策略：找到段内最极端的同向笔，在其位置拆分。
        最少需要 4 笔。
        """
        start_idx = xd.start_line.index
        end_idx = xd.end_line.index
        if end_idx - start_idx < 3:
            return None

        seg_bis = bis[start_idx:end_idx + 1]

        # DOWN 段找最低 low 的 DOWN 笔, UP 段找最高 high 的 UP 笔
        extreme_idx = None
        extreme_val = None
        for i in range(len(seg_bis)):
            b = seg_bis[i]
            if xd.type == "down" and b.type == "down":
                if extreme_val is None or b.low < extreme_val:
                    extreme_val = b.low
                    extreme_idx = start_idx + i
            elif xd.type == "up" and b.type == "up":
                if extreme_val is None or b.high > extreme_val:
                    extreme_val = b.high
                    extreme_idx = start_idx + i

        if extreme_idx is not None and extreme_idx > start_idx and extreme_idx < end_idx:
            return self._create_split_segments(
                xd, bis, [extreme_idx], '笔破坏'
            )

        return None

    def _create_split_segments(
        self, orig_xd: XD, bis: List[BI], split_points: List[int], reason: str
    ) -> Union[List[XD], None]:
        """
        根据拆分点列表创建拆分后的线段。
        split_points: 拆分点的 BI index（每个拆分点是前一段的最后一个 BI）
        """
        start_idx = orig_xd.start_line.index
        end_idx = orig_xd.end_line.index

        # 构建区间列表
        boundaries = [start_idx] + [sp + 1 for sp in split_points] + [end_idx + 1]
        segments = []

        xd_type = orig_xd.type
        for k in range(len(boundaries) - 1):
            seg_start = boundaries[k]
            seg_end = boundaries[k + 1] - 1

            if seg_start >= len(bis) or seg_end >= len(bis) or seg_start > seg_end:
                continue

            # 确定方向：交替方向
            if k > 0:
                xd_type = "down" if segments[-1].type == "up" else "up"

            xd = self._create_split_xd(
                bis, seg_start, seg_end, xd_type, len(segments), reason
            )
            if xd is not None:
                segments.append(xd)

        if len(segments) >= 2:
            return segments
        return None

    def _create_split_xd(
        self, bis: List[BI], start_bi_idx: int, end_bi_idx: int,
        xd_type: str, index: int, reason: str
    ) -> Union[XD, None]:
        """创建拆分后的线段对象（允许单一笔线段）"""
        if start_bi_idx >= len(bis) or end_bi_idx >= len(bis):
            return None

        start_bi = bis[start_bi_idx]
        end_bi = bis[end_bi_idx]

        ding_fx = XLFX(
            _type="ding",
            xl=TZXL("up", end_bi if xd_type == "up" else start_bi, start_bi, False, True),
            xls=[None, TZXL("up", end_bi if xd_type == "up" else start_bi, start_bi, False, True), None],
            done=True,
        )
        di_fx = XLFX(
            _type="di",
            xl=TZXL("down", end_bi if xd_type == "down" else start_bi, start_bi, False, True),
            xls=[None, TZXL("down", end_bi if xd_type == "down" else start_bi, start_bi, False, True), None],
            done=True,
        )

        xd = XD(
            start=start_bi.start if xd_type == "up" else start_bi.start,
            end=end_bi.end,
            start_line=start_bi,
            end_line=end_bi,
            _type=xd_type,
            ding_fx=ding_fx,
            di_fx=di_fx,
            index=index,
            default_zs_type=self.default_xd_zs_type,
        )
        xd.done = True
        xd.is_split = reason

        # 设置高低点
        xd.high = max(bis[j].high for j in range(start_bi_idx, end_bi_idx + 1))
        xd.low = min(bis[j].low for j in range(start_bi_idx, end_bi_idx + 1))

        return xd

    def _find_first_xd_start(self, bis: List[BI]) -> Tuple[str, int]:
        """
        确定第一条线段的方向和起始位置。

        算法：
        1. 在 DOWN 笔特征序列中找所有 ding FX 位置
        2. 在 UP 笔特征序列中找第一个 di FX 位置
        3. 对每个 ding，尝试 _find_xd_end 看是否能形成有效 DOWN 段
        4. 对 di，尝试看是否能形成有效 UP 段
        5. 使用第一个有效的段作为起始段
        """
        if len(bis) < 3:
            return bis[0].type, 0

        # 获取所有 ding 候选（DOWN 笔 char seq）
        ding_candidates = self._find_all_tzxl_fx(bis, "down", "up", "ding")
        # 获取所有 di 候选（UP 笔 char seq）
        di_candidates = self._find_all_tzxl_fx(bis, "up", "down", "di")

        # 合并候选并按 BI index 排序
        candidates = []
        for bi_idx, is_bad in ding_candidates:
            start_idx = bi_idx if bis[bi_idx].type == "down" else (bi_idx + 1 if bi_idx + 1 < len(bis) else None)
            if start_idx is not None:
                candidates.append(("down", start_idx, bi_idx))
        for bi_idx, is_bad in di_candidates:
            start_idx = bi_idx if bis[bi_idx].type == "up" else (bi_idx + 1 if bi_idx + 1 < len(bis) else None)
            if start_idx is not None:
                candidates.append(("up", start_idx, bi_idx))

        # 按起始位置排序
        candidates.sort(key=lambda c: c[1])

        # 尝试每个候选，找第一个能形成有效线段的
        for xd_type, start_idx, fx_bi_idx in candidates:
            result = self._find_xd_end(bis, start_idx, xd_type)
            if result is not None:
                end_bi_idx, ding_fx, di_fx, tzxls = result
                # 验证：确认分型中间元素中没有超过起始点的笔
                # 对 DOWN 段，di FX 中间元素不应有 high > start.high
                # 对 UP 段，ding FX 中间元素不应有 low < start.low
                start_bi = bis[start_idx]
                if xd_type == "down" and di_fx and di_fx.xl:
                    if any(l.high > start_bi.high for l in di_fx.xl.lines):
                        continue
                elif xd_type == "up" and ding_fx and ding_fx.xl:
                    if any(l.low < start_bi.low for l in ding_fx.xl.lines):
                        continue
                return xd_type, start_idx

        # 兜底：简单规则
        first_bi = bis[0]
        if len(bis) >= 2:
            if first_bi.type == "down" and bis[1].type == "up":
                if bis[1].high > first_bi.high:
                    return "down", 2
            elif first_bi.type == "up" and bis[1].type == "down":
                if bis[1].low < first_bi.low:
                    return "up", 2
        return first_bi.type, 0

    def _find_all_tzxl_fx(
        self, bis: List[BI], bi_type: str, bh_direction: str, fx_type: str,
    ) -> List[Tuple[int, bool]]:
        """
        找所有指定类型的 FX（返回列表）。
        Returns: [(fx_middle_bi_index, is_line_bad), ...]
        """
        tzxl_bis = [bi for bi in bis if bi.type == bi_type]
        if len(tzxl_bis) < 3:
            return []

        tzxls: List[TZXL] = []
        for bi in tzxl_bis:
            pre_line = bis[bi.index - 1] if bi.index > 0 else bi
            new_tzxl = TZXL(
                bh_direction=bh_direction, line=bi, pre_line=pre_line,
                line_bad=False, done=bi.is_done(),
            )
            if len(tzxls) == 0:
                tzxls.append(new_tzxl)
                continue
            last_tzxl = tzxls[-1]
            old_contains_new = last_tzxl.max >= new_tzxl.max and last_tzxl.min <= new_tzxl.min
            new_contains_old = new_tzxl.max >= last_tzxl.max and new_tzxl.min <= last_tzxl.min
            if old_contains_new:
                last_tzxl.lines.append(bi)
                last_tzxl.done = bi.is_done()
                last_tzxl.line_bad = False
                last_tzxl.update_maxmin()
            elif new_contains_old:
                new_tzxl.line_bad = True
                tzxls.append(new_tzxl)
            else:
                tzxls.append(new_tzxl)

        if len(tzxls) < 3:
            return []

        results = []
        for i in range(1, len(tzxls) - 1):
            curr = tzxls[i]
            prev = tzxls[i - 1]
            nxt = tzxls[i + 1]
            if fx_type == "ding" and curr.max > prev.max and curr.max > nxt.max:
                key_bi = max(curr.lines, key=lambda l: l.high)
                results.append((key_bi.index, curr.line_bad))
            elif fx_type == "di" and curr.min < prev.min and curr.min < nxt.min:
                key_bi = min(curr.lines, key=lambda l: l.low)
                results.append((key_bi.index, curr.line_bad))
        return results

    def _find_xd_end(
        self, bis: List[BI], start_bi_idx: int, xd_type: str,
    ) -> Union[Tuple[int, XLFX, XLFX, List[TZXL]], None]:
        """
        从 start_bi_idx 开始，寻找当前方向线段的结束位置。

        包含处理规则（方向感知）：
        - OLD⊃NEW（旧元素包含新元素）→ 合并，line_bad=False
        - NEW⊃OLD（新元素包含旧元素）→ 不合并，新元素 line_bad=True
        - 无包含 → 新元素 line_bad=False

        返回 (end_bi_idx, ding_fx, di_fx, tzxls) 或 None
        """
        # 特征序列方向：向上线段取向下笔，向下线段取向上笔
        tzxl_bi_type = "down" if xd_type == "up" else "up"
        # 包含处理方向
        bh_direction = "up" if xd_type == "up" else "down"
        # 寻找的分型类型
        target_fx_type = "ding" if xd_type == "up" else "di"

        # 收集反方向笔作为特征序列元素
        tzxl_bis = []
        for i in range(start_bi_idx, len(bis)):
            if bis[i].type == tzxl_bi_type:
                tzxl_bis.append(bis[i])

        if len(tzxl_bis) < 3:
            return None

        # 构建特征序列（方向感知的包含处理）
        tzxls: List[TZXL] = []
        for bi in tzxl_bis:
            pre_line = bis[bi.index - 1] if bi.index > 0 else bi
            done = bi.is_done()
            new_tzxl = TZXL(
                bh_direction=bh_direction,
                line=bi,
                pre_line=pre_line,
                line_bad=False,
                done=done,
            )

            if len(tzxls) == 0:
                tzxls.append(new_tzxl)
                continue

            last_tzxl = tzxls[-1]
            old_contains_new = last_tzxl.max >= new_tzxl.max and last_tzxl.min <= new_tzxl.min
            new_contains_old = new_tzxl.max >= last_tzxl.max and new_tzxl.min <= last_tzxl.min

            if old_contains_new:
                # OLD⊃NEW → 合并到旧元素
                last_tzxl.lines.append(bi)
                last_tzxl.done = done
                last_tzxl.line_bad = False
                last_tzxl.update_maxmin()
            elif new_contains_old:
                # NEW⊃OLD → 不合并，新元素标记 line_bad=True
                new_tzxl.line_bad = True
                tzxls.append(new_tzxl)
            else:
                # 无包含 → 正常添加
                tzxls.append(new_tzxl)

        # 在特征序列中寻找序列分型
        # line_bad 处理规则：
        #   - 特征序列位置较深（>= 3）的 bad FX 视为有效，直接使用
        #   - 位置较浅的 bad FX → 暂存，继续找非 bad FX
        #   - 遇到非 bad FX → 比较极值：
        #     - 非 bad 更极端（ding 更高 / di 更低）→ 使用非 bad
        #     - bad 更极端 → 使用 bad（它是真正的极值点）
        #   - 扫描结束无非 bad → 使用 bad（后备）
        if len(tzxls) < 3:
            return None

        first_bad_result = None
        first_bad_extreme = None  # bad FX 的极值（ding=max, di=min）

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
                # 检查笔破坏
                if not self._check_xd_bi_pohuai(bis, start_bi_idx, curr_xl, xd_type):
                    result = self._build_xd_fx_result(
                        bis, start_bi_idx, xd_type, target_fx_type,
                        curr_xl, prev_xl, next_xl, tzxls,
                    )
                    if result is not None:
                        # 位置较深的 bad FX（>= 3 个前置特征序列元素）视为有效
                        is_line_bad = curr_xl.line_bad and i < 3
                        if is_line_bad:
                            # bad FX → 暂存第一个，继续找非 bad
                            if first_bad_result is None:
                                first_bad_result = result
                                first_bad_extreme = curr_xl.max if target_fx_type == "ding" else curr_xl.min
                            continue

                        # 非 bad FX
                        if first_bad_result is not None:
                            # 比较极值：非 bad 更极端才替代 bad
                            is_more_extreme = (
                                (target_fx_type == "ding" and curr_xl.max > first_bad_extreme)
                                or (target_fx_type == "di" and curr_xl.min < first_bad_extreme)
                            )
                            if is_more_extreme:
                                # 非 bad 更极端 → 使用非 bad
                                _, ding_fx, di_fx, _ = result
                                target_xlfx = ding_fx if target_fx_type == "ding" else di_fx
                                target_xlfx.is_line_bad = True
                                return result
                            else:
                                # bad 更极端 → 使用 bad
                                return first_bad_result

                        return result

        # 没找到非 bad FX，回退到 bad FX
        if first_bad_result is not None:
            return first_bad_result

        return None

    def _build_xd_fx_result(
        self, bis, start_bi_idx, xd_type, target_fx_type,
        curr_xl, prev_xl, next_xl, tzxls,
    ):
        """构建 FX 结果元组"""
        xlfx = XLFX(
            _type=target_fx_type,
            xl=curr_xl,
            xls=[prev_xl, curr_xl, next_xl],
            done=next_xl.done,
        )
        xlfx.is_line_bad = curr_xl.line_bad

        if xd_type == "up":
            end_bi = max(curr_xl.lines, key=lambda l: l.high)
            end_bi_idx = end_bi.index
            if bis[end_bi_idx].type == "down" and end_bi_idx > 0:
                end_bi_idx -= 1
            ding_fx = xlfx
            di_fx = XLFX(
                _type="di",
                xl=tzxls[0],
                xls=[None, tzxls[0], tzxls[1] if len(tzxls) > 1 else None],
                done=True,
            )
        else:
            end_bi = min(curr_xl.lines, key=lambda l: l.low)
            end_bi_idx = end_bi.index
            if bis[end_bi_idx].type == "up" and end_bi_idx > 0:
                end_bi_idx -= 1
            di_fx = xlfx
            ding_fx = XLFX(
                _type="ding",
                xl=tzxls[0],
                xls=[None, tzxls[0], tzxls[1] if len(tzxls) > 1 else None],
                done=True,
            )

        if end_bi_idx - start_bi_idx >= 2:
            return (end_bi_idx, ding_fx, di_fx, tzxls)
        return None

    def _check_xd_bi_pohuai(
        self, bis: List[BI], start_bi_idx: int, target_xl: TZXL, xd_type: str
    ) -> bool:
        """
        检查是否发生笔破坏。
        笔破坏：在分型确认元素之后，有笔超过线段起始位置的高/低点。
        返回 True 表示发生笔破坏（线段不应在此结束）。
        """
        if self.xd_allow_bi_pohuai == Config.XD_BI_POHUAI_NO.value:
            return False

        start_bi = bis[start_bi_idx]
        if target_xl.line is None:
            return False

        last_line_idx = target_xl.lines[-1].index if target_xl.lines else target_xl.line.index

        for bi_idx in range(last_line_idx + 1, len(bis)):
            bi = bis[bi_idx]
            if xd_type == "up":
                if bi.type == "down" and bi.low < start_bi.low:
                    if self.xd_allow_bi_pohuai == Config.XD_BI_POHUAI_YES_QK.value:
                        has_qk = any(
                            ck.q
                            for ck in self.cl_klines[
                                bi.start.k.index : bi.end.k.index + 1
                            ]
                        )
                        return has_qk
                    return True
            else:
                if bi.type == "up" and bi.high > start_bi.high:
                    if self.xd_allow_bi_pohuai == Config.XD_BI_POHUAI_YES_QK.value:
                        has_qk = any(
                            ck.q
                            for ck in self.cl_klines[
                                bi.start.k.index : bi.end.k.index + 1
                            ]
                        )
                        return has_qk
                    return True
            break

        return False

    def _create_xd(
        self,
        bis: List[BI],
        start_bi_idx: int,
        end_bi_idx: int,
        xd_type: str,
        index: int,
        done: bool = True,
        ding_fx: XLFX = None,
        di_fx: XLFX = None,
        tzxls: List[TZXL] = None,
    ) -> Union[XD, None]:
        """创建线段对象"""
        if start_bi_idx >= len(bis) or end_bi_idx >= len(bis):
            return None
        if start_bi_idx == end_bi_idx:
            return None

        start_bi = bis[start_bi_idx]
        end_bi = bis[end_bi_idx]

        # 确保分型存在
        if ding_fx is None:
            # 创建默认的顶分型
            ding_fx = XLFX(
                _type="ding",
                xl=TZXL("up", end_bi if xd_type == "up" else start_bi,
                        start_bi, False, True),
                xls=[None, TZXL("up", end_bi if xd_type == "up" else start_bi,
                                start_bi, False, True), None],
                done=done,
            )
        if di_fx is None:
            di_fx = XLFX(
                _type="di",
                xl=TZXL("down", end_bi if xd_type == "down" else start_bi,
                        start_bi, False, True),
                xls=[None, TZXL("down", end_bi if xd_type == "down" else start_bi,
                                start_bi, False, True), None],
                done=done,
            )

        xd = XD(
            start=start_bi.start if xd_type == "up" else start_bi.start,
            end=end_bi.end,
            start_line=start_bi,
            end_line=end_bi,
            _type=xd_type,
            ding_fx=ding_fx,
            di_fx=di_fx,
            index=index,
            default_zs_type=self.default_xd_zs_type,
        )
        xd.done = done

        # 设置特征序列
        if tzxls is not None:
            xd.tzxls = tzxls

        # 设置高低点（基于笔的高低点）
        if xd_type == "up":
            xd.high = max(bis[j].high for j in range(start_bi_idx, end_bi_idx + 1))
            xd.low = min(bis[j].low for j in range(start_bi_idx, end_bi_idx + 1))
        else:
            xd.high = max(bis[j].high for j in range(start_bi_idx, end_bi_idx + 1))
            xd.low = min(bis[j].low for j in range(start_bi_idx, end_bi_idx + 1))

        return xd

    def _update_xd_highlow(self):
        """更新线段的高低点"""
        for xd in self.xds:
            start_k = xd.start.k.k_index
            end_k = xd.end.k.k_index
            if self.xd_qj == Config.XD_QJ_CK.value:
                ck_start = xd.start.k.index
                ck_end = xd.end.k.index
                xd.high = max(ck.h for ck in self.cl_klines[ck_start : ck_end + 1])
                xd.low = min(ck.l for ck in self.cl_klines[ck_start : ck_end + 1])
            elif self.xd_qj == Config.XD_QJ_K.value:
                xd.high = max(k.h for k in self.src_klines[start_k : end_k + 1])
                xd.low = min(k.l for k in self.src_klines[start_k : end_k + 1])
            # xd_qj_dd: 默认用 start/end 分型的顶底值，已在 _create_xd 中设置

    # ---- 笔背驰计算 ----
    def _calc_bi_bc(self):
        """计算笔级别的背驰"""
        if len(self.bis) < 3:
            return
        for zs_type in self.zs_bi_type:
            zss = [zs for zs in self.bi_zss if zs.zs_type == "bi"]
            self._calc_line_bc(self.bis, zss, zs_type)

    def _calc_xd_bc(self):
        """计算线段级别的背驰"""
        if len(self.xds) < 3:
            return
        for zs_type in self.zs_xd_type:
            zss = [zs for zs in self.xd_zss if zs.zs_type == "xd"]
            self._calc_line_bc(self.xds, zss, zs_type)

    def _calc_line_bc(
        self, lines: List[LINE], zss: List[ZS], zs_type: str
    ):
        """通用的线背驰计算"""
        from chanlun.cl_interface import compare_ld_beichi

        for idx in range(2, len(lines)):
            line = lines[idx]
            prev_same = lines[idx - 2]  # 前一同向段

            # 1. 笔/线段背驰：三段，第三段创新高/低，力度减弱
            if line.type == prev_same.type:
                is_new_extreme = False
                if line.type == "up" and line.high > prev_same.high:
                    is_new_extreme = True
                elif line.type == "down" and line.low < prev_same.low:
                    is_new_extreme = True

                if is_new_extreme:
                    bc = compare_ld_beichi(
                        prev_same.get_ld(self), line.get_ld(self), line.type
                    )
                    bc_type = "bi" if isinstance(line, BI) else "xd"
                    line.add_bc(bc_type, None, prev_same, [], bc, zs_type)

            # 2. 盘整背驰
            if len(zss) > 0:
                # 找到当前线已离开的中枢（当前线在中枢最后一段之后，且不属于任何中枢）
                line_in_any_zs = any(line.index <= zs.lines[-1].index >= line.index and line in zs.lines for zs in zss)
                related_zs = None
                if not line_in_any_zs:
                    for zs in reversed(zss):
                        if zs.lines and line.index > zs.lines[-1].index:
                            related_zs = zs
                            break
                if related_zs is not None:
                    bc, compare_line = self.beichi_pz(related_zs, line)
                    if bc:
                        line.add_bc("pz", related_zs, compare_line, [], True, zs_type)

            # 3. 趋势背驰
            if len(zss) >= 2:
                bc, compare_lines = self.beichi_qs(lines, zss, line)
                if bc:
                    line.add_bc("qs", zss[-1], compare_lines[0] if compare_lines else None,
                                compare_lines, True, zs_type)

    # ---- 笔买卖点计算 ----
    def _calc_bi_mmd(self):
        """计算笔级别的买卖点"""
        for zs_type in self.zs_bi_type:
            zss = [zs for zs in self.bi_zss if zs.zs_type == "bi"]
            self._calc_line_mmd(self.bis, zss, zs_type)

    def _calc_xd_mmd(self):
        """计算线段级别的买卖点"""
        for zs_type in self.zs_xd_type:
            zss = [zs for zs in self.xd_zss if zs.zs_type == "xd"]
            self._calc_line_mmd(self.xds, zss, zs_type)
        # 调用自定义买卖点
        from chanlun.cl_interface import user_custom_mmd

        for zs_type in self.zs_xd_type:
            zss = [zs for zs in self.xd_zss if zs.zs_type == "xd"]
            for xd in self.xds:
                user_custom_mmd(self, xd, self.xds, zs_type, zss)

    def _calc_line_mmd(
        self, lines: List[LINE], zss: List[ZS], zs_type: str
    ):
        """通用的买卖点计算"""
        if len(lines) < 2 or len(zss) == 0:
            return

        for line_idx in range(len(lines)):
            line = lines[line_idx]

            # === 三类买卖点 ===
            # 回调不进入中枢
            for zs in zss:
                if not zs.done or not zs.real:
                    continue
                if len(zs.lines) == 0:
                    continue
                last_zs_line = zs.lines[-1]

                # 三类买点：向上离开中枢后，紧随其后的第一段向下回调不回中枢
                if (
                    self.cl_mmd_cal_not_in_zs_3mmd
                    and line.type == "down"
                    and line.index == last_zs_line.index + 1
                    and line.low > zs.zg
                ):
                    # 检查中枢段数
                    if zs.line_num >= 9 and not self.cl_mmd_cal_not_in_zs_gt_9_3mmd:
                        continue
                    line.add_mmd("3buy", zs, zs_type, msg="回调不进入中枢，三买")

                # 三类卖点：向下离开中枢后，紧随其后的第一段向上回调不回中枢
                if (
                    self.cl_mmd_cal_not_in_zs_3mmd
                    and line.type == "up"
                    and line.index == last_zs_line.index + 1
                    and line.high < zs.zd
                ):
                    if zs.line_num >= 9 and not self.cl_mmd_cal_not_in_zs_gt_9_3mmd:
                        continue
                    line.add_mmd("3sell", zs, zs_type, msg="回调不进入中枢，三卖")

            # === 一类买卖点 ===
            # 1. 趋势背驰 → 一类买卖点
            if self.cl_mmd_cal_qs_1mmd and len(zss) >= 2:
                bc_result, compare_lines = self.beichi_qs(lines, zss, line)
                if bc_result:
                    if line.type == "down":
                        line.add_mmd("1buy", zss[-1], zs_type, msg="趋势背驰一买")
                    elif line.type == "up":
                        line.add_mmd("1sell", zss[-1], zs_type, msg="趋势背驰一卖")

            # 2. 三类买卖点后背驰 → 一类买卖点
            if line_idx >= 2:
                prev_same = lines[line_idx - 2]
                prev_opposite = lines[line_idx - 1]
                for mmd in prev_opposite.get_mmds(zs_type):
                    if (
                        self.cl_mmd_cal_not_qs_3mmd_1mmd
                        and mmd.name == "3sell"
                        and line.type == "down"
                    ):
                        # 三卖后创新低且背驰 → 一买
                        if line.low < prev_same.low and line.bc_exists(
                            ["bi", "pz", "qs"], zs_type
                        ):
                            line.add_mmd(
                                "1buy", mmd.zs, zs_type, msg="三卖后背驰一买"
                            )
                    if (
                        self.cl_mmd_cal_not_qs_3mmd_1mmd
                        and mmd.name == "3buy"
                        and line.type == "up"
                    ):
                        if line.high > prev_same.high and line.bc_exists(
                            ["bi", "pz", "qs"], zs_type
                        ):
                            line.add_mmd(
                                "1sell", mmd.zs, zs_type, msg="三买后背驰一卖"
                            )

            # === 二类买卖点 ===
            if line_idx >= 2:
                prev_same = lines[line_idx - 2]

                # 1. 一买后不创新低 → 二买
                if self.cl_mmd_cal_1mmd_not_lh_2mmd:
                    for mmd in prev_same.get_mmds(zs_type):
                        if mmd.name == "1buy" and line.type == "down":
                            if line.low > prev_same.low:
                                line.add_mmd(
                                    "2buy", mmd.zs, zs_type, msg="一买后不创新低二买"
                                )
                        if mmd.name == "1sell" and line.type == "up":
                            if line.high < prev_same.high:
                                line.add_mmd(
                                    "2sell", mmd.zs, zs_type, msg="一卖后不创新高二卖"
                                )

                # 2. 趋势后不创新低/高 → 二买/卖
                if self.cl_mmd_cal_qs_not_lh_2mmd and len(zss) >= 2:
                    qs_dir = self.zss_is_qs(zss[-2], zss[-1])
                    if qs_dir == "down" and line.type == "down":
                        if line.low > prev_same.low:
                            line.add_mmd(
                                "2buy", zss[-1], zs_type, msg="趋势后不创新低二买"
                            )
                    elif qs_dir == "up" and line.type == "up":
                        if line.high < prev_same.high:
                            line.add_mmd(
                                "2sell", zss[-1], zs_type, msg="趋势后不创新高二卖"
                            )

                # 3. 三买/卖后不创新低/高或背驰 → 二买/卖
                if self.cl_mmd_cal_3mmd_not_lh_bc_2mmd:
                    for mmd in prev_same.get_mmds(zs_type):
                        if mmd.name == "3buy" and line.type == "up":
                            if line.high < prev_same.high or line.bc_exists(
                                ["bi", "pz", "qs"], zs_type
                            ):
                                line.add_mmd(
                                    "2sell", mmd.zs, zs_type,
                                    msg="三买后不创新高或背驰二卖",
                                )
                        if mmd.name == "3sell" and line.type == "down":
                            if line.low > prev_same.low or line.bc_exists(
                                ["bi", "pz", "qs"], zs_type
                            ):
                                line.add_mmd(
                                    "2buy", mmd.zs, zs_type,
                                    msg="三卖后不创新低或背驰二买",
                                )

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
