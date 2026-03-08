# cl_open.py 完整缠论引擎实现计划

> 目标：实现 `cl_open.py` 使其输出与 `cl_pyarmor.py` 完全一致  
> 约束：PyArmor 试用授权有效期至 2026-03-20，所有对比测试必须在此之前完成  
> 测试数据：BTC/USDT、ETH/USDT，覆盖 w / d / 4h / 60m / 30m / 15m / 10m / 5m / 1m

---

## 一、现状分析

### 1.1 已实现 (cl_open.py, 775 行)

| 模块 | 方法 | 状态 | 说明 |
|------|------|------|------|
| K线构建 | `_build_src_klines()` | ✅ | DataFrame → Kline 列表 |
| K线合并 | `_merge_klines()` | ✅ | 包含处理，方向合并 |
| 分型识别 | `_find_fxs()` | ✅ | 顶底分型，含未完成分型处理 |
| 分型过滤 | `_filter_fxs()` | ✅ | 顶底交替，极值保留 |
| 分型包含 | `_check_fx_bh()` | ✅ | 6 种 fx_bh 配置 |
| 笔构建 | `_build_bis()` | ✅ | 4 种笔类型 (old/new/jdb/dd) |
| 笔高低 | `_update_bi_highlow()` | ✅ | 3 种 bi_qj + bi_bzh 标准化 |
| 笔中枢 | `_build_bi_zss()` → `create_dn_zs()` | ✅ 基础 | 段内中枢，缺标准中枢/方向中枢/分类中枢 |
| 盘整背驰 | `beichi_pz()` | ✅ 基础 | MACD力度对比，但 enter_line 取值有 bug |
| MACD | `_compute_macd()` | ✅ | talib 计算 |
| 增量计算 | `_incremental_compute()` | ✅ 简易 | 目前全量重算 |

### 1.2 待实现

| 模块 | 方法 | 优先级 | 复杂度 |
|------|------|--------|--------|
| **线段 (XD)** | `_build_xds()` | P0 | 高 |
| **走势段 (ZSD)** | `_build_zsds()` | P1 | 中 |
| **趋势段 (QSD)** | `_build_qsds()` | P2 | 中 |
| **中枢 — 4种类型** | 标准/段内/方向/分类 | P0 | 高 |
| **多中枢类型支持** | `zs_bi_type` / `zs_xd_type` 多选 | P0 | 中 |
| **买卖点 (MMD)** | 1/2/3/l2/l3 买卖点 | P0 | 高 |
| **背驰 (BC)** | bi/xd/zsd/pz/qs 背驰 | P0 | 高 |
| **趋势背驰** | `beichi_qs()` | P0 | 中 |
| **趋势判断** | `zss_is_qs()` 完整版 | P0 | 低 |
| **线段拆分** | 非标准/中枢扩展/中枢多段/反向中枢 | P1 | 中 |
| **笔拆分** | 笔重叠K线拆分 | P1 | 中 |
| **配置补全** | 缺少的 config 读取 | P0 | 低 |

### 1.3 需读取但未实现的配置项

```python
# __init__ 中需新增读取的配置 —
# 线段相关
self.xd_qj       = cl_config.get("xd_qj", Config.XD_QJ_DD.value)
self.xd_bzh      = cl_config.get("xd_qj_bzh", Config.ZSD_BZH_NO.value)  # 已弃用
self.xd_allow_bi_pohuai = cl_config.get("xd_allow_bi_pohuai", Config.XD_BI_POHUAI_YES.value)
self.xd_allow_split_no_highlow    = int(cl_config.get("xd_allow_split_no_highlow", 1))
self.xd_allow_split_zs_kz        = int(cl_config.get("xd_allow_split_zs_kz", 0))
self.xd_allow_split_zs_more_line  = int(cl_config.get("xd_allow_split_zs_more_line", 1))
self.xd_allow_split_zs_no_direction = int(cl_config.get("xd_allow_split_zs_no_direction", 1))
self.xd_zs_max_lines_split        = int(cl_config.get("xd_zs_max_lines_split", 11))

# 走势段
self.zsd_qj     = cl_config.get("zsd_qj", Config.ZSD_QJ_DD.value)

# 中枢相关
self.zs_bi_type  = cl_config.get("zs_bi_type", [Config.ZS_TYPE_BZ.value])  # 列表，多选
self.zs_xd_type  = cl_config.get("zs_xd_type", [Config.ZS_TYPE_BZ.value])  # 列表，多选
self.zs_qj       = cl_config.get("zs_qj", Config.ZS_QJ_DD.value)
self.zs_cd       = cl_config.get("zs_cd", Config.ZS_CD_THREE.value)
self.zs_wzgx     = cl_config.get("zs_wzgx", Config.ZS_WZGX_GD.value)

# 买卖点开关
self.cl_mmd_cal_qs_1mmd              = int(cl_config.get("cl_mmd_cal_qs_1mmd", 1))
self.cl_mmd_cal_not_qs_3mmd_1mmd     = int(cl_config.get("cl_mmd_cal_not_qs_3mmd_1mmd", 1))
self.cl_mmd_cal_qs_3mmd_1mmd         = int(cl_config.get("cl_mmd_cal_qs_3mmd_1mmd", 1))
self.cl_mmd_cal_qs_not_lh_2mmd       = int(cl_config.get("cl_mmd_cal_qs_not_lh_2mmd", 1))
self.cl_mmd_cal_qs_bc_2mmd           = int(cl_config.get("cl_mmd_cal_qs_bc_2mmd", 1))
self.cl_mmd_cal_3mmd_not_lh_bc_2mmd  = int(cl_config.get("cl_mmd_cal_3mmd_not_lh_bc_2mmd", 1))
self.cl_mmd_cal_1mmd_not_lh_2mmd     = int(cl_config.get("cl_mmd_cal_1mmd_not_lh_2mmd", 1))
self.cl_mmd_cal_3mmd_xgxd_not_bc_2mmd = int(cl_config.get("cl_mmd_cal_3mmd_xgxd_not_bc_2mmd", 1))
self.cl_mmd_cal_not_in_zs_3mmd       = int(cl_config.get("cl_mmd_cal_not_in_zs_3mmd", 1))
self.cl_mmd_cal_not_in_zs_gt_9_3mmd  = int(cl_config.get("cl_mmd_cal_not_in_zs_gt_9_3mmd", 1))

# 其他
self.allow_bi_fx_strict = int(cl_config.get("allow_bi_fx_strict", 0))
self.kline_qk           = cl_config.get("kline_qk", Config.KLINE_QK_NONE.value)
```

---

## 二、实现分阶段计划

### Phase 0：配置补全 + beichi_pz 修复 + zss_is_qs 完善

**目标**：把所有缺失的配置项读入 `__init__`，修复已知 bug。

**任务清单**：

- [ ] 0.1 在 `__init__` 中新增所有缺失的配置读取（线段/走势段/中枢/买卖点/其他）
- [ ] 0.2 修复 `beichi_pz()`：当前 `enter_ld` 错误地用了 `now_line.get_ld(self)` 两次
- [ ] 0.3 完善 `zss_is_qs()`：根据 `zs_wzgx` 配置（zgd / zggdd / gd）判断两中枢趋势关系
- [ ] 0.4 新增数据容器：`self.zsds`, `self.qsds`, `self.zsd_zss`, `self.qsd_zss` 等
- [ ] 0.5 更新 `get_zsds()` / `get_qsds()` / `get_zsd_zss()` / `get_qsd_zss()` 返回容器

**验证标准**：配置项可正确读取，不影响已有功能。

---

### Phase 1：线段构建 (`_build_xds`)

**目标**：实现特征序列方法的线段划分，这是整个系统最核心也最复杂的部分。

**算法概要**：

1. 从笔列表 `self.bis` 中，取与线段方向相反的笔作为**特征序列**元素
2. 对特征序列做**包含处理**（方向同K线合并：向上取高高，向下取低低）→ 生成 `TZXL` 对象
3. 在合并后的特征序列中寻找**序列分型** (`XLFX`)：连续三个 `TZXL` 构成顶/底分型
4. 序列分型确认 → 确定线段的结束点和新线段的起始点 → 生成 `XD` 对象
5. 线段内笔破坏检测（`xd_allow_bi_pohuai` 配置）
6. 线段拆分处理（非标准/中枢扩展/中枢多段/反向中枢）

**任务清单**：

- [ ] 1.1 特征序列构建：从笔列表提取反方向笔
- [ ] 1.2 特征序列包含处理：合并成 `TZXL` 对象
- [ ] 1.3 序列分型检测：在 `TZXL` 序列中寻找 `XLFX`
- [ ] 1.4 线段生成：基于 `XLFX` 创建 `XD` 对象，填充 `start_line` / `end_line` / `ding_fx` / `di_fx`
- [ ] 1.5 线段高低点更新（`_update_xd_highlow`）：根据 `xd_qj` 配置
- [ ] 1.6 笔破坏处理：根据 `xd_allow_bi_pohuai` 配置
- [ ] 1.7 线段拆分 — 非标准拆分：起始/结束非最高/最低 → 拆分（≥9笔）
- [ ] 1.8 线段拆分 — 中枢扩展拆分：相邻中枢重叠 → 拆分
- [ ] 1.9 线段拆分 — 中枢多段拆分：中枢内超过 N 段（默认11）→ 拆分
- [ ] 1.10 线段拆分 — 反向中枢拆分：线段内出现与线段方向相反的中枢 → 拆分
- [ ] 1.11 未完成线段处理：最后一段未形成序列分型时的处理

**关键数据结构**：

```python
# TZXL — 特征序列元素
TZXL(bh_direction, line, pre_line, line_bad, done)
  .max / .min  → 经包含处理后的高低值
  .lines       → 被包含合并的笔列表

# XLFX — 序列分型  
XLFX(type, xl, xls=[TZXL, TZXL, TZXL], done)
  .type        → "ding" / "di"
  .qk          → 序列分型是否有缺口
  .is_line_bad → 是否有笔包含情况
  .fx_high / .fx_low → 分型区间

# XD — 线段
XD(start, end, start_line, end_line, type, ding_fx, di_fx, index)
  .tzxls       → 特征序列列表
  .done        → 线段是否完成
  .is_split    → 拆分原因
```

**验证标准**：与 `cl_pyarmor` 的 `get_xds()` 输出对比，线段数量、方向、起止位置一致率 > 95%。

---

### Phase 2：中枢体系重构 — 4 种中枢类型

**目标**：实现标准中枢(bz)、段内中枢(dn)、方向中枢(fx)、分类中枢(fl) 四种类型。

**算法概要**：

| 类型 | 说明 | 结束条件 |
|------|------|----------|
| 标准中枢 (bz) | 前三段重合区间维持 | 三类买卖点（线不回中枢） |
| 段内中枢 (dn) | 新线段开始时重新计算 | 新线段起始 or 三类买卖点 |
| 方向中枢 (fx) | 进入/离开段方向相反 | 方向不匹配 or 三类买卖点 |
| 分类中枢 (fl) | 标准+段内结合 | 超9段 or 三类买卖点 |

**任务清单**：

- [ ] 2.1 重构 `create_dn_zs()`：使其支持 4 种中枢类型
- [ ] 2.2 实现标准中枢 (`_build_zs_bz`)：前三段确立，后续段在 zg-zd 区间内延伸，离开即结束
- [ ] 2.3 实现方向中枢 (`_build_zs_fx`)：进入段与离开段方向相反
- [ ] 2.4 实现分类中枢 (`_build_zs_fl`)：类似段内中枢但前一段中枢可跨段延续
- [ ] 2.5 多中枢类型支持：`zs_bi_type` / `zs_xd_type` 是列表，需对每种类型分别计算
- [ ] 2.6 中枢高低点：根据 `zs_qj` 计算线的 `zs_high` / `zs_low`
- [ ] 2.7 中枢重叠区间：根据 `zs_cd` (three / more) 计算 zg/zd
- [ ] 2.8 中枢方向标记：`up` / `down` / `zd`（震荡）
- [ ] 2.9 中枢 `done` / `real` 状态管理

**验证标准**：与 `cl_pyarmor` 的 `get_bi_zss()` / `get_xd_zss()` 输出对比。

---

### Phase 3：背驰体系

**目标**：实现完整的背驰判断。

**背驰类型**：

| 类型 | 代码 | 说明 |
|------|------|------|
| 笔背驰 | `bi` | 连续三笔，第三笔创新高/低且力度减弱 |
| 线段背驰 | `xd` | 连续三段，第三段创新高/低且力度减弱 |
| 走势段背驰 | `zsd` | 同上，走势段级别 |
| 盘整背驰 | `pz` | 中枢进入段 vs 离开段力度减弱 |
| 趋势背驰 | `qs` | 两个同向不重叠中枢，最后一段力度减弱 |

**任务清单**：

- [ ] 3.1 修复 `beichi_pz()`：正确取 `enter_line` 的力度
- [ ] 3.2 实现 `beichi_qs()`：找最后两个不重叠同向中枢，比较力度
- [ ] 3.3 实现笔背驰 (`_calc_bi_bc`)：三笔背驰判断
- [ ] 3.4 实现线段背驰 (`_calc_xd_bc`)：三段背驰判断
- [ ] 3.5 走势段背驰：类似线段背驰
- [ ] 3.6 背驰结果写入 `BI.bcs` / `XD.bcs` / `BI.zs_type_bcs` / `XD.zs_type_bcs`

**验证标准**：与 `cl_pyarmor` 的 `bi.line_bcs()` / `xd.line_bcs()` 输出对比。

---

### Phase 4：买卖点体系

**目标**：实现 1/2/3/l2/l3 买卖点。

**买卖点规则**（可通过配置开关控制）：

| 买卖点 | 规则 | 对应配置 |
|--------|------|----------|
| 1buy | 两中枢趋势+背驰 | `cl_mmd_cal_qs_1mmd` |
| 1buy | 3sell后创新低+背驰 | `cl_mmd_cal_not_qs_3mmd_1mmd` / `cl_mmd_cal_qs_3mmd_1mmd` |
| 2buy | 趋势后不创新低 | `cl_mmd_cal_qs_not_lh_2mmd` |
| 2buy | 趋势创新低后背驰 | `cl_mmd_cal_qs_bc_2mmd` |
| 2buy | 3buy后不创新低/背驰 | `cl_mmd_cal_3mmd_not_lh_bc_2mmd` |
| 2buy | 1buy后不创新低 | `cl_mmd_cal_1mmd_not_lh_2mmd` |
| 3buy | 回调不进入中枢(<9段) | `cl_mmd_cal_not_in_zs_3mmd` |
| 3buy | 回调不进入中枢(≥9段) | `cl_mmd_cal_not_in_zs_gt_9_3mmd` |
| l2buy | 类二买（自定义规则） | `user_custom_mmd` |
| l3buy | 类三买（自定义规则） | `user_custom_mmd` |

> 卖点规则与买点完全对称。

**任务清单**：

- [ ] 4.1 实现笔买卖点计算 (`_calc_bi_mmd`)
- [ ] 4.2 实现 3 类买卖点：离开中枢后回调不入中枢
- [ ] 4.3 实现 1 类买卖点：趋势背驰 + 3类后背驰
- [ ] 4.4 实现 2 类买卖点：趋势后不创新高低 / 1类后不创新高低
- [ ] 4.5 实现线段买卖点（结构同笔，用 `xd_zss`）
- [ ] 4.6 接入 `user_custom_mmd()` 自定义类买卖点（l2buy/l3buy 等）
- [ ] 4.7 多中枢类型下的买卖点：每种 `zs_type` 单独计算，写入 `zs_type_mmds`

**验证标准**：与 `cl_pyarmor` 的 `bi.line_mmds()` / `xd.line_mmds()` 输出对比。

---

### Phase 5：走势段 + 趋势段

**目标**：实现走势段 (ZSD) 和趋势段 (QSD) 的构建。

**说明**：走势段是以线段为基础元素，用类似线段构建的方式（特征序列+序列分型）构建更高级别的段。趋势段则以走势段为基础再构建一层。

**任务清单**：

- [ ] 5.1 实现走势段构建 (`_build_zsds`)：以线段为元素，特征序列方法
- [ ] 5.2 走势段高低点更新：根据 `zsd_qj` 配置
- [ ] 5.3 走势段中枢构建 (`_build_zsd_zss`)
- [ ] 5.4 走势段买卖点/背驰
- [ ] 5.5 趋势段构建 (`_build_qsds`)：以走势段为元素
- [ ] 5.6 趋势段中枢/买卖点/背驰

**验证标准**：与 `cl_pyarmor` 的 `get_zsds()` / `get_qsds()` 输出对比。

---

### Phase 6：计算流程整合 + 增量优化

**目标**：把以上所有模块串联到 `_full_compute()` 和 `_incremental_compute()` 中。

**任务清单**：

- [ ] 6.1 更新 `_full_compute()` 调用链：
  ```
  src_klines → cl_klines → fxs → bis → xds → zsds → qsds
  → bi_zss → xd_zss → zsd_zss → qsd_zss
  → bi_bc → xd_bc → zsd_bc → qsd_bc
  → bi_mmd → xd_mmd → zsd_mmd → qsd_mmd
  → user_custom_mmd
  ```
- [ ] 6.2 优化 `_incremental_compute()`：只从变化点开始重算
- [ ] 6.3 `get_last_bi_zs()` / `get_last_xd_zs()` 实现：倒推最后几笔/段的中枢
- [ ] 6.4 确保 `get_bi_zss(zs_type)` / `get_xd_zss(zs_type)` 支持按类型过滤

---

## 三、测试方案

### 3.1 对比测试框架

创建 `tests/test_cl_open_vs_pyarmor.py`：

```python
"""
对比测试框架：cl_open vs cl_pyarmor

对同一组行情数据，分别用两个引擎计算，逐项对比输出。
"""
import ccxt
import pandas as pd
from chanlun.cl_open import CL as CL_Open
from chanlun.cl_pyarmor import CL as CL_Pyarmor

# 数据获取
def fetch_klines(symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

# 对比函数
def compare_output(cd_open, cd_pyarmor, label: str):
    results = {}
    # 1. 缠论K线数量
    results["cl_klines"] = len(cd_open.get_cl_klines()) == len(cd_pyarmor.get_cl_klines())
    # 2. 分型数量和位置
    results["fxs_count"] = len(cd_open.get_fxs()) == len(cd_pyarmor.get_fxs())
    # 3. 笔数量
    results["bis_count"] = len(cd_open.get_bis()) == len(cd_pyarmor.get_bis())
    # 4. 线段
    results["xds_count"] = len(cd_open.get_xds()) == len(cd_pyarmor.get_xds())
    # 5. 中枢
    results["bi_zss_count"] = len(cd_open.get_bi_zss()) == len(cd_pyarmor.get_bi_zss())
    results["xd_zss_count"] = len(cd_open.get_xd_zss()) == len(cd_pyarmor.get_xd_zss())
    # 6. 买卖点
    open_bis = cd_open.get_bis()
    pyarmor_bis = cd_pyarmor.get_bis()
    # ... 逐笔对比 mmds / bcs
    return results
```

### 3.2 测试数据矩阵

| 标的 | 周期 | K线数量 | 目的 |
|------|------|---------|------|
| BTC/USDT | 1m | 1000 | 高频，大量分型/笔 |
| BTC/USDT | 5m | 1000 | 短线级别 |
| BTC/USDT | 15m | 1000 | 中频 |
| BTC/USDT | 30m | 1000 | 中频 |
| BTC/USDT | 60m | 1000 | 小时级别 |
| BTC/USDT | 4h | 1000 | 4小时级别 |
| BTC/USDT | 1d | 1000 | 日线级别 |
| BTC/USDT | 1w | 500 | 周线级别 |
| ETH/USDT | 5m | 1000 | 第二标的验证 |
| ETH/USDT | 60m | 1000 | 第二标的验证 |
| ETH/USDT | 1d | 1000 | 第二标的验证 |

### 3.3 测试配置矩阵

测试时需覆盖的核心配置组合：

| 维度 | 可选值 | 默认 |
|------|--------|------|
| `bi_type` | old / new / jdb / dd | old |
| `bi_bzh` | yes / no | yes |
| `bi_qj` | dd / ck / k | dd |
| `fx_bh` | yes / no / dingdi / diding / no_qbh / no_hbq | yes |
| `zs_bi_type` | bz / dn / fx / fl | [bz] |
| `zs_cd` | three / more | three |
| `zs_wzgx` | zgd / zggdd / gd | gd |

> 不需要穷举所有组合，选择**默认配置 + 每维度单独变化**即可，约 20 组测试。

### 3.4 对比指标

每组测试输出以下对比指标：

```
=== BTC/USDT 60m (默认配置) ===
缠论K线: open=342 pyarmor=342 match=✅
分型数:  open=128 pyarmor=128 match=✅
笔数:    open=64  pyarmor=64  match=✅
线段数:  open=12  pyarmor=12  match=✅
笔中枢:  open=8   pyarmor=8   match=✅
线段中枢: open=2   pyarmor=2   match=✅
走势段:  open=3   pyarmor=3   match=✅

--- 笔买卖点差异 ---
笔[23]: open=[3buy] pyarmor=[3buy]  ✅
笔[45]: open=[]     pyarmor=[2buy]  ❌ ← 需排查

--- 笔背驰差异 ---
笔[60]: open=[pz]   pyarmor=[pz,bi] ❌ ← 缺 bi 背驰
```

### 3.5 Phase 级别的验证节点

| Phase | 验证内容 | Pass 标准 |
|-------|---------|-----------|
| Phase 0 | 配置读取 + 现有功能不退化 | 原有 test 通过 |
| Phase 1 | 线段数量/方向/位置 | 与 pyarmor 一致率 > 95% |
| Phase 2 | 中枢数量/类型/区间 | 与 pyarmor 一致率 > 90% |
| Phase 3 | 背驰标记完整性 | 与 pyarmor 一致率 > 85% |
| Phase 4 | 买卖点命中 | 与 pyarmor 一致率 > 85% |
| Phase 5 | 走势段/趋势段 | 与 pyarmor 一致率 > 80% |
| Phase 6 | 全流程集成 | 回测结果偏差 < 5% |

---

## 四、实现顺序与依赖关系

```
Phase 0 (配置/Bug修复)
    │
    ▼
Phase 1 (线段 XD)  ←─── 核心难点，预计代码量最大 (300-500行)
    │
    ├──▶ Phase 2 (中枢体系)
    │       │
    │       ├──▶ Phase 3 (背驰)
    │       │       │
    │       │       └──▶ Phase 4 (买卖点)
    │       │
    │       └──▶ Phase 5 (走势段/趋势段) ─── 可与 Phase 3/4 并行
    │
    └──▶ Phase 6 (整合+增量优化) ─── 最后收尾
```

---

## 五、风险与注意事项

1. **PyArmor 试用到期**：2026-03-20 过期，必须在此前完成所有对比测试并缓存对比结果
2. **线段划分差异**：特征序列方法的边界情况很多，初始实现很可能与 pyarmor 不完全一致，需要逐 case 调试
3. **未完成线段/笔**：实时行情中最后一笔/段经常处于"未完成"状态，需正确处理 `done` 标记
4. **增量计算性能**：目前全量重算可接受，后续如有性能问题需优化
5. **浮点精度**：价格比较时需注意浮点精度问题

---

## 六、文件结构

```
plan/
  cl_open_implementation_plan.md    ← 本文件
tests/
  test_cl_open_vs_pyarmor.py        ← 对比测试脚本
  test_data/                        ← 缓存的测试数据（避免重复请求交易所）
src/chanlun/
  cl_open.py                        ← 主实现文件（修改）
  cl_interface.py                   ← 接口定义（只读参考）
  cl_pyarmor.py                     ← 加密版（对比参考）
```
