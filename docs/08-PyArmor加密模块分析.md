# PyArmor 加密模块分析

## 背景

chanlun-pro 项目的缠论核心计算引擎最初使用 [PyArmor](https://pyarmor.dashingsoft.com/) 进行了加密保护。加密后的文件是一个约 81KB 的二进制文件（`cl_pyarmor.py`），无法直接阅读或反编译。使用加密版需要向原作者购买 PyArmor 授权文件。

我们选择**不购买授权，而是自己实现缠论指标**。在 `chan` 分支上，我们编写了 `cl_open.py` 作为 `cl_pyarmor.py` 的开源替代实现。第一阶段目标是实现分型和笔，以此为基础跑通策略回测流程。后续将继续补全线段、中枢、买卖点等完整缠论指标。

### Git 历史

```
# chan 分支上的关键 commit：

157e2e7  2026-02-09  gavfu   Add cl_open           ← 我们实现的开源 CL 引擎
caab1b4  2026-02-27  wangxu  增加缠论配置的重置/导入/导出功能  ← 上游最新 (已 merge)
```

`cl_open.py` 的实现采用了**最小改动原则**——只新增 `cl_open.py` 文件并修改 `cl.py` 的一行导入，不改动其他文件，以便后续从上游 `master` 分支 merge 新功能时减少冲突。

当前 `cl.py` 的导入指向 `cl_open.py`：

## PyArmor 运行时

```
src/
├── .pyarmor.ikey                     # PyArmor 身份密钥（二进制）
├── pyarmor_runtime_005445/           # PyArmor 运行时
│   ├── __init__.py
│   ├── 授权文件放入到此/              # 授权文件（pyarmor.rkey）放置目录
│   ├── darwin_aarch64/               # macOS ARM64
│   ├── darwin_arm64/
│   ├── darwin_x86_64/                # macOS Intel
│   ├── linux_aarch64/
│   ├── linux_armv7/
│   ├── linux_x86/
│   ├── linux_x86_64/
│   ├── windows_x86/
│   └── windows_x86_64/
└── chanlun/
    ├── cl.py                         # 当前：from cl_open import CL
    ├── cl_open.py                    # 开源实现（部分）
    └── cl_pyarmor.py                 # 加密实现（完整但不可用）
```

如需恢复使用加密版：
1. 获取 PyArmor 授权文件（`pyarmor.rkey`）
2. 放入 `src/pyarmor_runtime_005445/授权文件放入到此/`
3. 修改 `cl.py`：`from chanlun.cl_pyarmor import CL`

## 开源版 vs 加密版功能对照

### 已在 cl_open.py 中实现的功能

| 功能 | 方法 | 状态 | 说明 |
|------|------|------|------|
| K 线构建 | `_build_src_klines()` | ✅ 完整 | DataFrame → Kline 对象 |
| K 线合并（包含处理） | `_merge_klines()` | ✅ 完整 | 上升取高、下降取低，缺口检测 |
| 分型识别 | `_find_fxs()` | ✅ 完整 | 顶分型/底分型 |
| 分型过滤 | `_filter_fxs()` | ✅ 完整 | 顶底交替、包含关系检查 |
| 分型包含检查 | `_check_fx_bh()` | ✅ 完整 | 支持所有 `fx_bh` 配置选项 |
| 笔构建 | `_build_bis()` | ✅ 完整 | 支持 old/new/jdb/dd 四种类型 |
| 笔高低点 | `_update_bi_highlow()` | ✅ 完整 | 支持 dd/ck/k 三种区间 + 标准化 |
| 笔中枢 | `_build_bi_zss()` → `create_dn_zs()` | ✅ 基本实现 | 简易版段内中枢，3笔重叠+延伸 |
| MACD 计算 | `_compute_macd()` | ✅ 完整 | talib.MACD，参数可配 |
| 盘整背驰 | `beichi_pz()` | ⚠️ 简易实现 | 逻辑简化，可能与加密版有差异 |
| 趋势判断 | `zss_is_qs()` | ✅ 完整 | 两中枢位置关系判断 |

### 未在 cl_open.py 中实现的功能（返回空值）

| 功能 | 方法 | 返回值 | 重要程度 |
|------|------|--------|---------|
| **线段计算** | `get_xds()` | `self.xds`（始终为 `[]`） | 🔴 核心 |
| **走势段** | `get_zsds()` | `[]` | 🔴 核心 |
| **趋势段** | `get_qsds()` | `[]` | 🟡 高级 |
| **线段中枢** | `get_xd_zss()` | `self.xd_zss`（始终为 `[]`） | 🔴 核心 |
| **走势段中枢** | `get_zsd_zss()` | `[]` | 🟡 高级 |
| **趋势段中枢** | `get_qsd_zss()` | `[]` | 🟡 高级 |
| **趋势背驰** | `beichi_qs()` | `(False, [])` | 🔴 核心 |
| **买卖点判定** | (分散在各处) | 笔上的 mmds 始终为 `[]` | 🔴 核心 |

### 对现有策略的影响

由于线段、买卖点、背驰等核心功能缺失：

| 策略 | 可用性 | 影响 |
|------|--------|------|
| `StrategyDemo` | ❌ 不可用 | 依赖 `bi.line_mmds()`，MMD 始终为空 |
| `StrategyXDMMD` | ❌ 不可用 | 依赖线段和线段买卖点 |
| `StrategyZsdXdBi1MMD` | ❌ 不可用 | 依赖走势段、线段、笔多级别信号 |
| 所有其他策略 | ❌ 不可用 | 均依赖买卖点信号 |

**结论**：当前开源版 `cl_open.py` 虽然成功实现了从 K 线到笔的计算链路，但缺少线段以上级别的结构和买卖点判定，因此**所有内置策略在当前状态下都无法正常工作**。

## 需要补全的功能详解

以下按实现优先级排列，描述每个缺失功能的算法要求。

### 1. 买卖点判定（最优先）

即使线段未实现，也可以先在笔级别实现基本的买卖点。核心规则在 `cl_interface.py` 的文档注释和 `cookbook/docs/缠论买卖点和背驰规则.md` 中有详细描述。

**需要实现的逻辑**：

1. 在每根笔计算完成后，基于其所在的中枢结构判定买卖点
2. 调用 `user_custom_mmd()` 添加类二/类三买卖点
3. 调用 `beichi_pz()` 和 `beichi_qs()` 判断背驰

**关键数据流**：

```
笔列表 + 笔中枢列表
  → 遍历每根下跌笔：检查是否是 1买/2买/3买/类2买/类3买/下跌笔背驰买
  → 遍历每根上涨笔：检查是否是 1卖/2卖/3卖/类2卖/类3卖/上涨笔背驰卖
  → 填充 BI.mmds, BI.bcs
```

**判定规则摘要**（详见 `缠论计算引擎.md`）：

- **1买**：两同级别中枢下降趋势 + 趋势背驰
- **2买**：1买后回踩不创新低
- **3买**：向上离开中枢后回踩不入中枢
- **背驰通过 MACD 柱体面积比较**：`compare_ld_beichi()`

### 2. 线段计算（核心）

线段是由多笔组成的更高级别结构。按照缠论原文，线段端点通过**特征序列的分型**来确定。

**算法步骤**：

1. 从笔列表构建特征序列（`TZXL`）
   - 向上线段中，取所有下跌笔作为特征序列元素
   - 向下线段中，取所有上涨笔作为特征序列元素
2. 对特征序列做包含处理（类似 K 线包含）
3. 在处理后的特征序列中寻找分型（`XLFX`）
   - 顶分型 → 向上线段的结束
   - 底分型 → 向下线段的结束
4. 生成 `XD` 对象

**数据结构已就绪**：`TZXL`、`XLFX`、`XD` 类都已在 `cl_interface.py` 中定义。

**参考配置**：
- `xd_qj`：线段区间取值方式
- `xd_bi_pohuai`：笔破坏线段的判定规则

### 3. 趋势背驰（核心）

`beichi_qs()` 当前返回 `(False, [])`，需要实现：

```python
def beichi_qs(self, lines, zss, now_line):
    """
    条件：
    1. 至少 2 个同向中枢
    2. 中枢之间无重叠
    3. 最后一段创新极值
    4. 最后一段 MACD 力度 < 前一段
    """
    # 1. 找到同向的相邻中枢
    # 2. 检查中枢不重叠 (zss_is_qs)
    # 3. 比较 MACD 力度 (compare_ld_beichi)
    # 4. 返回 (是否背驰, 参与比较的线段列表)
```

### 4. 走势段与趋势段（高级）

走势段（ZSD）和趋势段（QSD）是线段之上更高级别的结构：
- **走势段**：多条线段形成的走势
- **趋势段**：多个走势段形成的大级别趋势

实现方式类似于从笔到线段的递归——用线段作为基础元素，构建更高级别的特征序列和分型。

### 5. 线段中枢 / 走势段中枢（依赖 3/4）

一旦线段和走势段实现，中枢构建可直接复用 `create_dn_zs()`：

```python
# 线段中枢
self.xd_zss = self.create_dn_zs("xd", self.xds)

# 走势段中枢
self.zsd_zss = self.create_dn_zs("zsd", self.zsds)
```

## 实现路线图

我们的计划是在 `chan` 分支上逐步补全 `cl_open.py`，最终实现完整的缠论指标体系。以下是分阶段计划：

### Phase 1：笔级别买卖点 ← 下一步

```
目标：在 _build_bi_zss() 之后，遍历笔和中枢，判定买卖点
依赖：已有的笔 + 笔中枢 + MACD
效果：StrategyDemo 和简单笔级别策略可用，可以开始回测
```

### Phase 2：线段计算

```
目标：实现特征序列 → 序列分型 → 线段构建
依赖：Phase 1 的笔
效果：StrategyXDMMD 可用，线段级别分析
```

### Phase 3：线段买卖点 + 趋势背驰

```
目标：在线段上计算买卖点，实现完整的趋势背驰判定
依赖：Phase 2 的线段 + 线段中枢
效果：所有策略可用
```

### Phase 4：走势段 / 趋势段

```
目标：实现走势段和趋势段（线段的递归）
依赖：Phase 2/3
效果：StrategyZsdXdBi1MMD 完全可用
```

## 当前可用的过渡方案

在逐步补全的过程中，已实现的笔 + 中枢 + MACD 已经足够构建基本策略：

### 方案：直接基于笔和中枢结构判断

当前已实现的笔 + 中枢 + MACD 已经足够构建基本策略，示例：

```python
class SimpleBI_Strategy(Strategy):
    def open(self, code, market_data, poss):
        cd = market_data.get_cl_data(code, market_data.frequencys[0])
        bis = cd.get_bis()
        bi_zss = cd.get_bi_zss("zs_type_bz")
        
        if len(bis) < 3 or len(bi_zss) < 1:
            return []
        
        last_bi = bis[-1] if bis[-1].is_done() else bis[-2]
        
        # 不依赖买卖点，而是直接用结构判断
        # 例如：下跌笔在中枢下方 + MACD 背驰
        zs = bi_zss[-1]
        if (last_bi.type == "down" 
            and last_bi.low < zs.zd  # 低于中枢下沿
            and self.bi_td(last_bi, cd)):  # 笔停顿
            
            # 手动检查 MACD 背驰
            if len(bis) >= 3:
                prev_same_dir = bis[-3] if bis[-3].type == last_bi.type else None
                if prev_same_dir:
                    from chanlun.cl_interface import compare_ld_beichi
                    ld1 = prev_same_dir.get_ld(cd)
                    ld2 = last_bi.get_ld(cd)
                    if compare_ld_beichi(ld1, ld2, "down"):
                        return [Operation(code=code, opt="buy", mmd="custom_bc_buy",
                                         loss_price=last_bi.low)]
        return []
```

### 可参考的其他开源缠论实现

实现线段等算法时，以下项目可作为参考：
- [czsc](https://github.com/waditu/czsc) — 较完整的缠论实现
- [chan.py](https://github.com/entropage/chan.py) — 另一个缠论实现

## 关键参考材料

理解缠论算法实现所需的核心资料：

1. **缠中说禅原文**（位于 `../../chanxishe/chzhshch-blog/docs/_shared/stocks/`）：
   - 教你炒股票 62：分型、笔与线段
   - 教你炒股票 65：再说说分型、笔、线段
   - 教你炒股票 67：线段的划分标准
   - 教你炒股票 71：线段划分标准的再分辨
   - 教你炒股票 77：一些概念的再分辨
   - 以及整个教你炒股票系列关于中枢、背驰、买卖点的课程

2. **项目内已有的数据结构定义**：`cl_interface.py`（1464 行），所有字段和方法都已定义好

3. **项目内的策略代码**：展示了各种数据结构的实际使用方式

4. **项目内的配置文档**：`cookbook/docs/缠论配置项说明.md`
