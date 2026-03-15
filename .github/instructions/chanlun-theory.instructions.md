---
description: "缠论（缠中说禅理论）核心概念详解，包括K线合并、分型、笔、线段、走势中枢、背驰、买卖点的定义与规则，以及在 cl_open.py / cl_interface.py 中的代码实现映射。Use when working on CL engine, cl_open.py, cl_interface.py, or any code involving 缠论 computation."
---

# 缠论理论核心概念与代码映射

> 理论来源：缠中说禅博客教程 001~107（[博客存档](https://github.com/chanxishe/chzhshch-blog/tree/main/docs/_shared/stocks)）  
> 代码实现：`src/chanlun/cl_open.py`（引擎）、`src/chanlun/cl_interface.py`（数据结构）

---

## 一、K 线合并（包含处理）

### 规则
相邻两根 K 线的「包含关系」：`高1 >= 高2 且 低1 <= 低2`（或反向）时，称为包含，需合并。

合并方向取决于当前**趋势方向**：
- **上升趋势**（前两合并K线的高点依次升高）：取两者最高的 H，取两者最高的 L → 保留高点
- **下降趋势**（前两合并K线的高点依次降低）：取两者最低的 H，取两者最低的 L → 保留低点

### 代码
```python
# cl_interface.py
class CLKline:
    k_index: int        # 极值原始K线的索引（Fix 6：取提供极值的那根，非最后一根）
    klines: List[Kline] # 被合并进来的原始K线列表
    h, l: float         # 合并后的高低点
```
```python
# cl_open.py
def _merge_kline(self, k: Kline) -> None: ...  # 合并逻辑入口
```

---

## 二、分型（FX）

### 定义
连续三根合并K线：
- **顶分型**（`fx_type = 'ding'` / `FX_TYPE_DING`）：中间K线的最高点 > 左侧 且 > 右侧
- **底分型**（`fx_type = 'di'` / `FX_TYPE_DI`）：中间K线的最低点 < 左侧 且 < 右侧

### 有效性（间距校验）
相邻两个分型之间必须有足够的**独立K线**（非合并的原始K线）间隔：
- `fx_qj = fx_qj_k`（当前配置）：按原始K线数计算，默认至少1根独立K线（间距>=2）
- `fx_qy = fx_qy_three`：前后各取3根合并K线校验分型有效性

### 代码
```python
# cl_interface.py
@dataclass
class FX:
    index: int          # 分型序号
    k: CLKline          # 分型的中间合并K线
    klines: List[CLKline]  # 组成分型的三根合并K线
    type: str           # 'ding'（顶）或 'di'（底）
    val: float          # 分型值（顶取最高点，底取最低点）
    real: bool          # 是否已确认（非待确认）
```
```python
# cl_open.py
def _check_fx(self) -> None: ...  # 分型识别
def _bi_fx_valid(self, pre_fx, cur_fx) -> bool: ...  # 分型间距校验
```

---

## 三、笔（BI）

### 定义
两个相邻且**方向相反**的有效分型之间的连线：
- **上升笔**（`bi_type = 'up'`）：底分型 → 顶分型
- **下降笔**（`bi_type = 'down'`）：顶分型 → 底分型

### bi_type_old 模式规则（当前配置）
1. 两端分型之间至少有 **5 根合并K线**（含两端）
2. 起止分型之间不得有同向更极端的分型
3. `fx_check_k_nums = 13`：校验窗口（前后各13根原始K线）

### 笔拆分（bi_split）
当笔内部出现「强度足够大的反向走势」时，该笔被拆分：
- `bi_split_k_cross_nums = "20,1"`：拆分条件阈值
- `_bi_special_bi_split`：核心拆分方法
  - Fix 7：使用 `k_gap >= 4`（原始K线间距）而非 cl_gap 判断缺口
  - `_find_split1_from_triplet`：只用三元组候选，不往前追溯

### 代码
```python
# cl_interface.py
@dataclass
class BI:
    index: int
    type: str           # 'up' 或 'down'
    start: FX           # 起始分型
    end: FX             # 结束分型
    fx_num: int         # 包含的分型数（含两端）
    is_done: bool       # 是否已完成（末端分型已确认）
```
```python
# cl_open.py
def _build_bis(self) -> None: ...       # 笔构建主逻辑
def _bi_special_bi_split(self) -> None: # 笔拆分
```

---

## 四、线段（XD）

### 定义
由**至少三笔**构成（方向交替），从底（顶）分型开始，到顶（底）分型结束。

### 成段条件（1 型线段）
连续三笔 A-B-C（如：上-下-上）：
- 第三笔 C 的终点超过第一笔 A 的终点方向 → 前三笔构成一条线段

### 被破坏与延伸
线段形成后，若后续走势破坏了线段的端点：
- 破坏意味着线段终止，需重新确定端点
- `xd_bzh = xd_bzh_no`（当前配置）：不进行端点修正

### 代码
```python
# cl_interface.py
@dataclass
class XD:
    index: int
    type: str           # 'up' 或 'down'
    start: BI           # 起始笔
    end: BI             # 结束笔
    bis: List[BI]       # 包含的笔列表
    is_done: bool
```
```python
# cl_open.py
def _build_xds(self) -> None: ...        # 线段构建
def _xd_is_line_bad(self, ...) -> bool:  # 线段有效性检查（Fix 2：按位置判断，非分型号）
def _xd_cal_line_xlfx(self, ...) -> ...: # 线型分型计算
```

---

## 五、走势中枢（ZS）

### 定义
某级别走势中，**至少三个连续次级别走势类型**的重叠区间：

```
ZS_HIGH = min(高1, 高2, 高3)
ZS_LOW  = max(低1, 低2, 低3)
```
前提：`ZS_LOW < ZS_HIGH`（必须存在重叠）

### 中枢的「生、住、坏、灭」
- **生**：首三段形成
- **住（延伸）**：后续走势继续在中枢内震荡
- **坏（被突破）**：离开中枢的走势不回来 → 趋势形成
- **灭（完成）**：下一个同向中枢出现

### 代码
```python
# cl_interface.py
@dataclass
class ZS:
    index: int
    type: str           # 'up' 或 'down'（中枢的趋势方向）
    start: LINE         # 起始线（笔或线段）
    end: LINE           # 结束线
    zs_type: str        # 'bi' 或 'xd'（笔中枢 or 线段中枢）
    high: float         # ZS 上沿
    low: float          # ZS 下沿
    zg: float           # 中枢最高高点
    zd: float           # 中枢最低低点
```

---

## 六、背驰（Beichi）

### 定义
在趋势中，**后一段走势**相比**前一段走势**力度减弱，预示趋势将要反转。

### MACD 面积背驰判断法（主要方法）
在上涨趋势中：
- 将走势分为「进入中枢段」和「离开中枢段」
- 对比两段的 **MACD 红柱/绿柱面积**（DIF-DEA 围成的面积）
- 若后段面积 < 前段面积 + 价格未创新高（或新高幅度更小）→ 顶背驰

下跌反之。

### 代码
```python
# cl_interface.py
@dataclass
class MACD_INFOS:
    dif_up_cross_num: int    # DIF 上穿零轴次数
    dif_down_cross_num: int  # DIF 下穿零轴次数
    gold_cross_num: int      # 金叉次数
    die_cross_num: int       # 死叉次数
    last_dif: float
    last_dea: float

# BI/XD 对象上的背驰相关属性
bi.bc_bi: str      # 笔背驰类型
bi.bc_xd: str      # 线段背驰类型
```

---

## 七、买卖点

### 理论定义
| 类型 | 名称 | 触发条件 |
|------|------|---------|
| 一买 | 第一类买点 | 下跌趋势末端出现底背驰 |
| 二买 | 第二类买点 | 第一买点后回调不破前低，再次向上 |
| 三买 | 第三类买点 | 上涨突破前高后，回调不破前中枢高点（ZS_HIGH） |
| 一卖 | 第一类卖点 | 上涨趋势末端出现顶背驰 |
| 二卖 | 第二类卖点 | 第一卖点后反弹不超前高，再次向下 |
| 三卖 | 第三类卖点 | 下跌破前低后，反弹不回前中枢低点（ZS_LOW） |

### 代码
```python
# cl_interface.py
# BI/XD 对象上
bi.mmd: List[str]   # 买卖点列表，如 ['1buy', '2sell', 'l3sell']
# 前缀: 'l' = 类买卖点（不严格满足）

# Config 中
mmd_cal_qs_type: str  # 买卖点计算使用的中枢类型
```

---

## 八、级别关系

缠论的核心是**多级别嵌套**结构：

```
1分钟K线的笔 → 1分钟的线段 → 1分钟的走势
    ↕（次级别）
5分钟K线的笔 → 5分钟的线段 → 5分钟的走势
    ↕（次级别）
30分钟K线的笔 → ...
```

每个级别的「走势」由**次级别的走势类型**构成。中枢的有效性需要看次级别。

在代码中，多级别分析通过传入不同时间周期的 `CL` 实例来实现，统一通过 `web_batch_get_cl_datas()` 获取多级别数据。

---

## 九、ICL 接口速查

```python
# src/chanlun/cl_interface.py — 所有策略代码通过此接口操作
class ICL:
    def process_klines(self, klines: pd.DataFrame) -> 'ICL': ...
    def get_klines(self) -> List[CLKline]: ...     # 合并后的缠论K线
    def get_src_klines(self) -> List[Kline]: ...   # 原始K线
    def get_fxs(self) -> List[FX]: ...             # 分型列表
    def get_bis(self) -> List[BI]: ...             # 笔列表
    def get_xds(self) -> List[XD]: ...             # 线段列表
    def get_bi_zss(self) -> List[ZS]: ...          # 笔中枢列表
    def get_xd_zss(self) -> List[ZS]: ...          # 线段中枢列表
    def get_mmds(self) -> List[...]: ...           # 买卖点
    def get_idx(self) -> dict: ...                 # MACD/MA/BOLL 等技术指标
```
