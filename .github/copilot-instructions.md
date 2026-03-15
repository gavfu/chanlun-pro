# chanlun-pro — GitHub Copilot 项目说明

## 项目定位

**chanlun-pro** 是一套基于**缠中说禅理论（缠论）**的量化分析与交易系统。
原作者 Wang Xu（[yijixiuxin/chanlun-pro](https://github.com/yijixiuxin/chanlun-pro)），我们在 `chan` 分支上进行二次开发：用自行实现的开源缠论引擎 `cl_open.py` 替代了原 PyArmor 加密版 `cl_pyarmor.py`，以摆脱加密授权依赖。

缠论理论参见：[`docs/`](../docs/) 及 [`.github/instructions/chanlun-theory.instructions.md`](./instructions/chanlun-theory.instructions.md)。

---

## 一、关键文件速查

| 文件 | 作用 |
|------|------|
| `src/chanlun/cl.py` | CL 入口 shim，当前：`from chanlun.cl_open import CL` |
| `src/chanlun/cl_open.py` | **开源缠论引擎**（核心开发目标），实现 `ICL` 接口 |
| `src/chanlun/cl_pyarmor.py` | PyArmor 加密的原版引擎（参考对比用，勿修改） |
| `src/chanlun/cl_interface.py` | 所有数据结构定义（`Kline`、`CLKline`、`FX`、`BI`、`XD`、`ZS`、`Config`、`ICL`） |
| `src/chanlun/cl_utils.py` | 工具函数：`web_batch_get_cl_datas()`、`query_cl_chart_config()` 等 |
| `src/chanlun/file_db.py` | Pickle 文件缓存，`cl.CL` 的实例化在此 (`FileCacheDB.get_web_cl_data`) |
| `web/chanlun_chart/app.py` | Flask+Tornado Web 服务入口（9900 端口） |
| `web/chanlun_chart/cl_app/__init__.py` | Flask 路由、APScheduler 任务 |
| `src/chanlun/exchange/` | 多市场行情接口统一抽象（`ExchangeBase`） |
| `src/chanlun/backtesting/` | 回测引擎（策略代码与实盘共用） |

---

## 二、架构分层

```
Web UI (Flask+Tornado+TradingView) / Jupyter Notebook
    └── 业务层：Strategy / BackTest / Monitor / Trader
        └── 核心层：CL 缠论引擎 / MultiLevelAnalyse / kcharts
            └── 数据层：Exchange 统一接口 / DB(SQLAlchemy) / FileDB(Pickle)
                └── 基础：config.py / fun.py / base.py
```

详见 [`docs/02-架构设计与模块划分.md`](../docs/02-架构设计与模块划分.md)。

---

## 三、CL 引擎关键约定

### 接口

所有代码通过 `from chanlun.cl import CL` 获取引擎类（不要直接 import `cl_open` 或 `cl_pyarmor`）。

### 核心配置参数（`Config` dataclass）

| 参数 | 当前使用值 | 说明 |
|------|-----------|------|
| `bi_type` | `bi_type_old` | 原始笔计算方式（默认） |
| `fx_qj` | `fx_qj_k` | 分型间距用K线数计算 |
| `fx_qy` | `fx_qy_three` | 分型有效性：前后各3根 |
| `bi_fx_cgd` | `bi_fx_cgd_yes` | 缺口处理：连续同向分型视为缺口 |
| `fx_check_k_nums` | `13` | bi_type_old 模式下分型校验K线数 |
| `bi_split_k_cross_nums` | `"20,1"` | 笔拆分参数 |
| `xd_bzh` | `xd_bzh_no` | 线段不修正 |

### cl_open.py 实现进度与已知差异

- ✅ K 线合并（Fix 6：k_index 取极值原始K线）
- ✅ 分型识别（与 cl_pyarmor 完全一致）
- ✅ 笔构建（Fix 1~5, Fix 8: 延伸使用严格不等式 `<`/`>`）
- ✅ 线段（XD）：全部5个测试数据集 100% 匹配
- ✅ 笔拆分（Fix 7: k_gap 重叠检测；Fix 9: `_find_split1_from_triplet` 混合策略 — 先按位置+`_bi_fx_valid`校验，回退用极值排序）
- ✅ MACD 指标（Fix 10: `hist = hist * 2` 对齐 talib 输出）
- ✅ 走势段（ZSD）& 趋势走势段（QSD）：全部5个测试数据集与 cl_pyarmor 输出一致（zsds/qsds/zsd_zss/qsd_zss 均为 0/0）
- ✅ ZSD 算法隔离验证：将 cl_pyarmor 的 XD 列表注入 cl_open，三个大数据集（BTC4h5k/ETH4h5k/BTCd3k）均与 cl_pyarmor 100% 一致（ZSD/QSD内容、ZSD_ZSS精确匹配）
- ✅ BTC60/ETH60/BTCd 小数据集（500-1000根K线）：笔、线段完全匹配（0差异）
- ⚠️ 大数据集端到端差异：cl_open 与 cl_pyarmor 的笔列表在大数据集上存在分歧，cl_open 在某些边界情况下比 cl_pyarmor 更早确认笔（cl_gap=4 的情况），导致后续线段、ZSD 等结果不同。已排除 cl_gap 阈值（测试 cl_gap>=5 会破坏小数据集匹配），根因为 cl_pyarmor 在确认步骤有未知的额外校验逻辑。
- ⚠️ 笔端点精度差异数据：BTC5m 2笔, ETH5m 6笔（pre-split）；大数据集 BTC4h5k/ETH4h5k/BTCd3k 有级联差异

测试数据与诊断脚本在 `tests/`（`diag_xd_all.py`、`diag_zsd_all.py`、`diag_zsd_isolated.py` 为主要对比脚本）。

### ZSD/QSD 实现说明

`_build_zsds()` 和 `_build_qsds()` 使用与线段完全相同的特征序列算法（`_build_xds`），分别以 XD 和 ZSD 作为基本元素（LINE 接口通用）。

ZSD 模式通过 `self._zsd_mode = True` 激活，在 `_find_xd_end` 内施加额外约束：
1. **`len(tzxls) >= 5`**：走势段级别需要至少 5 个特征序列元素才能查找分型（XD 级别只需 3 个）
2. **`prev_xl.line_bad → skip`**：前驱特征序列元素为 line_bad 时，当前分型被跳过（cl_pyarmor 行为）
3. **`_split_xds` 跳过**：走势段不进行拆分后处理

关键修正（fallback 过滤）：当 `_find_first_xd_start` 对 XD 序列找不到有效序列分型时，会触发兜底 fallback 返回 `(type, 0)`，导致生成一个 done=False 的部分走势段。cl_pyarmor 不包含此类「无有效起点」的尾部部分段，因此在 `_build_zsds`/`_build_qsds` 中过滤：若结果全为 done=False，则清空。

大数据集（BTC4h5k/ETH4h5k/BTCd3k）：ZSD 算法本身 100% 正确（隔离注入 pyarmor XD 验证），端到端差异源于 XD 层面分歧，非 ZSD 算法问题。

---

## 四、编码约定

- Python 3.11+，严格类型注解
- 所有缠论数据对象均为 dataclass，定义在 `cl_interface.py`，**不要在 `cl_open.py` 里重定义**
- 策略代码通过 `ICL` 接口操作缠论数据，不依赖具体实现类
- 回测与实盘使用相同策略代码，仅切换 `MarketDatas` 和 `Trader` 实现
- 交易所访问：`from chanlun.exchange import get_exchange`（单例+工厂）
- 配置文件：`src/chanlun/config.py`（从 `config.py.demo` 复制，不提交到 git）

---

## 五、开发环境

```bash
cd /Users/gavfu/github/gavfu/chanlun-pro
source .venv/bin/activate          # Python 3.11 虚拟环境

# 运行 Web 服务
cd web/chanlun_chart && python app.py

# 运行对比测试
python tests/diag_xd_all.py
```

---

## 六、文档索引

| 文档 | 内容 |
|------|------|
| [`docs/01-项目概览.md`](../docs/01-项目概览.md) | 项目定位、支持市场、目录结构 |
| [`docs/02-架构设计与模块划分.md`](../docs/02-架构设计与模块划分.md) | 分层架构、模块依赖图 |
| [`docs/03-缠论计算引擎.md`](../docs/03-缠论计算引擎.md) | ICL 接口、数据结构、计算流程 |
| [`docs/04-行情数据与交易所接入.md`](../docs/04-行情数据与交易所接入.md) | Exchange 抽象、各市场数据源 |
| [`docs/05-策略编写与回测系统.md`](../docs/05-策略编写与回测系统.md) | 策略 API、回测框架 |
| [`docs/08-PyArmor加密模块分析.md`](../docs/08-PyArmor加密模块分析.md) | cl_pyarmor vs cl_open 对比历史 |
| [`.github/instructions/chanlun-theory.instructions.md`](./instructions/chanlun-theory.instructions.md) | **缠论理论核心概念与代码映射** |
