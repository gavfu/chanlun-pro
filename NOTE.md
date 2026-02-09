# 回测 Binance ETH/USDT (2025年) 操作指南

## 前置条件

### 1. 配置文件

确保 `src/chanlun/config.py` 已从 `config.py.demo` 复制并配置好。关键配置项：

```python
# 数据库配置（可用 sqlite，无需安装 MySQL）
DB_TYPE = "sqlite"
DB_DATABASE = "chanlun_klines"

# 如需科学上网访问 Binance（中国大陆地区需要）
PROXY_HOST = '127.0.0.1'
PROXY_PORT = 7890

# Binance API（仅下载公开行情数据可留空，但建议配置以获得更稳定的访问）
BINANCE_APIKEY = ''
BINANCE_SECRET = ''
```

### 2. 安装依赖（macOS）

macOS 上安装有几个注意事项：

```bash
cd /Users/gavfu/github/gavfu/chanlun-pro

# 1. Python 版本：需要 3.11.x（pyenv 管理）
pyenv install 3.11.11    # 如尚未安装
pyenv local 3.11.11

# 2. 安装 ta-lib C 库（macOS 必须通过 homebrew）
brew install ta-lib

# 3. 安装 TA-Lib Python 包（从源码编译，不能用项目自带的 Windows wheel）
pip install TA-Lib

# 4. 安装 pytdx（使用项目自带的 wheel）
pip install package/pytdx-1.72r2-py3-none-any.whl

# 5. 安装项目本体（editable 模式，不装依赖）+ 其他依赖
pip install -e . --no-deps
pip install "akshare>=1.16.98" "alpaca-py>=0.40.1" "apscheduler>=3.11.0" \
  "baostock>=0.8.9" "ccxt>=4.4.88" "chardet>=5.2.0" "dbutils>=3.1.1" \
  "dtaidistance>=2.3.13" "flask>=3.1.1" "flask-login>=0.6.3" "futu-api>=9.2.5208" \
  "gevent>=25.5.1" "ib-insync>=0.9.86" "ipywidgets>=8.1.7" "lark-oapi>=1.4.17" \
  "matplotlib>=3.10.3" "mytt>=2.9.3" "numpy==1.26.4" "openai>=1.84.0" \
  "pandas==2.1.0" "pinyin>=0.4.0" "playwright>=1.53.0" "polygon-api-client>=1.14.6" \
  "prettytable>=3.16.0" "pyarmor>=9.1.7" "pyarrow>=22.0.0" "pyecharts==2.0.8" \
  "pyfolio-reloaded>=0.9.9" "pymysql>=1.1.1" "pytest>=8.4.1" "redis>=6.2.0" \
  "requests>=2.32.3" "scipy>=1.15.3" "setuptools>=80.9.0" "sqlalchemy>=2.0.41" \
  "tenacity>=9.1.2" "tornado>=6.5.1" "tqdm>=4.67.1" "tqsdk>=3.8.2"

# 6. 添加 src/ 到 Python 路径（让 pyarmor_runtime_005445 可被找到）
SITE_DIR=$(python3 -c "import site; print(site.getsitepackages()[0])")
echo "$PWD/src" > "$SITE_DIR/chanlun-src.pth"
```

> ⚠️ **跳过的包**：`openctp-ctp` 没有 macOS wheels（仅 Linux/Windows），是期货 CTP 接口，做数字货币回测不需要。
> ⚠️ **pyproject.toml 里的 `ta-lib` 源指向 Windows .whl**：macOS 上需用 `brew install ta-lib` + `pip install TA-Lib` 从源码编译。

### 3. 同步 ETH/USDT 历史行情数据到本地数据库

**回测依赖本地数据库中的行情数据，必须先同步数据。**

可以用项目自带的同步脚本，也可以写一个精简的脚本只下载 ETH/USDT。

#### 方式一：使用自带脚本（下载全部币种，耗时较长）

```bash
cd /Users/gavfu/github/gavfu/chanlun-pro
python script/crontab/reboot_sync_currency_klines.py
```

#### 方式二：只下载 ETH/USDT（推荐）

创建一个脚本文件 `my_sync_eth.py`：

```python
from chanlun.exchange.exchange_binance import ExchangeBinance
from chanlun.exchange.exchange_db import ExchangeDB
import traceback

"""
只同步 ETH/USDT 行情到数据库
"""
exchange = ExchangeDB("currency")
line_exchange = ExchangeBinance()

codes = ["ETH/USDT"]
sync_frequencys = ["d", "30m", "5m"]  # 按需选择回测所用周期

for code in codes:
    for f in sync_frequencys:
        while True:
            try:
                last_dt = exchange.query_last_datetime(code, f)
                if last_dt is None:
                    klines = line_exchange.klines(
                        code, f, end_date="2024-01-01 00:00:00",
                        args={"use_online": True},
                    )
                    if len(klines) == 0:
                        klines = line_exchange.klines(
                            code, f, start_date="2024-01-01 00:00:00",
                            args={"use_online": True},
                        )
                else:
                    klines = line_exchange.klines(
                        code, f, start_date=last_dt,
                        args={"use_online": True},
                    )
                print(f"Synced {code} {f}: {len(klines)} bars")
                exchange.insert_klines(code, f, klines)
                if len(klines) <= 1:
                    break
            except Exception:
                print(f"Error syncing {code} {f}")
                traceback.print_exc()
                break
print("Done!")
```

运行：

```bash
cd /Users/gavfu/github/gavfu/chanlun-pro
python my_sync_eth.py
```

> ⚠️ 如果在中国大陆，需要确保代理已开启且 `config.py` 中 `PROXY_HOST/PORT` 配置正确。
> 同步脚本需要反复执行（或多等一会），直到覆盖 2025 年全年数据。

---

## 运行回测

### 方式一：直接写 Python 脚本运行（推荐）

创建 `my_backtest_eth.py`：

```python
from chanlun.backtesting import backtest
from chanlun.strategy.strategy_demo import StrategyDemo
from chanlun.cl_utils import query_cl_chart_config

# 缠论配置（使用数字货币市场的默认配置）
cl_config = query_cl_chart_config("currency", "ETH/USDT")

# 回测配置
bt_config = {
    # 策略结果保存的文件
    "save_file": "./data/bk/currency_eth_demo.pkl",
    # 设置策略对象
    "strategy": StrategyDemo(),
    # 回测模式：signal 信号模式，固定金额开仓；trade 交易模式，按照实际金额开仓
    "mode": "trade",
    # 市场配置
    "market": "currency",
    # 基准代码，用于获取回测的时间列表
    "base_code": "ETH/USDT",
    # 回测的标的代码
    "codes": ["ETH/USDT"],
    # 回测的周期（策略中通过 market_data.frequencys[0] 获取）
    "frequencys": ["30m"],
    # 回测开始时间
    "start_datetime": "2025-01-01 00:00:00",
    # 回测结束时间
    "end_datetime": "2025-12-31 23:59:00",
    # 初始账户资金
    "init_balance": 100000,
    # 交易手续费率（Binance taker 0.04%，maker 0.02%，这里取 0.06% 含滑点）
    "fee_rate": 0.0006,
    # 最大持仓数量（分仓）
    "max_pos": 1,
    # 缠论计算配置
    "cl_config": cl_config,
}

BT = backtest.BackTest(bt_config)

# 运行回测
BT.run()

# 保存结果
BT.save()

# 输出回测摘要
BT.info()
BT.result()

print("Done")
```

运行命令：

```bash
cd /Users/gavfu/github/gavfu/chanlun-pro
python my_backtest_eth.py
```

### 方式二：在 JupyterLab 中运行（方便可视化查看图表）

```bash
cd /Users/gavfu/github/gavfu/chanlun-pro
jupyter lab
```

打开 `notebook/回测_数字货币策略.ipynb`，将其中的配置修改为：
- `market`: `"currency"`
- `base_code`: `"ETH/USDT"`
- `codes`: `["ETH/USDT"]`
- `frequencys`: `["30m"]`（或 `["30m", "5m"]` 做多级别分析）
- `start_datetime`: `"2025-01-01 00:00:00"`
- `end_datetime`: `"2025-12-31 23:59:00"`

然后逐单元格执行即可。Notebook 中可以调用 `BT.show_charts()` 和 `BT.backtest_charts()` 查看带缠论标注的图表和资金曲线。

---

## 查看回测结果

### 命令行输出

脚本运行完后会打印类似下面的表格，包含每种买卖点的胜率、盈亏比等：

```
+------------+------+------+--------+--------+--------+--------+----------+----------+----------+--------+
|   买卖点   | 成功 | 失败 |  胜率  |  盈利  |  亏损  | 净利润 | 回吐比例 | 平均盈利 | 平均亏损 | 盈亏比 |
+------------+------+------+--------+--------+--------+--------+----------+----------+----------+--------+
| 一类买点   |  ... | ...  |  ...%  |  ...   |  ...   |  ...   |   ...    |   ...    |   ...    |  ...   |
| ...        |      |      |        |        |        |        |          |          |          |        |
|   汇总     |  ... | ...  |  ...%  |  ...   |  ...   |  ...   |   ...    |   ...    |   ...    |  ...   |
+------------+------+------+--------+--------+--------+--------+----------+----------+----------+--------+
```

### 加载已保存的结果

```python
from chanlun.backtesting import backtest

BT = backtest.BackTest()
BT.load("./data/bk/currency_eth_demo.pkl")
BT.result()

# 查看历史持仓明细
pos_df = BT.positions()
print(pos_df)

# 查看图表（需在 JupyterLab 中）
BT.show_charts("ETH/USDT", "30m")
BT.backtest_charts()
```

---

## 关键参数说明

| 参数 | 说明 |
|------|------|
| `mode` | `"signal"` 信号模式（不考虑资金，统计胜率盈亏比）；`"trade"` 交易模式（模拟真实资金和仓位） |
| `frequencys` | 回测周期列表，如 `["30m"]` 或 `["30m", "5m"]`（多级别），策略中按下标取数据 |
| `cl_config` | 缠论计算参数（笔类型、线段划分、中枢类型等），可通过 `query_cl_chart_config` 获取默认值 |
| `max_pos` | 最大持仓分仓数，单标的设为 1 即可 |
| `fee_rate` | 手续费率，Binance USDT 永续合约建议设 0.0004~0.0006 |

## 注意事项

1. **必须先同步数据**：回测引擎从本地数据库 (`ExchangeDB`) 读取 K 线，不会实时从 Binance 拉取。
2. **PyArmor 授权**：核心 `cl.py` 被加密，需要有效的授权许可文件才能运行。首次使用可联系作者获取 20 天试用授权。
3. **代理**：在中国大陆访问 Binance API 需要代理。
4. **自定义策略**：建议复制 `strategy_demo.py` 为 `my_strategy_xxx.py`（`my_` 前缀会被 git 忽略），避免代码更新冲突。
