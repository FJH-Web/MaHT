# StatAgent

StatAgent 是一个 **Hypothesis-Centric Statistical Inference Framework**。它围绕可检验的统计假设组织证据、方法检索、算法执行、后验更新和报告，而不是让 LLM 生成统计代码。

当前网页、方法名称、统计结果字段、可视化标题与坐标轴、逐项解释和论文 HTML 报告均采用简体中文。Matplotlib 会依次探测 Microsoft YaHei、SimHei、Noto Sans CJK SC 等中文字体，并关闭 Unicode 负号冲突，避免中文标题显示为方框或乱码。

当前版本实现了有限、可审计的 Evidence Loop。LLM 不只是入口解析器，而承担四类受约束的语义决策：需求解析、候选方法裁决、结果/图形解释和论文段落组织；统计量与图片始终由本地注册算法生成。

```text
User text
  -> DeepSeek Parser boundary
  -> HypothesisState / StatisticalObject only
  -> Hypothesis Graph
  -> Method Card retrieval -> DeepSeek constrained matching
  -> Registered local statistical API
  -> Method-specific local figures
  -> DeepSeek result/figure interpretation
  -> Statistical Memory
  -> Decision policy
  -> Active Hypothesis Scheduler
  -> DeepSeek paper sections -> deterministic HTML template
```

## 设计保证

- DeepSeek 负责结构化理解、候选方法匹配、数据列角色建议、结果/图形解释和论文叙事，但不生成或执行统计代码。
- Agent 内部只交换 `StatisticalObject`，业务逻辑不依赖自由文本或 `if task == ...` 路由。
- 执行前必须检索 Method Card；Dispatcher 只允许运行注册表中的本地函数。
- Hypothesis Graph 强制无环依赖并支持更新、拓扑序和 Mermaid 可视化。
- Statistical Memory 保存推断谱系和版本历史，不保存聊天记录。
- AHS 的默认效用为 `Expected Information Gain - Cost`，接口可替换为 MARL。
- 每个核心对象均有 Theory-to-Code 注释和对应文档。
- 每张 Method Card 声明方法专属可视化配方；典型推断会自动生成 2–3 张结果图并写入 HTML。

## 中文探索性图谱

上传数据后，Evidence Loop 会把数据集级探索性图形写入 Statistical Memory，并与方法专属图形一起交给大模型逐图解释：

- 变量类型构成图
- 数值变量分布与核密度图
- 数值变量箱线图
- 数值变量提琴图
- 正态 Q–Q 诊断图
- 经验累积分布函数图
- 相关系数热图
- 数值变量散点矩阵
- 各变量缺失比例图
- 缺失模式矩阵
- 分类变量频数图
- 潜在离群值比例图

大模型先对每个主要统计结果字段和每张图分别解释，再基于全部假设、统计量、效应量、诊断和图形证据，统一生成中文论文的摘要、结果、综合讨论和总体总结。

## 打开网页

安装依赖后直接双击 `run_webapp.bat`，或在项目目录运行：

```powershell
E:\anconda\python.exe -m streamlit run statagent\app\webapp.py
```

浏览器打开 `http://localhost:8501`。上传 CSV、输入 DeepSeek Key 和研究需求后，页面会依次展示假设图、候选方法及选择理由、数据映射、统计结果、方法专属图形与 LLM 解读，并提供自包含 HTML 报告下载。

## 安装

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

当前算法库只注册假设检验方法，不再安装或暴露机器学习、回归、降维及纯描述统计算法。

## DeepSeek 配置

复制 `.env.example` 的变量到你的安全运行环境。项目不会自动读取或提交 `.env`，也不会在日志中输出密钥。

```powershell
$env:DEEPSEEK_API_KEY="your-rotated-key"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:DEEPSEEK_MODEL="deepseek-chat"
```

网页侧也可以直接填写 Base URL 和模型名；如果使用兼容网关，请确认 Base URL 指向包含 `/chat/completions` 的 API 根地址。

不要把真实密钥写进源码、测试、Method Card 或实验配置。任何曾经出现在聊天或日志中的密钥都应立即轮换。

## 最小用法

生产入口使用 `build_default_app()`，其 Parser 默认为 DeepSeek。数据必须由调用方以结构化形式提供；LLM 不接触数据执行代码。

```python
from statagent.app import ExecutionInput, StatAgentConfig, build_default_app

app = build_default_app(StatAgentConfig(memory_path="artifacts/memory.json"))

def provide_data(message):
    # 实际系统中按 hypothesis.id 从受控数据层取得预定义输入。
    return ExecutionInput(
        data={"group_a": [1.2, 1.8, 2.0], "group_b": [3.1, 3.5, 3.8]},
        parameters={"equal_var": False},
    )

report = app.run("检验治疗组与对照组的均值是否不同", provide_data)
print(report)
```

不调用外部模型的完整示例见 [`examples/local_evidence_loop.py`](examples/local_evidence_loop.py)。

## 假设检验方法库

| 分支 | 方法 |
|---|---|
| 分布假设检验 | Shapiro–Wilk、Energy 多元正态性、Henze–Zirkler、Royston、Kolmogorov–Smirnov、卡方拟合优度、二项精确检验 |
| 方差齐性检验 | 两样本方差 F、Bartlett、Levene |
| 参数检验 | 单样本/独立样本/配对 t 检验、单因素方差分析、单样本/双样本 Z 检验、Holm 成对 t 比较、Dunnett、Tukey HSD |
| 非参数检验 | Mann–Whitney U、单样本/配对 Wilcoxon、Kruskal–Wallis、Moses、Wald–Wolfowitz、Friedman |
| 列联表检验 | 卡方独立性、Fisher 精确、Cochran–Mantel–Haenszel |
| 相关与随机性 | 双变量相关检验、多变量两两相关检验、随机性游程检验 |
| 生存假设检验 | Log-rank 检验 |

每个 Method Card 都包含 Description、Tags、Input、Output、API、Reference、Assumption、Example 和版本号。新增方法只需注册新的 Card 与预部署函数，无需修改路由器。

## 目录

```text
statagent/
├── agents/                  # 六类 SDP Agent
├── protocol/                # HypothesisState 与 StatisticalObject
├── hypothesis/              # G_H
├── retrieval/               # HSR、embedding 与 vector store 接口
├── statistical_library/     # 本地预部署算法与 Method Card catalog
├── api/                     # registry 与 dispatcher
├── memory/                  # Statistical Memory
├── scheduler/               # AHS 接口与默认实现
├── report/                  # 报告扩展点
├── datasets/                # 数据适配扩展点
├── experiments/             # 可复现实验扩展点
└── app/                     # 有限 Evidence Loop 组装
```

理论映射见 [`docs/theory-to-code.md`](docs/theory-to-code.md)，模块边界见 [`docs/architecture.md`](docs/architecture.md)。

## 测试

```powershell
python -m pytest
```

测试覆盖协议序列化、图依赖、方法卡检索、注册式执行、统计结果、JSON Memory 版本历史、AHS 效用、中文探索性图谱和离线端到端证据循环。测试不会调用 DeepSeek，也不会联网。

## v0.1 明确不包含

自动生成统计代码、AutoGPT 式无限循环、Browser Agent、Tool Calling Agent、自动联网、Prompt 优化、多 LLM 协商和 MCP 均不在本版本范围内。
