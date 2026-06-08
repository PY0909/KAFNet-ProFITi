# 开题稳健版执行计划

> **给 agentic workers：** 必须使用 `subagent-driven-development`（推荐）或 `executing-plans` 按任务逐项执行本计划。步骤使用复选框（`- [ ]`）追踪状态。

**目标：** 以“开题稳健版”为主线，完成开题材料、真实文献核验、仓库分类治理和最小真实实验矩阵的可执行落地。

**架构：** 工作分为四条工作流：文档开题、文献核验、仓库治理、实验补齐。前两条先支撑开题通过，后两条保证 GitHub 仓库可复现并为后续基线、消融、多 seed 实验铺路。

**技术栈：** Markdown、CSV、Git/GitHub、Python、PyTorch、pytest、pandas、NumPy、scikit-learn、现有 `code/kaf_profiti` 实验框架。

---

## 范围检查

本计划覆盖多个工作流，但它们服务同一个交付目标：形成可开题、可复现、可继续补实验的研究仓库。执行时必须按任务顺序推进，并在每个任务结束提交一次。代码类任务从任务 6 开始；任务 1-5 先完成开题与仓库基础资产。

## 文件结构

本计划会创建或修改以下文件：

- `README.md`：仓库入口，说明论文方向、目录分类、数据策略和复现实验入口。
- `CHANGELOG.md`：记录每次有意义改动。
- `docs/opening-report/开题报告-草稿.md`：开题报告 Markdown 草稿。
- `docs/opening-report/开题报告-写作说明.md`：说明如何从草稿转成学校模板。
- `docs/references/reference-search-protocol.md`：文献检索与核验规则。
- `docs/references/references_verified.csv`：真实文献核验表。
- `docs/experiment-roadmap.md`：真实基线、消融、多 seed、鲁棒性实验路线。
- `data-manifests/metropt3.md`：MetroPT-3 数据来源、文件、划分和上传策略。
- `data-manifests/cmapss.md`：C-MAPSS 数据来源、文件、划分和上传策略。
- `data-manifests/tep.md`：TEP 扩展数据来源、边界和上传策略。
- `code/tests/test_model_variants.py`：新增模型变体测试。
- `code/tests/test_risk_metrics.py`：风险评分与指标测试。
- `code/kaf_profiti/models/gaussian_head.py`：KAFNet-Gaussian 概率头。
- `code/kaf_profiti/models/kaf_gaussian.py`：KAFNet 编码器 + Gaussian Head 变体。
- `code/kaf_profiti/experiments/registry.py`：启用消融变体注册。
- `code/kaf_profiti/industrial/risk.py`：阈值化风险评分。
- `code/kaf_profiti/experiments/metrics.py`：风险指标入口对接。
- `code/run_experiment.py`：保存风险阈值配置和多模型运行兼容性。
- `code/build_tables.py` 与 `code/kaf_profiti/experiments/tables.py`：汇总 `mean ± std` 和统计表。

## 任务 1：建立仓库入口与数据清单

**文件：**
- 创建：`README.md`
- 创建：`data-manifests/metropt3.md`
- 创建：`data-manifests/cmapss.md`
- 创建：`data-manifests/tep.md`
- 修改：`CHANGELOG.md`

- [ ] **步骤 1：写入仓库入口 README**

在 `README.md` 写入以下内容：

```markdown
# KAFNet-ProFITi

本仓库用于论文《面向工业设备异步多传感器的概率状态预测与风险评估方法研究》的开题、实验与复现管理。

## 当前阶段

- 研究路线：开题稳健版。
- 主数据集：MetroPT-3、C-MAPSS FD001。
- 扩展数据集：C-MAPSS FD002-FD004、TEP。
- 当前证据边界：已有 KAFNet-ProFITi 原型和单 seed 冒烟实验，尚未完成真实基线、消融、多 seed 和鲁棒性实验。

## 仓库目录

```text
docs/
  opening-report/          开题报告草稿与写作说明
  references/              文献检索协议与核验表
  superpowers/             设计文档与执行计划
code/
  kaf_profiti/             核心模型、工业数据协议与实验框架
  tests/                   单元测试和实验框架测试
data-manifests/            数据来源、划分、下载与上传策略
```

## 数据策略

原始数据集、模型检查点、预测数组和大型生成文件不进入普通 Git。数据来源、预处理方式、划分规则和随机种子记录在 `data-manifests/` 与实验配置中。

## 规范文档

- 仓库治理：`docs/repository-governance.md`
- 开题可行性设计：`docs/superpowers/specs/2026-06-08-opening-feasibility-design.md`
- 执行计划：`docs/superpowers/plans/2026-06-08-opening-stable-execution.md`
```

- [ ] **步骤 2：写入 MetroPT-3 数据清单**

在 `data-manifests/metropt3.md` 写入：

```markdown
# MetroPT-3 数据清单

## 数据来源

- 数据集：MetroPT-3 Air Compressor Dataset
- 来源：UCI Machine Learning Repository
- 链接：https://archive.ics.uci.edu/dataset/791/metropt%2B3%2Bdataset

## 本课题用途

MetroPT-3 作为主工业设备数据集，用于空气压缩机多传感器未来状态概率预测，以及基于故障报告窗口的风险评估。

## 本地文件

```text
dataset/metropt+3+dataset/MetroPT3(AirCompressor).csv
dataset/metropt+3+dataset/Data Description_Metro.pdf
```

## Git 策略

原始 CSV 和 PDF 不进入普通 Git。仓库只跟踪读取代码、划分清单、配置、指标表和论文图表。

## 当前协议

- Train：2020-02-01 至第一月前 80%。
- Valid：第一月后 20%。
- Test：2020-03-01 之后数据。
- 主缺失率：0.3。
- 主随机种子：2026、2027、2028。
```

- [ ] **步骤 3：写入 C-MAPSS 数据清单**

在 `data-manifests/cmapss.md` 写入：

```markdown
# C-MAPSS 数据清单

## 数据来源

- 数据集：NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set
- 来源：NASA Prognostics Center of Excellence
- 链接：https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

## 本课题用途

C-MAPSS FD001 作为主退化验证数据集，用于多传感器状态预测和基于 RUL 阈值的风险验证。FD002-FD004 在 FD001 稳定后扩展。

## 本地文件

```text
dataset/CMAPSSData/train_FD001.txt
dataset/CMAPSSData/test_FD001.txt
dataset/CMAPSSData/RUL_FD001.txt
```

## Git 策略

原始 TXT 和 PDF 不进入普通 Git。仓库只跟踪读取代码、划分清单、配置、指标表和论文图表。

## 当前协议

- Train/Valid：官方 train 文件按 engine id 进行 80/20 划分。
- Test：官方 test 文件结合 RUL 文件。
- 风险标签：`RUL <= threshold`。
- 主随机种子：2026、2027、2028。
```

- [ ] **步骤 4：写入 TEP 扩展数据清单**

在 `data-manifests/tep.md` 写入：

```markdown
# TEP 数据清单

## 数据来源

- 数据集：Tennessee Eastman Process Simulation Dataset
- 来源：Harvard Dataverse
- 链接：https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1

## 本课题用途

TEP 作为过程工业扩展验证数据集。开题阶段不将完整 faulty-run 结果作为主承诺。

## 边界

当前默认协议未完整展开 faulty-run 主实验。后续若解决故障协议、内存和运行时间问题，可作为第五章扩展验证。

## Git 策略

原始 RData 和大型中间文件不进入普通 Git。仓库只记录数据来源、协议、配置和小体积结果。
```

- [ ] **步骤 5：记录变更**

在 `CHANGELOG.md` 的 `2026-06-08` 下追加：

```markdown
- 新增 README 与三个数据清单，明确数据来源、实验用途和 Git 上传边界。
```

- [ ] **步骤 6：验收**

运行：

```bash
git status --short
```

预期：显示 `README.md`、`data-manifests/*.md`、`CHANGELOG.md` 为新增或修改，不出现 `dataset/`、`.pt`、`.npy` 被跟踪。

- [ ] **步骤 7：提交**

```bash
git add README.md data-manifests/metropt3.md data-manifests/cmapss.md data-manifests/tep.md CHANGELOG.md
git commit -m "repo: 添加仓库入口和数据清单"
git push
```

## 任务 2：建立文献核验协议与空表

**文件：**
- 创建：`docs/references/reference-search-protocol.md`
- 创建：`docs/references/references_verified.csv`
- 修改：`CHANGELOG.md`

- [ ] **步骤 1：写入文献检索协议**

在 `docs/references/reference-search-protocol.md` 写入：

```markdown
# 文献检索与核验协议

## 数量要求

- 总数：40-45 篇核验文献。
- 英文：至少 22 篇。
- 中文：至少 12 篇。
- 年份：优先 2022-2026。
- 每篇必须有真实链接。

## 主题分组

1. 工业预测性维护与设备状态预测。
2. 异步、不规则、缺失多元时间序列建模。
3. 概率预测、不确定性量化与校准。
4. RUL、健康状态预测与风险评估。
5. KAFNet、ProFITi、GraFITi、GRU-D、mTAN、ODE-RNN、TCN、PatchTST 等相关模型。

## 核验规则

每篇文献必须记录：

```text
language,year,title,authors,venue,topic,url,used_in,verified_date
```

英文文献优先使用 DOI、出版社页、会议页或 arXiv。中文文献优先使用 CNKI、万方、维普、期刊官网或 DOI。无法确认来源真实性的条目不进入最终开题报告。

## 使用位置

- `1.2 国内外研究现状`：用于综述。
- `1.3 述评`：用于归纳不足。
- `5.1 研究方法`：用于说明实验与模型依据。
- `主要参考文献`：只列入已核验条目。
```

- [ ] **步骤 2：写入 CSV 表头**

在 `docs/references/references_verified.csv` 写入：

```csv
language,year,title,authors,venue,topic,url,used_in,verified_date
```

- [ ] **步骤 3：记录变更**

在 `CHANGELOG.md` 的 `2026-06-08` 下追加：

```markdown
- 新增文献检索协议和核验表，后续参考文献必须先核验再写入开题报告。
```

- [ ] **步骤 4：验收**

运行：

```bash
python - <<'PY'
import csv
from pathlib import Path
path = Path("docs/references/references_verified.csv")
rows = list(csv.reader(path.open()))
assert rows[0] == ["language","year","title","authors","venue","topic","url","used_in","verified_date"]
print("references csv header ok")
PY
```

预期输出：

```text
references csv header ok
```

- [ ] **步骤 5：提交**

```bash
git add docs/references/reference-search-protocol.md docs/references/references_verified.csv CHANGELOG.md
git commit -m "refs: 添加文献核验协议"
git push
```

## 任务 3：生成开题报告 Markdown 草稿

**文件：**
- 创建：`docs/opening-report/开题报告-草稿.md`
- 创建：`docs/opening-report/开题报告-写作说明.md`
- 修改：`CHANGELOG.md`

- [ ] **步骤 1：写入开题报告草稿骨架**

在 `docs/opening-report/开题报告-草稿.md` 写入：

```markdown
# 面向工业设备异步多传感器的概率状态预测与风险评估方法研究

## 1. 选题背景、国内外研究现状及述评

### 1.1 选题背景

工业设备在运行过程中会持续产生压力、温度、电流、流量、振动等多源传感器数据。这些数据能够反映设备运行状态、退化趋势和潜在故障风险，是预测性维护和智能运维的重要基础。与理想规则采样时间序列不同，真实工业场景中的传感器经常受到采样频率不一致、通信延迟、传感器离线、工况切换和异常中断等因素影响，形成异步、缺失、多变量耦合的观测序列。若直接使用固定插值或点预测模型，容易引入伪观测并忽略未来状态的不确定性。

本研究面向工业设备异步多传感器数据，拟构建能够同时进行未来状态概率预测和风险评估的方法。研究重点不是单纯判断某一时刻是否故障，而是从历史异步观测中学习规整表示，预测未来多传感器状态的联合分布，并进一步将预测分布转化为可解释的风险分数和预警指标。

### 1.2 国内外研究现状

#### 1.2.1 工业预测性维护与设备状态预测

本节写入预测性维护、RUL、工业状态监测相关文献。

#### 1.2.2 异步缺失多元时间序列建模

本节写入 GRU-D、ODE-RNN、mTAN、GraFITi、ProFITi、不规则时间序列和缺失建模相关文献。

#### 1.2.3 概率预测、不确定性量化与风险评估

本节写入 probabilistic forecasting、normalizing flow、CRPS、calibration、risk assessment 相关文献。

### 1.3 研究述评

现有研究仍存在三方面不足：第一，许多方法依赖规则采样或固定插值，难以充分适应工业传感器异步缺失观测；第二，点预测模型难以表达未来状态的不确定性和多传感器联合相关性；第三，工业运维需要可解释且校准的风险输出，而现有预测模型往往只报告误差指标，缺少从预测分布到风险预警的统一评估流程。

## 2. 研究内容和论文结构

本文拟围绕工业设备异步多传感器的概率状态预测与风险评估开展研究，主要研究内容包括：

1. 构建工业异步多传感器数据协议与缺失模拟机制。
2. 构建 KAFNet-ProFITi 联合概率状态预测模型。
3. 构建基于预测分布样本的风险评分与鲁棒性评估方法。

拟定论文结构如下：

- 第 1 章：绪论。
- 第 2 章：相关理论与技术基础。
- 第 3 章：工业异步多传感器数据协议与缺失构造。
- 第 4 章：KAFNet-ProFITi 联合概率状态预测方法。
- 第 5 章：风险评估、缺失鲁棒性与统计验证。
- 第 6 章：总结与展望。

## 3. 创新之处

1. 提出面向工业异步多传感器的工况感知规整表示方法。
2. 提出查询条件化的 KAFNet-ProFITi 联合概率状态预测模型。
3. 提出面向工业运维的基于分布样本的风险评分与鲁棒评估协议。

## 4. 可行性分析

当前已经完成 KAFNet-ProFITi 融合原型、MetroPT-3 与 C-MAPSS 数据适配、统一结果保存目录和单 seed 冒烟实验。已有结果证明原型框架可运行，但尚不能作为最终论文对比结论。后续将补齐真实基线、消融实验、多 seed 统计、缺失鲁棒性和风险评估。

## 5. 研究方法与手段

本研究拟采用文献调研法、模型构建法、受控实验分析法和可复现仓库管理方法。实验中所有模型使用统一划分、统一缺失 mask、train-only 归一化规则和固定随机种子，结果报告 `mean ± std`。

## 6. 研究重点与难点

研究重点包括异步缺失协议、联合概率预测、风险评分和鲁棒性评估。研究难点包括概率指标公平比较、风险标签构造、联合 flow 训练稳定性和多 seed 统计可信度。

## 主要参考文献

参考文献来自 `docs/references/references_verified.csv`，只使用已核验的真实论文链接。
```

- [ ] **步骤 2：写入写作说明**

在 `docs/opening-report/开题报告-写作说明.md` 写入：

```markdown
# 开题报告写作说明

## 来源模板

结构参考 `开题参考/开题报告.docx`，但内容改为工业异步传感器概率预测与风险评估。

## 写作规则

1. 已有实验只能写成原型和冒烟实验，不写成最终优于基线。
2. MetroPT-3 和 C-MAPSS 是主数据集，TEP 是扩展数据集。
3. 参考文献必须来自 `docs/references/references_verified.csv`。
4. 创新点保持三条，不增加未设计的新模块。
5. 第 4 章对应概率预测方法，第 5 章对应风险评估和鲁棒性验证。

## 转成学校模板

先完成 Markdown 草稿，再根据 `开题参考/开题报告.docx` 的栏目复制到 Word 模板。复制前检查：

- 标题与英文标题。
- 研究方向。
- 文献数量和年份。
- 进度计划。
- 指导教师意见区域不由本仓库填写。
```

- [ ] **步骤 3：记录变更**

在 `CHANGELOG.md` 的 `2026-06-08` 下追加：

```markdown
- 新增开题报告 Markdown 草稿和写作说明，明确已有原型与计划实验的表述边界。
```

- [ ] **步骤 4：验收**

运行：

```bash
rg -n "优于|显著|SOTA|已经完成完整对比" docs/opening-report/开题报告-草稿.md
```

预期：无输出。若出现上述词语，改为“计划对比”“原型可运行”“后续补齐”等保守表述。

- [ ] **步骤 5：提交**

```bash
git add docs/opening-report/开题报告-草稿.md docs/opening-report/开题报告-写作说明.md CHANGELOG.md
git commit -m "paper: 添加开题报告草稿"
git push
```

## 任务 4：建立实验路线文档

**文件：**
- 创建：`docs/experiment-roadmap.md`
- 修改：`CHANGELOG.md`

- [ ] **步骤 1：写入实验路线**

在 `docs/experiment-roadmap.md` 写入：

```markdown
# 实验补齐路线

## 当前边界

现有结果是 `kaf_profiti_joint` 在 MetroPT-3 和 C-MAPSS FD001 上的单 seed 冒烟实验。它证明原型可运行，但不支撑论文级对比结论。

## 第一阶段：消融优先

目标：证明最终模型每个核心模块必要。

模型：

- `kafnet`
- `kafnet_gaussian`
- `kaf_profiti_marginal`
- `kaf_profiti_joint`
- `kaf_profiti_joint_no_context`

数据集：

- MetroPT-3
- C-MAPSS FD001

随机种子：

- 2026
- 2027
- 2028

指标：

- MAE
- RMSE
- NLL
- CRPS
- PICP
- ECE

## 第二阶段：真实基线

目标：证明最终模型不是只优于弱消融。

最小基线：

- `tcn_gaussian`
- `gru_d`
- `profiti`

可扩展基线：

- `patchtst_gaussian`
- `mtan`
- `grafiti`

## 第三阶段：缺失鲁棒性

缺失率：

- 0
- 0.1
- 0.3
- 0.5
- 0.7

核心模型：

- `tcn_gaussian`
- `gru_d`
- `profiti`
- `kaf_profiti_joint`

## 第四阶段：风险评估

MetroPT-3 使用故障报告窗口构造风险标签。C-MAPSS 使用 RUL 阈值构造风险标签。风险评分必须从阈值越界概率、状态标签或 RUL 阈值出发，不能使用无物理含义的样本绝对值公式作为最终风险结果。

## 汇总要求

- 每个模型保存 config。
- 每个模型保存 metrics JSON。
- 表格由 metrics 自动构建。
- 主表使用 `mean ± std`。
- 大型 checkpoints 和 predictions 不进入普通 Git。
```

- [ ] **步骤 2：记录变更**

在 `CHANGELOG.md` 的 `2026-06-08` 下追加：

```markdown
- 新增实验补齐路线，按消融、真实基线、缺失鲁棒性和风险评估分阶段推进。
```

- [ ] **步骤 3：验收**

运行：

```bash
rg -n "kafnet_gaussian|gru_d|missing|mean ± std|风险" docs/experiment-roadmap.md
```

预期：输出命中实验路线中的对应条目。

- [ ] **步骤 4：提交**

```bash
git add docs/experiment-roadmap.md CHANGELOG.md
git commit -m "docs: 添加实验补齐路线"
git push
```

## 任务 5：核验首批真实文献

**文件：**
- 修改：`docs/references/references_verified.csv`
- 修改：`CHANGELOG.md`

- [ ] **步骤 1：检索英文文献**

使用官方来源或可信索引检索 22 篇英文文献。每篇写入：

```csv
English,2024,论文标题,作者,期刊或会议,主题,真实链接,1.2.x,2026-06-08
```

优先主题：

- irregular time series
- probabilistic forecasting
- predictive maintenance
- RUL prediction
- calibration
- normalizing flows for time series
- missing data imputation and forecasting

- [ ] **步骤 2：检索中文文献**

使用 CNKI、万方、维普、期刊官网或 DOI 检索 12 篇中文文献。每篇写入：

```csv
Chinese,2024,论文标题,作者,期刊,主题,真实链接,1.2.x,2026-06-08
```

优先主题：

- 预测性维护
- 剩余寿命预测
- 工业设备状态监测
- 多传感器时间序列
- 不确定性预测
- 故障预警

- [ ] **步骤 3：检查数量和年份**

运行：

```bash
python - <<'PY'
import csv
from collections import Counter
from pathlib import Path
rows = list(csv.DictReader(Path("docs/references/references_verified.csv").open()))
counts = Counter(row["language"] for row in rows)
years = [int(row["year"]) for row in rows]
assert len(rows) >= 40, len(rows)
assert counts["English"] >= 20, counts
assert counts["Chinese"] >= 10, counts
assert all(2022 <= year <= 2026 for year in years), years
assert all(row["url"].startswith(("http://", "https://")) for row in rows)
print("references verified:", len(rows), dict(counts))
PY
```

预期输出形如：

```text
references verified: 40 {'English': 22, 'Chinese': 12}
```

- [ ] **步骤 4：记录变更**

在 `CHANGELOG.md` 的执行日期下追加：

```markdown
- 完成首批真实参考文献核验，满足英文 20 篇以上、中文 10 篇以上和 2022-2026 年份要求。
```

- [ ] **步骤 5：提交**

```bash
git add docs/references/references_verified.csv CHANGELOG.md
git commit -m "refs: 核验首批真实参考文献"
git push
```

## 任务 6：添加 KAFNet-Gaussian 消融模型

**文件：**
- 创建：`code/tests/test_model_variants.py`
- 创建：`code/kaf_profiti/models/gaussian_head.py`
- 创建：`code/kaf_profiti/models/kaf_gaussian.py`
- 修改：`code/kaf_profiti/models/__init__.py`
- 修改：`code/kaf_profiti/experiments/registry.py`
- 修改：`CHANGELOG.md`

- [ ] **步骤 1：写失败测试**

在 `code/tests/test_model_variants.py` 写入：

```python
import torch

from kaf_profiti.industrial.batch import IndustrialBatch
from kaf_profiti.experiments.registry import create_model, get_model_spec


def make_batch(batch_size=2, history_len=8, pred_len=3, num_sensors=4, context_dim=3):
    generator = torch.Generator().manual_seed(17)
    x_obs = torch.randn(batch_size, history_len, num_sensors, generator=generator)
    t_obs = torch.arange(history_len, dtype=torch.float32).repeat(batch_size, 1)
    m_obs = (torch.rand(batch_size, history_len, num_sensors, generator=generator) > 0.2).float()
    x_obs = x_obs * m_obs
    t_q = torch.arange(history_len, history_len + pred_len, dtype=torch.float32).repeat(batch_size, 1)
    y_q = torch.randn(batch_size, pred_len, num_sensors, generator=generator)
    m_q = torch.ones(batch_size, pred_len, num_sensors)
    context = torch.randn(batch_size, context_dim, generator=generator)
    return IndustrialBatch(
        X_obs=x_obs,
        T_obs=t_obs,
        M_obs=m_obs,
        T_q=t_q,
        Y_q=y_q,
        M_q=m_q,
        context=context,
        y_flat=y_q.reshape(batch_size, pred_len * num_sensors),
        mq_flat=m_q.reshape(batch_size, pred_len * num_sensors),
        query_channel_ids=torch.arange(num_sensors).repeat(pred_len),
        rul=torch.tensor([50.0, 60.0]),
        unit_id=torch.tensor([1, 2]),
    )


def test_kafnet_gaussian_registry_enabled_and_loss_backward():
    spec = get_model_spec("kafnet_gaussian")
    assert spec.status == "enabled"
    model = create_model(
        "kafnet_gaussian",
        num_sensors=4,
        context_dim=3,
        device="cpu",
        hidden_dim=16,
        te_dim=5,
        kernel_count=3,
        n_layers=1,
        n_heads=2,
        flow_layers=1,
        preconv_dim=4,
        lambda_point=0.1,
    )
    batch = make_batch()
    loss = model.loss(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )


def test_kafnet_gaussian_samples_shape():
    model = create_model(
        "kafnet_gaussian",
        num_sensors=4,
        context_dim=3,
        device="cpu",
        hidden_dim=16,
        te_dim=5,
        kernel_count=3,
        n_layers=1,
        n_heads=2,
        flow_layers=1,
        preconv_dim=4,
        lambda_point=0.1,
    )
    batch = make_batch()
    samples = model.sample(batch, nsamples=5)
    assert samples.shape == (2, 5, 3, 4)
    assert torch.isfinite(samples).all()
```

- [ ] **步骤 2：运行测试确认失败**

运行：

```bash
PYTHONPATH=code pytest code/tests/test_model_variants.py -q
```

预期：失败，原因是 `kafnet_gaussian` 未注册或模块不存在。

- [ ] **步骤 3：实现 Gaussian Head**

在 `code/kaf_profiti/models/gaussian_head.py` 写入：

```python
import math

import torch
from torch import Tensor, nn


class GaussianHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mean = nn.Linear(hidden_dim, 1)
        self.log_var = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_var.weight)
        nn.init.constant_(self.log_var.bias, -1.0)

    def distribution_params(self, hidden_states: Tensor, mask: Tensor):
        mean = self.mean(hidden_states).squeeze(-1) * mask
        log_var = self.log_var(hidden_states).squeeze(-1).clamp(-6.0, 6.0)
        return mean, log_var

    def nll(self, y: Tensor, hidden_states: Tensor, mask: Tensor) -> Tensor:
        mean, log_var = self.distribution_params(hidden_states, mask)
        var = torch.exp(log_var)
        nll = 0.5 * (((y - mean).pow(2) / var) + log_var + math.log(2.0 * math.pi))
        return (nll * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)

    def sample(self, hidden_states: Tensor, mask: Tensor, nsamples: int = 100) -> Tensor:
        mean, log_var = self.distribution_params(hidden_states, mask)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn(mean.shape[0], nsamples, mean.shape[1], device=mean.device)
        return (mean[:, None, :] + eps * std[:, None, :]) * mask[:, None, :]

    @staticmethod
    def masked_mse(y: Tensor, mean: Tensor, mask: Tensor) -> Tensor:
        return (((mean - y) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)
```

- [ ] **步骤 4：实现 KAFGaussian 模型**

在 `code/kaf_profiti/models/kaf_gaussian.py` 写入：

```python
from dataclasses import asdict, dataclass
from typing import Dict

import torch
from torch import Tensor, nn

from .gaussian_head import GaussianHead
from .kafnet_encoder import KAFNetEncoder
from .query_condition_adapter import QueryConditionAdapter


@dataclass
class KAFGaussianConfig:
    num_sensors: int
    context_dim: int
    hidden_dim: int = 32
    te_dim: int = 5
    kernel_count: int = 4
    n_layers: int = 2
    n_heads: int = 2
    preconv_dim: int = 8
    lambda_point: float = 0.1
    device: str = "cpu"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class KAFGaussian(nn.Module):
    def __init__(self, config: KAFGaussianConfig):
        super().__init__()
        self.config = config
        self.encoder = KAFNetEncoder(
            num_sensors=config.num_sensors,
            hidden_dim=config.hidden_dim,
            kernel_count=config.kernel_count,
            time_dim=config.te_dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            preconv_dim=config.preconv_dim,
            context_dim=config.context_dim,
        )
        self.adapter = QueryConditionAdapter(
            num_sensors=config.num_sensors,
            hidden_dim=config.hidden_dim,
            time_dim=config.te_dim,
            context_dim=config.context_dim,
        )
        self.head = GaussianHead(config.hidden_dim)
        self._hidden_states = None
        self.to(torch.device(config.device))

    @property
    def hidden_states(self) -> Tensor:
        if self._hidden_states is None:
            raise RuntimeError("Must call distribution first")
        return self._hidden_states

    def distribution(self, batch) -> Tensor:
        z_var = self.encoder(batch.X_obs, batch.T_obs, batch.M_obs, batch.context)
        self._hidden_states = self.adapter(
            z_var,
            batch.T_q,
            batch.query_channel_ids,
            batch.context,
        )
        return self._hidden_states

    def loss(self, batch) -> Tensor:
        hidden = self.distribution(batch)
        nll = self.head.nll(batch.y_flat, hidden, batch.mq_flat).mean()
        if self.config.lambda_point <= 0:
            return nll
        mean, _ = self.head.distribution_params(hidden, batch.mq_flat)
        return nll + self.config.lambda_point * self.head.masked_mse(batch.y_flat, mean, batch.mq_flat)

    def sample(self, batch, nsamples: int = 100) -> Tensor:
        hidden = self.distribution(batch)
        flat = self.head.sample(hidden, batch.mq_flat, nsamples=nsamples)
        batch_size = batch.X_obs.shape[0]
        pred_len = batch.T_q.shape[1]
        return flat.reshape(batch_size, nsamples, pred_len, self.config.num_sensors)
```

- [ ] **步骤 5：更新注册表**

在 `code/kaf_profiti/experiments/registry.py`：

```python
from kaf_profiti.models.kaf_gaussian import KAFGaussian, KAFGaussianConfig
```

将 `_MODEL_SPECS` 中 `kafnet_gaussian` 的状态改为 `enabled`。

在 `create_model(...)` 中增加分支：

```python
    if name == "kafnet_gaussian":
        config = KAFGaussianConfig(
            num_sensors=num_sensors,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            te_dim=te_dim,
            kernel_count=kernel_count,
            n_layers=n_layers,
            n_heads=n_heads,
            preconv_dim=preconv_dim,
            lambda_point=lambda_point,
            device=device,
        )
        return KAFGaussian(config)
```

保留 `kaf_profiti_joint` 原分支。

- [ ] **步骤 6：更新模型导出**

在 `code/kaf_profiti/models/__init__.py` 增加：

```python
from .kaf_gaussian import KAFGaussian, KAFGaussianConfig
from .gaussian_head import GaussianHead
```

- [ ] **步骤 7：运行测试**

运行：

```bash
PYTHONPATH=code pytest code/tests/test_model_variants.py -q
```

预期：2 个测试通过。

- [ ] **步骤 8：运行现有模型测试**

运行：

```bash
PYTHONPATH=code pytest code/tests/test_model_components.py -q
```

预期：全部通过。

- [ ] **步骤 9：记录变更并提交**

在 `CHANGELOG.md` 的执行日期下追加：

```markdown
- 新增 `kafnet_gaussian` 消融模型，作为最终联合 flow 的概率输出对照。
```

提交：

```bash
git add code/tests/test_model_variants.py code/kaf_profiti/models/gaussian_head.py code/kaf_profiti/models/kaf_gaussian.py code/kaf_profiti/models/__init__.py code/kaf_profiti/experiments/registry.py CHANGELOG.md
git commit -m "model: 添加 kafnet gaussian 消融"
git push
```

## 任务 7：启用 Marginal Flow 与 w/o context 消融

**文件：**
- 修改：`code/tests/test_model_variants.py`
- 修改：`code/kaf_profiti/experiments/registry.py`
- 修改：`CHANGELOG.md`

- [ ] **步骤 1：添加失败测试**

在 `code/tests/test_model_variants.py` 追加：

```python
def test_marginal_and_no_context_variants_are_enabled():
    for name in ["kaf_profiti_marginal", "kaf_profiti_joint_no_context"]:
        spec = get_model_spec(name)
        assert spec.status == "enabled"
        model = create_model(
            name,
            num_sensors=4,
            context_dim=3,
            device="cpu",
            hidden_dim=16,
            te_dim=5,
            kernel_count=3,
            n_layers=1,
            n_heads=2,
            flow_layers=1,
            preconv_dim=4,
            lambda_point=0.1,
        )
        batch = make_batch()
        loss = model.loss(batch)
        assert torch.isfinite(loss)
```

- [ ] **步骤 2：运行测试确认失败**

```bash
PYTHONPATH=code pytest code/tests/test_model_variants.py::test_marginal_and_no_context_variants_are_enabled -q
```

预期：失败，原因是模型未启用或未注册。

- [ ] **步骤 3：更新模型注册状态**

在 `code/kaf_profiti/experiments/registry.py`：

- 将 `kaf_profiti_marginal` 状态改为 `enabled`。
- 新增：

```python
    "kaf_profiti_joint_no_context": ModelSpec(
        "kaf_profiti_joint_no_context",
        "KAFNet + ProFITi Joint Flow w/o Context",
        "enabled",
        "ablation",
    ),
```

- [ ] **步骤 4：更新 create_model 分支**

在 `create_model(...)` 中增加：

```python
    if name == "kaf_profiti_marginal":
        config = KAFProFITiConfig(
            num_sensors=num_sensors,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            te_dim=te_dim,
            kernel_count=kernel_count,
            n_layers=n_layers,
            n_heads=n_heads,
            flow_layers=flow_layers,
            preconv_dim=preconv_dim,
            lambda_point=lambda_point,
            marginal_training=True,
            device=device,
        )
        return KAFProFITi(config)

    if name == "kaf_profiti_joint_no_context":
        config = KAFProFITiConfig(
            num_sensors=num_sensors,
            context_dim=0,
            hidden_dim=hidden_dim,
            te_dim=te_dim,
            kernel_count=kernel_count,
            n_layers=n_layers,
            n_heads=n_heads,
            flow_layers=flow_layers,
            preconv_dim=preconv_dim,
            lambda_point=lambda_point,
            marginal_training=False,
            device=device,
        )
        return KAFProFITi(config)
```

在 `KAFProFITi.distribution(...)` 中处理 `context_dim=0`：

```python
        context = batch.context if self.config.context_dim > 0 else None
        z_var = self.encoder(batch.X_obs, batch.T_obs, batch.M_obs, context)
        self._hidden_states = self.adapter(
            z_var,
            batch.T_q,
            batch.query_channel_ids,
            context,
        )
```

- [ ] **步骤 5：运行测试**

```bash
PYTHONPATH=code pytest code/tests/test_model_variants.py -q
```

预期：所有模型变体测试通过。

- [ ] **步骤 6：记录变更并提交**

在 `CHANGELOG.md` 的执行日期下追加：

```markdown
- 启用 `kaf_profiti_marginal` 与 `kaf_profiti_joint_no_context` 两个核心消融变体。
```

提交：

```bash
git add code/tests/test_model_variants.py code/kaf_profiti/experiments/registry.py code/kaf_profiti/models/kaf_profiti.py CHANGELOG.md
git commit -m "model: 启用 flow 消融变体"
git push
```

## 任务 8：修正风险评分为阈值或标签驱动

**文件：**
- 创建：`code/tests/test_risk_metrics.py`
- 修改：`code/kaf_profiti/industrial/risk.py`
- 修改：`code/kaf_profiti/experiments/metrics.py`
- 修改：`code/run_experiment.py`
- 修改：`CHANGELOG.md`

- [ ] **步骤 1：写风险评分测试**

在 `code/tests/test_risk_metrics.py` 写入：

```python
import torch

from kaf_profiti.industrial.risk import threshold_risk_score


def test_threshold_risk_score_detects_upper_crossing():
    samples = torch.tensor(
        [
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[2.0, 0.0], [2.0, 0.0]],
                [[3.0, 0.0], [3.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )
    upper = torch.tensor([1.0, 5.0])
    lower = torch.tensor([-5.0, -5.0])
    risk = threshold_risk_score(samples, upper_limits=upper, lower_limits=lower)
    assert risk.shape == (1,)
    assert torch.isclose(risk[0], torch.tensor(2.0 / 3.0), atol=1e-6)


def test_threshold_risk_score_returns_zero_without_crossing():
    samples = torch.zeros(2, 4, 3, 2)
    upper = torch.ones(2)
    lower = -torch.ones(2)
    risk = threshold_risk_score(samples, upper_limits=upper, lower_limits=lower)
    assert torch.equal(risk, torch.zeros(2))
```

- [ ] **步骤 2：运行测试确认失败**

```bash
PYTHONPATH=code pytest code/tests/test_risk_metrics.py -q
```

预期：失败，原因是 `threshold_risk_score` 不存在或行为不匹配。

- [ ] **步骤 3：实现阈值风险评分**

在 `code/kaf_profiti/industrial/risk.py` 增加：

```python
import torch
from torch import Tensor


def threshold_risk_score(
    samples: Tensor,
    upper_limits: Tensor,
    lower_limits: Tensor,
) -> Tensor:
    if samples.dim() != 4:
        raise ValueError("samples must have shape (B, S, Lp, N)")
    upper = upper_limits.to(samples.device).view(1, 1, 1, -1)
    lower = lower_limits.to(samples.device).view(1, 1, 1, -1)
    if upper.shape[-1] != samples.shape[-1] or lower.shape[-1] != samples.shape[-1]:
        raise ValueError("threshold dimension must match sensor dimension")
    crossed = (samples > upper) | (samples < lower)
    any_crossed = crossed.any(dim=(2, 3)).float()
    return any_crossed.mean(dim=1)
```

- [ ] **步骤 4：替换 metrics 中的风险入口**

在 `code/kaf_profiti/experiments/metrics.py` 保留旧函数但改为明确名称：

```python
def heuristic_risk_score_from_samples(samples: Tensor) -> Tensor:
    return torch.sigmoid(samples.abs().mean(dim=(1, 2, 3)))
```

并增加：

```python
from kaf_profiti.industrial.risk import threshold_risk_score
```

- [ ] **步骤 5：更新 run_experiment**

在 `ExperimentConfig` 增加：

```python
risk_upper_quantile: float = 0.95
risk_lower_quantile: float = 0.05
```

在 `_evaluate(...)` 中用训练统计或 batch 目标构造阈值。第一版使用测试 batch 的目标分布只用于冒烟测试时会泄漏，因此执行时必须改为 `bundle` 中 train stats。若 train stats 尚无上下限，先在 `create_protocol_datasets` 返回的 `split_info` 中记录阈值来源，并在 `_evaluate` 参数中传入 train-derived thresholds。

具体实现采用：

```python
upper_limits = torch.tensor(bundle.risk_upper_limits, dtype=torch.float32, device=device)
lower_limits = torch.tensor(bundle.risk_lower_limits, dtype=torch.float32, device=device)
risk = threshold_risk_score(
    samples.reshape(samples.shape[0], config.nsamples, config.pred_len, num_sensors),
    upper_limits=upper_limits,
    lower_limits=lower_limits,
)
```

- [ ] **步骤 6：运行风险测试**

```bash
PYTHONPATH=code pytest code/tests/test_risk_metrics.py -q
```

预期：2 个测试通过。

- [ ] **步骤 7：运行实验框架测试**

```bash
PYTHONPATH=code pytest code/tests/test_experiment_framework.py -q
```

预期：全部通过；若数据路径不匹配，先记录失败原因，不修改数据集原始文件。

- [ ] **步骤 8：记录变更并提交**

在 `CHANGELOG.md` 的执行日期下追加：

```markdown
- 将风险评分从冒烟实验公式改为阈值驱动评分，并增加风险指标单元测试。
```

提交：

```bash
git add code/tests/test_risk_metrics.py code/kaf_profiti/industrial/risk.py code/kaf_profiti/experiments/metrics.py code/run_experiment.py CHANGELOG.md
git commit -m "eval: 添加阈值驱动风险评分"
git push
```

## 任务 9：增加多 seed 表格汇总

**文件：**
- 修改：`code/tests/test_experiment_framework.py`
- 修改：`code/kaf_profiti/experiments/tables.py`
- 修改：`CHANGELOG.md`

- [ ] **步骤 1：写表格汇总测试**

在 `code/tests/test_experiment_framework.py` 追加：

```python
def test_build_tables_adds_mean_std_summary(tmp_path):
    import json
    metrics_root = tmp_path / "metrics" / "metropt3" / "kaf_profiti_joint"
    metrics_root.mkdir(parents=True)
    for seed, mae in [(2026, 1.0), (2027, 1.2), (2028, 1.4)]:
        (metrics_root / f"metrics_seed{seed}.json").write_text(
            json.dumps(
                {
                    "dataset": "metropt3",
                    "model": "kaf_profiti_joint",
                    "seed": seed,
                    "missing_rate": 0.3,
                    "history": 12,
                    "horizon": 3,
                    "mae": mae,
                    "rmse": mae + 1.0,
                    "nll": mae + 2.0,
                    "crps": mae + 3.0,
                    "picp": 0.8,
                    "mpiw": 1.0,
                    "auroc": None,
                    "auprc": None,
                    "f1": None,
                    "ece": 0.1,
                    "lead_time": None,
                    "train_time_sec": 1.0,
                    "infer_time_ms_per_batch": 2.0,
                    "num_params": 10,
                    "gpu_memory_mb": None,
                    "status": "completed",
                    "error": None,
                }
            )
        )
    build_tables(tmp_path)
    summary = tmp_path / "tables" / "table7_statistical_test.csv"
    text = summary.read_text()
    assert "mae_mean" in text
    assert "mae_std" in text
    assert "kaf_profiti_joint" in text
```

- [ ] **步骤 2：运行测试确认失败**

```bash
PYTHONPATH=code pytest code/tests/test_experiment_framework.py::test_build_tables_adds_mean_std_summary -q
```

预期：失败，原因是 table7 还没有 `mae_mean` 和 `mae_std`。

- [ ] **步骤 3：实现 mean/std 汇总**

在 `code/kaf_profiti/experiments/tables.py` 中新增：

```python
def _mean_std_rows(metrics: List[Dict[str, object]]) -> List[Dict[str, object]]:
    frame = pd.DataFrame(metrics)
    numeric_cols = ["mae", "rmse", "nll", "crps", "picp", "auroc", "auprc", "f1", "ece"]
    rows = []
    for (dataset, model), group in frame.groupby(["dataset", "model"], dropna=False):
        row = {
            "dataset": dataset,
            "model": model,
            "seeds": ",".join(str(int(seed)) for seed in sorted(group["seed"].dropna().unique())),
            "runs": int(len(group)),
        }
        for col in numeric_cols:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            row[f"{col}_mean"] = float(values.mean()) if len(values) else None
            row[f"{col}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) == 1 else None
        rows.append(row)
    return rows
```

并将 table7 写入改为：

```python
    _write_csv(tables_dir / "table7_statistical_test.csv", _mean_std_rows(metrics))
```

- [ ] **步骤 4：运行测试**

```bash
PYTHONPATH=code pytest code/tests/test_experiment_framework.py::test_build_tables_adds_mean_std_summary -q
```

预期：通过。

- [ ] **步骤 5：运行表格构建测试**

```bash
PYTHONPATH=code pytest code/tests/test_experiment_framework.py::test_build_tables_reads_metrics_and_marks_not_implemented -q
```

预期：通过。

- [ ] **步骤 6：记录变更并提交**

在 `CHANGELOG.md` 的执行日期下追加：

```markdown
- 增加多 seed `mean ± std` 表格汇总逻辑，为统计检验表提供基础。
```

提交：

```bash
git add code/tests/test_experiment_framework.py code/kaf_profiti/experiments/tables.py CHANGELOG.md
git commit -m "eval: 添加多 seed 汇总表"
git push
```

## 任务 10：最终验证与上传检查

**文件：**
- 修改：`CHANGELOG.md`

- [ ] **步骤 1：运行文档占位扫描**

```bash
rg -n "TB[D]|TO[D]O|placeholde[r]|待[定]|未[定]|[?][?]" README.md docs/opening-report docs/references data-manifests CHANGELOG.md
```

预期：无输出。

- [ ] **步骤 2：运行大文件追踪检查**

```bash
git ls-files | rg "dataset/|\\.pt$|\\.pth$|\\.ckpt$|\\.npy$|__pycache__|\\.pytest_cache|\\.DS_Store"
```

预期：无输出。

- [ ] **步骤 3：运行核心测试**

```bash
PYTHONPATH=code pytest code/tests/test_model_components.py code/tests/test_model_variants.py code/tests/test_risk_metrics.py -q
```

预期：全部通过。

- [ ] **步骤 4：运行仓库状态检查**

```bash
git status --short
```

预期：只显示用户尚未归档的既有未追踪材料；本计划产生的文件都已提交。

- [ ] **步骤 5：记录最终变更**

在 `CHANGELOG.md` 的执行日期下追加：

```markdown
- 完成开题稳健版第一轮资产、消融入口、风险评分和多 seed 汇总验证。
```

- [ ] **步骤 6：提交并推送**

```bash
git add CHANGELOG.md
git commit -m "docs: 完成开题稳健版第一轮计划"
git push
```

## 自检

- 规格覆盖：本计划覆盖开题报告、文献核验、仓库治理、最小真实实验矩阵、消融、风险评分和多 seed 汇总。
- 占位扫描：使用上方命令检查交付文档，预期无输出。
- 类型一致性：模型注册名称固定为 `kafnet_gaussian`、`kaf_profiti_marginal`、`kaf_profiti_joint_no_context`；风险评分函数固定为 `threshold_risk_score`；文献表字段固定为 `language,year,title,authors,venue,topic,url,used_in,verified_date`。

## 执行交接

计划保存后有两种执行方式：

1. **子任务代理驱动（推荐）**：每个任务派发一个新的执行代理，任务间进行 review，迭代更快。
2. **当前会话内执行**：在本会话中使用 `executing-plans` 按批次执行，并在关键节点停下来 review。
