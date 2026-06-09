# 变更记录

本文件记录论文与实验仓库层面的重要变更。

## 2026-06-09

- Fixed: 修复 MetroPT 风险标签协议，新增 `fault_window`、`pre_fault_1h`、`pre_fault_6h`、`pre_fault_24h` 可配置标签模式。
- Fixed: 将风险评分从“任一传感器/任一预测步越界”改为“阈值越界比例”，降低多传感器场景下风险分数饱和为 1 的问题。
- Added: 新增风险诊断与 NLL 诊断字段，包括标签正例率、风险分数范围/标准差、NLL 有限性、hidden/y 尺度和 flow log-det 诊断。
- Added: 新增 `TCN-Gaussian` 与 `GRU-D` 两个真实强基线，并接入统一实验注册表、矩阵脚本和 metrics 输出。
- Changed: `run_experiment_matrix.py --model-group all` 现在包含 `tcn_gaussian`、`gru_d` 和已有 KAF/KAF-ProFITi 消融模型。
- Experiment: 完成本地小规模 smoke 验证，`metropt3 + kafnet_gaussian`、`cmapss_fd001 + tcn_gaussian`、`cmapss_fd001 + gru_d` 均可写出 metrics 与风险诊断表。

## 2026-06-08

- 新增论文开题可行性设计。
- 新增仓库上传规范与文件分类治理规则。
- 新增 Git 忽略策略，防止误传原始数据、模型检查点、预测数组、缓存和本地环境。
- 将开题可行性设计与仓库治理文档统一为中文正式版本，后续不再保留英文版。
- 新增开题稳健版执行计划，覆盖开题报告、文献核验、仓库治理和最小真实实验矩阵。
- 新增 README 与三个数据清单，明确数据来源、实验用途和 Git 上传边界。
- 新增文献检索协议和核验表，后续参考文献必须先核验再写入开题报告。
- 新增开题报告 Markdown 草稿和写作说明，明确已有原型与计划实验的表述边界。
- 新增实验补齐路线，按消融、真实基线、缺失鲁棒性和风险评估分阶段推进。
- 完成首批真实参考文献核验，满足英文 20 篇以上、中文 10 篇以上和 2022-2026 年份要求。
- 新建 `kaf_profiti` conda 环境并安装本地小规模验证所需依赖。
- 新增 `environment.yml` 与环境说明，固定本地验证和服务器全量实验的 conda 入口。
- 新增 `kafnet_gaussian` 消融模型，作为最终联合 flow 的概率输出对照。
- 启用 `kaf_profiti_marginal` 与 `kaf_profiti_joint_no_context` 两个核心消融变体。
- 新增训练集阈值驱动的风险评分，避免用测试集目标泄漏构造风险指标。
- 新增多 seed `mean/std` 汇总表，支持全量实验后直接生成统计汇总。
- 完成本地小规模验证：组件测试、实验框架测试、Gaussian 消融和 Joint Flow smoke 均可运行。
- 新增 `dataset/` 占位目录和数据放置说明，服务器 clone 后可直接看到默认数据结构。
- 将本地完整 C-MAPSS 数据文件与压缩后的 MetroPT-3 CSV 纳入仓库，避免服务器缺少默认数据文件。
- 新增一键实验矩阵脚本，支持主模型与消融模型的多数据集、多 seed 批量运行和表格汇总。
