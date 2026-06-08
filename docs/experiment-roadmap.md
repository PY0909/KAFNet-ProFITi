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
