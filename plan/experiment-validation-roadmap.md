# 实验验真路线图

题目：面向工业设备异步多传感器的概率状态预测与风险评估方法研究

本路线图用于把现有原型实验推进为能够支撑论文三个创新点的验证体系。当前已有结果来自 `/Users/ppy/Downloads/tables`，覆盖 C-MAPSS FD001 与 MetroPT-3、4 个 KAF/KAF-ProFITi 消融模型、3 个 seed、缺失率 0.3。该结果可以支撑“原型可运行”和“C-MAPSS 核心消融有积极信号”，但还不能支撑“优于真实基线”或“风险评估有效”的最终结论。

## 1. 当前实验状态判断

当前完成的有效实验是核心消融实验，不是完整主实验。`table2_main_forecasting.csv` 中已完成 24 行，分别为 `kafnet_gaussian`、`kaf_profiti_marginal`、`kaf_profiti_joint_no_context`、`kaf_profiti_joint` 在两个主数据集和三个 seed 上的结果。真实强基线 `tcn_gaussian`、`patchtst_gaussian`、`gru_d`、`ode_rnn`、`mtan`、`grafiti`、`profiti` 和 `kafnet` 仍为 `not_implemented`。

C-MAPSS FD001 上，`kaf_profiti_joint` 相比 `kafnet_gaussian` 的三 seed 平均结果呈现改善趋势。MAE 从 0.4381 降至 0.4142，RMSE 从 0.7514 降至 0.5953，NLL 从 0.6472 降至 0.5718，CRPS 从 0.3704 降至 0.3104。这个结果可以作为开题或中期阶段的积极可行性证据，但仍需真实基线、多缺失率和统计检验补强。

MetroPT-3 上，风险指标目前不能作为有效证据。`table3_risk_prediction.csv` 中所有模型 AUROC 均为 0.5，AUPRC 和 F1 也完全相同，说明风险分数没有对正负样本形成区分。MetroPT-3 的 NLL 也出现数量级异常，flow 模型 NLL 远高于 Gaussian head。该问题必须先修复或诊断清楚，否则后续大规模补跑会放大无效结果。

## 2. 三个创新点与验真实验映射

### 创新点一：异步多传感器统一协议与规整表示

拟验证的问题是：显式保留观测掩码、异步缺失和工况上下文，是否比简单规则采样预测更适合工业多传感器状态预测。

对应实验包括数据协议正确性检查、缺失率鲁棒性实验和上下文消融。数据协议正确性检查需要确认 train-only 归一化、按 engine 或时间划分、同一缺失 mask 复用、不同模型输入口径一致。缺失率鲁棒性实验设置 `0, 0.1, 0.3, 0.5, 0.7`，至少在 `cmapss_fd001` 上比较 `kafnet_gaussian` 与 `kaf_profiti_joint`。上下文消融比较 `kaf_profiti_joint` 与 `kaf_profiti_joint_no_context`。

允许写入论文的结论边界是：若模型随缺失率升高退化更慢，可写“该表示在人工异步缺失条件下表现出更好的鲁棒性”；若上下文消融没有改善，则应写“工况上下文在当前数据协议下贡献有限”，不能强行写成创新有效。

### 创新点二：KAFNet-ProFITi 联合概率预测

拟验证的问题是：联合 flow 是否比独立 Gaussian 和 marginal flow 更能改善概率预测质量。

对应实验为核心消融表。模型包括 `kafnet_gaussian`、`kaf_profiti_marginal`、`kaf_profiti_joint_no_context`、`kaf_profiti_joint`。指标包括 MAE、RMSE、NLL、CRPS、PICP、MPIW 和 ECE。主判断不应只看 MAE/RMSE，而要把 NLL、CRPS、区间覆盖率和区间宽度一起解释。C-MAPSS FD001 当前结果已经显示 joint 模型在 MAE、RMSE、NLL 和 CRPS 上有平均改善趋势；MetroPT-3 需要先解决 NLL 异常后才能纳入概率质量主结论。

允许写入论文的结论边界是：若 joint flow 在 C-MAPSS 与至少一个真实基线上稳定改善 NLL/CRPS，可写“联合概率建模提升了概率预测质量”；若只改善点误差而 NLL/CRPS 不稳，应收束为“联合 flow 对部分数据集有效，概率校准仍需改进”。

### 创新点三：预测分布驱动的风险评估

拟验证的问题是：未来传感器状态分布能否转化为有区分能力、可校准的风险分数。

对应实验包括风险标签审计、风险分数分布审计、风险指标表和风险时间线图。C-MAPSS 使用 `RUL <= threshold` 构造风险标签，MetroPT-3 使用故障报告窗口或故障前预警窗口构造标签。风险指标包括 AUROC、AUPRC、F1、ECE、lead time、positive rate 和 score std。当前 MetroPT-3 的 AUROC 全为 0.5，说明该创新点还没有被实验支持，不能写为已验证。

允许写入论文的结论边界是：只有在风险分数具有非零方差、标签包含正负样本、AUROC/AUPRC/F1 高于常数基线，并且能给出故障前风险变化曲线时，才能写“预测分布可用于风险预警”。如果风险结果仍为 0.5，应把风险评估降为未来工作或仅保留为方法框架。

## 3. 第一步：修 MetroPT 风险评估与 NLL 异常

本步骤优先级最高。当前不要继续对 MetroPT-3 大规模补跑，先完成诊断。

### 3.1 风险评估诊断

需要在每次评估中额外保存诊断字段：`label_positive_rate`、`label_unique_count`、`risk_score_min`、`risk_score_max`、`risk_score_mean`、`risk_score_std`、`risk_upper_limits`、`risk_lower_limits`。如果 `risk_score_std` 接近 0，AUROC 必然接近 0.5；如果 `label_unique_count < 2`，风险指标应标记为不可用而不是写入 0.5。

MetroPT-3 当前 `rul` 字段实际保存的是 `future_fault`，即预测窗口内是否存在故障标签。这个设计会导致正样本稀少，并且只覆盖故障窗口内部，不一定覆盖故障前预警。下一步应增加可配置标签：`fault_window`、`pre_fault_1h`、`pre_fault_6h`、`pre_fault_24h`。开题和论文主文建议优先使用 `pre_fault_6h` 或 `pre_fault_24h`，因为风险评估关注提前预警，而不是故障已经发生后的识别。

风险分数当前来自预测样本是否越过 train split 5%/95% 传感器阈值。若所有样本几乎都越界或几乎都不越界，分数会退化。需要对比三种风险分数：阈值越界概率、预测均值偏离正常范围的距离分数、预测不确定性加权分数。只有在诊断图中能看到故障前分数上升，才保留为论文风险评分。

### 3.2 NLL 异常诊断

MetroPT-3 的 flow NLL 数量级异常，优先怀疑尺度、log-determinant 或 flow 训练稳定性。需要增加 batch 级诊断：`y_flat_abs_mean`、`hidden_abs_mean`、`z_abs_mean`、`ldj_mean`、`gaussian_nll_mean`、`nll_mean`、`nll_isfinite`。如果 `ldj` 或 `z` 爆炸，应先限制 flow scale 或减少 flow layers；如果 Gaussian head NLL 正常但 flow NLL 极大，说明异常在 flow 头，不在数据读取。

修复顺序建议为：先固定 MetroPT-3 风险标签和诊断输出，再检查 flow head 的尺度稳定性。必要时将 MetroPT-3 的主概率模型先降级为 `kafnet_gaussian` 与 `kaf_profiti_joint_no_context` 对比，把 joint flow 在 MetroPT-3 上作为失败案例讨论，而不是强行写成正向结果。

## 4. 第二步：补两个真实强基线

优先补 `TCN-Gaussian` 和 `GRU-D`，原因是二者能分别覆盖规则采样强基线和缺失感知时序基线，工作量可控，且与本课题问题边界直接相关。

`TCN-Gaussian` 的输入使用 `X_obs` 与 `M_obs` 拼接，也可加入时间 gap 通道。输出与现有 `GaussianHead` 对齐，计算 MAE、RMSE、NLL、CRPS、PICP、MPIW 和风险指标。它用于回答“普通规则采样卷积模型在相同缺失 mask 下能做到什么程度”。

`GRU-D` 的输入使用观测值、mask、delta time 和经验均值衰减机制，输出未来窗口的 Gaussian 分布。它用于回答“显式缺失衰减机制是否足以解决工业异步缺失问题”。实现时不必一开始追求原论文所有细节，先实现统一接口和公平评价；后续若结果接近主模型，再补更严格的参数调优。

真实基线完成后，主表至少应包含 `TCN-Gaussian`、`GRU-D`、`KAFNet-Gaussian`、`KAFNet-ProFITi Joint`。如果时间允许，再补 `PatchTST-Gaussian` 或 `ProFITi`。论文中“优于真实基线”的表述必须等这一步完成后才能写。

## 5. 第三步：跑缺失率矩阵

缺失率矩阵用于验证创新点一和方法鲁棒性。建议先跑最小矩阵：

```text
datasets: cmapss_fd001
models: tcn_gaussian, gru_d, kafnet_gaussian, kaf_profiti_joint
missing_rates: 0, 0.1, 0.3, 0.5, 0.7
seeds: 2026
history: 96
horizon: 24
```

确认趋势正常后，再扩展为三 seed：

```text
datasets: cmapss_fd001, metropt3
models: tcn_gaussian, gru_d, kafnet_gaussian, kaf_profiti_joint
missing_rates: 0, 0.1, 0.3, 0.5, 0.7
seeds: 2026, 2027, 2028
```

若 MetroPT-3 的风险和 NLL 仍未修复，缺失率矩阵先不要把 MetroPT-3 纳入主结论，只保留 C-MAPSS 主表和 MetroPT-3 附录诊断。

## 6. 第四步：完整三 seed 与统计检验

当前 `table7_statistical_test.csv` 已有 `mean ± std`，但 `p_value` 为空。后续需要补配对统计检验。由于每个模型使用相同 seeds，比较主模型与基线时应采用 paired test。seed 数只有 3 时，p 值解释能力有限，应以 `mean ± std`、逐 seed 差值和效应方向为主，p 值仅作辅助。

推荐统计方案为：主模型与每个基线按 seed 配对，指标为 MAE、RMSE、NLL、CRPS。若 seed 数扩展到 5 个以上，可使用配对 t 检验或 Wilcoxon signed-rank 检验；当前 3 seed 阶段，可报告逐 seed 差值和平均相对变化，不把显著性作为核心证据。表格中增加 `baseline_model`、`metric`、`mean_diff`、`relative_change_pct`、`p_value` 和 `test_name`。

论文表述建议为：三 seed 结果报告 `mean ± std`，统计检验作为补充说明。若 p 值不显著但均值方向一致，可以写“呈现稳定改善趋势”；不能写“显著优于”。

## 7. 推荐执行顺序

第一阶段锁定诊断。新增 MetroPT 风险和 NLL 诊断字段，重跑 `metropt3 + kafnet_gaussian + kaf_profiti_joint` 单 seed 小矩阵，确认 AUROC 不是常数 0.5，NLL 没有数量级异常。

第二阶段补基线。实现 `TCN-Gaussian` 和 `GRU-D`，在 `cmapss_fd001` 上跑 seed 2026、missing_rate 0.3，与当前 KAF 模型比较。

第三阶段跑缺失率。先跑 C-MAPSS FD001 的单 seed 缺失率矩阵，确认趋势后扩展三 seed。

第四阶段扩展主实验。跑 C-MAPSS FD001 与 MetroPT-3 的三 seed 主矩阵，生成主表、消融表、鲁棒性表、效率表和统计表。

第五阶段写结果章。只把通过诊断和真实基线支持的结论写进正文；风险评估若仍不稳定，保留为局限和后续工作。

## 8. 开题报告中的稳健表述

可以写：本研究已经完成 KAFNet-ProFITi 原型和核心消融实验，C-MAPSS FD001 的三 seed 结果显示联合概率模型在若干预测指标上呈现改善趋势。后续将优先修复 MetroPT-3 风险评分与概率 NLL 异常，补齐 TCN-Gaussian 和 GRU-D 两类真实基线，并在多缺失率和多随机种子条件下验证方法的鲁棒性。

不能写：本方法已经证明优于现有基线，或者风险评估模块已经有效。当前真实基线尚未实现，MetroPT-3 风险 AUROC 仍为常数基线，因此最终论文结论必须等待后续实验补齐。
