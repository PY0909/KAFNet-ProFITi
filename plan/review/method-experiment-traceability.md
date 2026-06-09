# 方法-实验可追踪表

| 创新点 | 方法模块 | 必要实验 | 表/图 | 允许结论 | 当前证据状态 |
|---|---|---|---|---|---|
| 异步多传感器统一协议与规整表示 | 统一数据划分、train-only 归一化、共享缺失 mask、KAFNetEncoder、context adapter | 缺失率矩阵；上下文消融；数据协议审计 | Table 1；Missing robustness；Ablation | 可写“在统一缺失协议下具有鲁棒性趋势”，不能只凭单缺失率结论泛化 | 只有 missing_rate=0.3，缺 0/0.1/0.5/0.7 |
| KAFNet-ProFITi 联合概率预测 | KAFNetEncoder + QueryConditionAdapter + ProFITi Joint Flow | Gaussian vs marginal flow vs joint flow；真实基线对比 | Main forecasting；Ablation；Statistical summary | 若 NLL/CRPS 稳定改善，可写“概率预测质量改善”；若只改善 MAE，只能写“点误差改善趋势” | C-MAPSS 有积极消融信号；MetroPT NLL 异常 |
| 预测分布驱动风险评估 | threshold risk score；风险标签；ECE；lead time | 风险标签审计；风险分数分布审计；AUROC/AUPRC/F1；风险时间线 | Risk prediction；Calibration；Risk timeline | 只有 AUROC/AUPRC/F1 高于常数基线且风险曲线有提前变化时，才能写“支持风险预警” | 当前 MetroPT AUROC=0.5，C-MAPSS 风险指标为空，不支持正向结论 |
| 真实基线公平比较 | TCN-Gaussian；GRU-D；后续 PatchTST/ProFITi | 同划分、同 mask、同 Gaussian 概率头、同 seeds | Main forecasting；Efficiency | 可写“相对真实基线的性能比较” | 未实现，当前不可写优于真实基线 |
| 多 seed 统计可靠性 | 结果聚合、paired seed comparison | 3 seed mean ± std；配对差值；p-value 或非参数检验 | Statistical test | 3 seed 阶段以趋势和均值方差为主，不写显著优于 | table7 有 mean ± std，p_value 为空 |
