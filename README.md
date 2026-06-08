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
