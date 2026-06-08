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
