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
