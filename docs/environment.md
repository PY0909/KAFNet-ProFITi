# Conda 环境说明

本仓库使用独立 conda 环境 `kaf_profiti` 进行本地小规模验证和服务器全量实验。

## 创建环境

```bash
conda env create -f environment.yml
```

如果环境已经存在，可使用：

```bash
conda env update -n kaf_profiti -f environment.yml --prune
```

## 本地小规模验证

```bash
PYTHONPATH=code conda run -n kaf_profiti pytest code/tests/test_model_variants.py -q
PYTHONPATH=code conda run -n kaf_profiti pytest code/tests/test_model_components.py -q
```

## 服务器全量实验

服务器拉取代码后，先准备 `dataset/` 目录，再使用 `environment.yml` 创建环境。原始数据、模型检查点、预测数组和大型中间文件不进入 Git；实验配置、指标表和小体积结果按仓库治理规范上传。
