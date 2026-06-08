# 数据目录说明

本目录只保存数据放置说明，不保存真实原始数据。服务器 clone 仓库后，请按下面结构放置数据文件。

## C-MAPSS

默认路径：

```text
dataset/CMAPSSData/train_FD001.txt
dataset/CMAPSSData/test_FD001.txt
dataset/CMAPSSData/RUL_FD001.txt
```

扩展 FD002-FD004 时，同目录继续放置：

```text
train_FD002.txt
test_FD002.txt
RUL_FD002.txt
train_FD003.txt
test_FD003.txt
RUL_FD003.txt
train_FD004.txt
test_FD004.txt
RUL_FD004.txt
```

来源：NASA Prognostics Center of Excellence 数据集仓库中的 C-MAPSS Turbofan Engine Degradation Simulation Data Set。

## MetroPT-3

默认路径：

```text
dataset/metropt+3+dataset/MetroPT3(AirCompressor).csv.gz
```

代码第一次读取 MetroPT-3 时会自动解压得到 `MetroPT3(AirCompressor).csv`。

## TEP

默认路径：

```text
dataset/dataverse_files/TEP_FaultFree_Training.RData
dataset/dataverse_files/TEP_FaultFree_Testing.RData
dataset/dataverse_files/TEP_Faulty_Training.RData
dataset/dataverse_files/TEP_Faulty_Testing.RData
```

## 自定义数据根目录

如果数据不放在仓库默认 `dataset/` 下，运行实验时使用：

```bash
PYTHONPATH=code conda run -n kaf_profiti python code/run_experiment.py \
  --dataset cmapss_fd001 \
  --model kaf_profiti_joint \
  --data-root /path/to/data_root
```

其中 `/path/to/data_root` 是包含 `CMAPSSData/`、`metropt+3+dataset/` 或 `dataverse_files/` 的上一级目录。
