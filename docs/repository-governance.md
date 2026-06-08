# KAFNet-ProFITi 仓库治理规范

目标远端仓库：

<https://github.com/PY0909/KAFNet-ProFITi.git>

本仓库是以下论文的可复现研究工作区：

**面向工业设备异步多传感器的概率状态预测与风险评估方法研究**

仓库应保存代码、实验协议、已核验指标、论文草稿和变更记录。不要盲目上传原始数据、模型检查点、预测数组、本地缓存或私有环境文件。

## 上传原则

1. **可复现优先**：所有保存在 Git 中的结果都必须有对应配置、随机种子、划分规则、缺失 mask 规则、模型名称和指标文件。
2. **避免误传大文件**：原始数据、检查点、`.pt`、`.pth`、`.npy` 和生成的预测数组不进入普通 Git。
3. **追踪每次改动**：每次有意义的改动都必须反映在 `CHANGELOG.md` 中；实验改动还应反映在结果配置或实验日志中。
4. **区分源文件和生成产物**：源代码、配置和报告文本应跟踪；可再生成的输出只有在体积小且有 review 价值时才跟踪。
5. **明确第三方代码来源**：上游 KAFNet 和 ProFITi 代码应作为清晰标注的第三方目录保留许可证，或改成子模块/固定来源引用。

## 文件分类

| 类别 | 示例 | Git 策略 |
|---|---|---|
| 核心代码 | `code/kaf_profiti/**/*.py`、训练/评估脚本、测试 | 跟踪 |
| 实验配置 | `code/results/configs/*.yaml`、未来的 `configs/*.yaml` | 跟踪 |
| 指标和表格 | `code/results/metrics/**/*.json`、`code/results/tables/*.csv` | 小体积且核验后跟踪 |
| 划分清单 | `code/results/splits/*.json` | 小体积且与 seed 绑定时跟踪 |
| 图 | `code/results/figures/*.png`、`.svg` | 跟踪最终论文图；过大时使用 LFS |
| 论文材料 | 开题报告草稿、设计文档、核验参考文献表 | 跟踪 |
| 原始数据 | `dataset/`、UCI/NASA/TEP 下载文件、大型 CSV | 不跟踪；记录来源和预处理方式 |
| 大型预测产物 | `samples_seed*.npy`、`mean_seed*.npy`、`risk_seed*.npy` | 不跟踪；从 checkpoint/config 再生成 |
| 检查点 | `checkpoint_seed*.pt`、`.pth`、`.ckpt` | 普通 Git 不跟踪；必要时用 release assets 或外部存储 |
| 生成 mask | `.npz` masks | 通常不跟踪；用 seed/mode/rate 再生成，除非审计需要 |
| 缓存 | `__pycache__`、`.pytest_cache`、`.DS_Store` | 不跟踪 |
| 本地环境 | `.venv`、`.env`、conda 环境目录 | 不跟踪 |

## 推荐目录结构

```text
.
  CHANGELOG.md
  README.md
  requirement.txt
  docs/
    repository-governance.md
    superpowers/specs/
    opening-report/
    references/
  code/
    kaf_profiti/
    tests/
    results/
      configs/
      metrics/
      splits/
      tables/
      figures/
  third_party/
    KAFNet/
    ProFITi/
  data-manifests/
    metropt3.md
    cmapss.md
    tep.md
```

当前本地文件夹 `KAFNet-main/` 和 `ProFITi-main/` 后续应规范化到 `third_party/`，或改成 Git 子模块。不要在没有说明的情况下把修改后的项目代码和未修改的上游代码混在一起。

## 变更记录规则

每次有意义的更新都应在 `CHANGELOG.md` 中添加简短记录。

格式：

```text
## YYYY-MM-DD

- Added/Changed/Fixed: 简短描述。
- Experiment: 如果指标变化，记录 dataset/model/seed/missing-rate。
- Document: 如果写作变化，记录 report/spec/reference 文件。
```

对于实验结果，还应保留机器可读记录：

- `code/results/configs/{dataset}_{model}_seed{seed}.yaml`
- `code/results/metrics/{dataset}/{model}/metrics_seed{seed}.json`
- 由 metrics 重新构建的 `code/results/tables/*.csv`

## 提交信息规则

使用简短结构化提交信息：

```text
docs: add opening feasibility design
repo: add upload governance and gitignore
exp: add cmapss fd001 joint seed 2027 metrics
model: add gaussian ablation head
fix: correct metropt risk label window
```

建议 scope：

- `docs`
- `repo`
- `data`
- `model`
- `exp`
- `eval`
- `paper`
- `refs`
- `fix`

## 上传流程

1. 查看状态：

   ```bash
   git status --short
   ```

2. 检查改动文件，排除大型/生成产物。

3. 更新 `CHANGELOG.md`。

4. 只添加目标文件：

   ```bash
   git add <paths>
   ```

5. 提交：

   ```bash
   git commit -m "scope: concise message"
   ```

6. 确认提交不包含原始数据、检查点、预测数组、缓存或密钥后再推送：

   ```bash
   git push origin main
   ```

## 大文件策略

普通 Git 不应包含接近 GitHub 硬限制或频繁再生成的文件。

改用以下方式之一：

- 数据集来源链接加预处理脚本。
- 最终冻结检查点使用 release assets。
- 带 checksum 的外部存储。
- 仅在文件必要且有意版本化时使用 Git LFS。

对每个外部存储文件，应记录：

- 文件名
- 来源 URL 或存储位置
- checksum
- 生成命令
- 相关配置

## 参考文献追踪

核验后的参考文献应存放在：

```text
docs/references/references_verified.csv
```

必需字段：

```text
language,year,title,authors,venue,topic,url,used_in,verified_date
```

开题报告参考文献中只使用经过核验、年份为 2022-2026 且有真实链接的论文。
