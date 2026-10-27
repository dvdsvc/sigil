<img src="assets/mark.svg" width="68" alt="科学文档模型适配标志">

# 科学文档小模型适配：LoRA、CPT 与 SFT

**面向含噪地质文献中的样品、颗粒与分析编号识别，开展参数高效训练、同起点对照和完整文献评测。**

[English](README.md) · **简体中文**

以 Qwen3.5-0.8B 为底座，已完成真实文献语料准备、LoRA 训练、领域继续预训练与监督微调对照，以及输出可靠性审计。此展示副本保留研究实现与证据，原始文章、原始生成结果和模型权重留在私有归档。

[研究方法](docs/methods.md) · [实验结果](docs/results.md) · [数据说明](docs/data.md) · [运行与复核](docs/running.md) · [代码来源](docs/provenance.md) · [本地检查](docs/validation.md)

## 研究问题

科学编号往往分布在长正文和表格中，需要与年龄、加热阶段、仪器名称和参考文献编号区分。研究关注小模型如何学习这些领域文本与抽取任务，以及如何区分“识别到了编号”和“可靠地交付完整列表”。

核心对照为**直接 SFT**与**先 CPT、再执行相同 SFT**：共用未修改的原始底座与完全相同的新 LoRA 初始化。LoRA 是参数适配方式，CPT 与 SFT 则对应不同训练目标；三者共同构成训练方案。

## 已完成的研究工作

| 环节 | 实现与证据 |
|---|---|
| 科学文档与语料构建 | 保留正文顺序、表格行和 Office Math 文字；通过来源别名、DOCX、原文及归一化哈希筛查分组与重复。[文档投影](src/historical/project_docx_with_math.py)、[数据准备](src/historical/prepare_domain_pretraining_20261014.py) |
| 参数高效适配与对照控制 | 核对底座和 LoRA 初始张量哈希，设置七类投影层目标、答案部分损失、bf16 与梯度检查点，对齐 SFT 输入顺序、标签、随机种子和优化器重置。[实际训练代码](src/historical/train_domain_pretraining_ab_20261014.py)、[配置](evidence/primary/config.json) |
| 真实领域 CPT / SFT | 128 篇文章、1,464 次 CPT 更新、约 286 万目标 token；两组均使用 64 篇完整文章、72 次 SFT 更新、22,659 个答案 token。[训练记录](evidence/primary) |
| 完整文献评测与纠错 | 区分完整 JSON、截断前可恢复编号、重复循环和非原文续造；同时报告编号微平均与文章等权指标。[解析方法](src/output_metrics.py)、[输出审计](reports/OUTPUT_TRUNCATION_REVIEW_ZH.md) |
| 文档结构与推理策略分析 | 使用另一份固定权重完成结构分段实验，保持全文字符覆盖，修正表头、插入样品说明和数据行的绑定边界。[修正版评测代码](src/historical/evaluate_structural_chunks_tablefix_20261014.py)、[报告](reports/STRUCTURAL_BOUNDARY_REVIEW_ZH.md) |

模型加载与参数适配使用 Transformers、PEFT 等已有实现。本项目体现任务定义、语料构建、受控训练、长文档评测和实验诊断；LoRA 方法、底座架构及标准训练目标保留上游归属。

## 实际观察

两组在 RTX 3090 上训练 **3,194,880 个 LoRA 参数**。CPT 后固定语言模型验证损失由 **2.1534 降至 1.9583，下降约 9.06%**；两组 SFT 均未裁切文章，最长实际训练输入为 59,864 token。

| 固定 19 篇完整文章回归评测 | 直接 SFT | CPT 后相同 SFT |
|---|---:|---:|
| 完整 JSON 可用 | 12/19 | 12/19 |
| 仅消费完整 JSON 的已知编号恢复 | 23/210 | 18/210 |
| 包含截断前完整字符串的已知编号恢复 | 52/210 | 28/210 |
| 可恢复口径下的文章等权已知召回 | 47.61% | 51.14% |

按编号总量与按文章等权汇总的方向不同：CPT 改善了固定文本上的语言预测，但抽取收益没有在各指标中一致体现。14 次截断分别出现 8 次重复循环和 6 次非原文编号续造；JSON 未完成时仍可能已经识别部分编号，需要单独测量。

19 篇是反复检查过的开发／回归文献，已知清单并不穷尽，指标对应已知编号恢复，不是真实全文精确率/F1，也不作为新盲测成绩。CPT 组有额外训练算力。结构分段研究采用另一份已训练权重，修正后整篇为 94/210、分段为 75/210，与本组 CPT 对照分别解释。[结果与版本关系](docs/results.md)。

## 离线运行

Python 3.11+ 与标准库即可，不需要原始文章、权重、模型密钥或 GPU：

```sh
uv run --offline --no-project python -B scripts/reproduce_results.py
uv run --offline --no-project python -B scripts/demo.py
uv run --offline --no-project python -B -m unittest discover -s tests -v
uv run --offline --no-project python -B scripts/check_repository.py
```

需要指定解释器时，可在 `python -B` 前加 `--python /path/to/python3.12`；也可直接使用 `python -B ...`。复核命令检查训练日志与输入安排，并从无全文的逐篇计数重新汇总指标。示例使用手写虚构输入演示真实解析规则，不执行模型，也不作为模型效果证据。

## 目录与恢复

| 目录 | 内容 |
|---|---|
| `src/historical/` | 原有训练、文档投影、评估代码，必要的路径迁移有单独记录 |
| `src/output_metrics.py` | 从原代码提取的严格解析与保守前缀恢复，去除发布自动化 |
| `evidence/primary/` | 配置、初始化哈希、训练日志、输入安排、验证损失及无全文逐篇计数 |
| `evidence/structure/` | 独立的表格边界修正版推理实验 |
| `reports/` | 精选研究报告，文章与私有位置使用别名 |
| `scripts/`、`tests/` | 离线复核、解析示例与检查、私有材料恢复后的工作区准备 |
| `requirements/`、`provenance/` | 真实训练环境、依赖版本、原/副本哈希和变换记录 |

重新训练需要作者另行恢复私有语料与原始底座。[运行说明](docs/running.md)解释历史目录依赖，并提供显式创建新工作区的准备脚本。本次验证覆盖归档结果与离线入口，没有重新进行 GPU 训练。原始归档保留不动；许可和上游归属见[第三方说明](THIRD_PARTY_NOTICES.md)。

凭据配置与发布前检查：[安全说明](docs/security.md)。
