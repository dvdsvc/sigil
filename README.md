# Sigil: Small Language Model Adaptation for Scientific Identifier Extraction

**Scientific Identifier Extraction · LoRA Fine-Tuning · Domain Adaptation**

[English](README.md) · [简体中文](README.zh-CN.md)

This project uses small language models to identify sample, grain, spot and analysis identifiers in geochronology papers. It studies how domain training, task organization and long-document input affect recognition behavior.

The starting question is specific: **can a small model reliably distinguish identifiers of scientific study objects from text noise with a similar surface form?** The project investigates this through LoRA adaptation, continued pretraining versus supervised fine-tuning, and analysis of generation behavior on complete papers.

## 1. From Entity Recognition to Small Language Models

Early exploration involved triples, BERT-NER, GLiNER, word-frequency analysis and masked-language-model scoring. On real geological literature, these approaches could admit place names, ordinary entities, unusual punctuation and character fragments introduced during document parsing.

Geological sample and analysis identifiers follow no uniform naming convention. Determining whether a character sequence denotes a sample, an analytical spot, a location or an instrument often requires both prose and table context. Surface patterns and local statistics alone have difficulty resolving these ambiguities.

Subsequent work moved to small generative language models, using contextual modeling and domain LoRA training for identifier recognition. The task is to copy identifiers literally present in the document, excluding unrelated numbers such as ages, isotope ratios and citation indices. It also prohibits inferring an unwritten parent sample name or extending a numbering sequence.

The early traditional-method comparisons come from exploratory records. The archived training and quantitative evidence available for direct verification primarily concerns the subsequent small-model experiments.

## 2. Model and Training Design

The project uses **Qwen3.5-0.8B** with LoRA on a single **RTX 3090**, with **3,194,880 trainable parameters** in the recorded runs. Transformers and PEFT provide model loading and parameter adaptation; the training implementation controls loss scope, input order, optimization and checkpointing.

Training uses real geological papers containing both prose and tables. DOCX preprocessing preserves paragraphs, table rows and Office Math text in document order. Mathematics is retained as a text projection; image OCR and equation-layout reconstruction are outside this preprocessing step.

Two output contracts have been explored. The following examples illustrate their formats.

**Identifier recognition without types:**

```json
{
  "entity_ids": ["MZ04A", "MZ04A-01", "MZ04A-02"]
}
```

**Identifier recognition with types:**

```json
{
  "sample_ids": ["MZ04A"],
  "spot_ids": ["MZ04A-01", "MZ04A-02"]
}
```

These formats correspond to different task organizations. Earlier routes study recognition and type assignment together. The later CPT comparison uses `entity_ids` throughout, focusing on identifier recognition and output completeness.

| Primary CPT/SFT setting | Recorded configuration |
|---|---|
| LoRA | Rank 8, alpha 16, dropout 0.05; seven attention/feed-forward projection targets |
| Compute and memory | bf16, batch size 1, gradient checkpointing and SDPA |
| CPT objective | Autoregressive language modeling on raw domain text |
| SFT objective | Cross entropy on the identifier JSON answer and EOS only |
| Complete-document input | No article truncation during SFT; maximum recorded training input of 59,864 tokens |

[Training implementation](src/historical/train_domain_pretraining_ab_20261014.py) · [Document preprocessing](src/historical/project_docx_with_math.py) · [Configuration](evidence/primary/config.json) · [Methods](docs/methods.md)

## 3. Research Explorations

### 3.1 Recognize First, Then Classify—or Train Typed Extraction Directly?

Early experiments explored two training routes:

- **Route A:** learn to extract the union of all identifiers, then train typed extraction.
- **Route B:** train directly to identify and distinguish samples and spots from the source text.

The initial intuition was that learning shared identifier patterns before type distinctions might simplify training. Here, the second stage still generates typed lists from the original text; it is not a separate classifier consuming first-stage predictions.

The historical results show tradeoffs that depend on the evaluation set. For example, on development fragments after full-document continuation training, direct typed training reduced false positives with slightly lower recall. On held-out tables, that route had both higher precision and higher recall. These results inform analysis of task organization without establishing a universally superior route.

Some historical studies inherit different parent adapters and use different cumulative update counts. They compare training routes and are interpreted separately from the common-origin CPT experiment below. The resulting research question is: **how do training objectives and task order affect domain adaptation?**

[Strategy comparison](src/historical/train_strategy_comparison.py) · [Full-document training routes](src/historical/train_real_full_routes.py) · [Initialization and route lineage](reports/HISTORICAL_ROUTE_LINEAGE_ZH.md)

### 3.2 Does Domain Continued Pretraining Improve Extraction?

A separate experiment compares **direct SFT** with **CPT → SFT**. Both arms start from the same original base and identical fresh LoRA initialization. SFT data, input order, labels, random seeds and settings are aligned, with a fresh optimizer at the start of SFT in both arms.

CPT uses 128 domain papers and approximately 2.86 million target tokens over 1,464 updates. Both arms then use 64 complete real articles for the same 72-update SFT schedule. The CPT arm receives extra training before the shared SFT stage, so total compute is not matched.

The fixed language-model validation set contains 8 articles, 32 sequences and 54,960 target tokens. CPT reduces validation loss from **2.1534 to 1.9583, a relative decrease of approximately 9.06%**, but the improvement does not consistently translate into identifier extraction gains.

| Comparison on 19 complete papers | Direct SFT | CPT → SFT |
|---|---:|---:|
| Complete JSON usable | 12/19 | 12/19 |
| Known identifiers recovered when consuming complete JSON only | 23/210 | 18/210 |
| Known identifiers recovered including complete pre-truncation strings | 52/210 | 28/210 |
| Article-equal known recall with prefix recovery | 47.61% | 51.14% |

Article-equal recall averages over the **11 papers with nonempty known-identifier inventories**. The remaining eight empty inventories are unknown and still contribute to delivery statistics. Total identifier recovery and article-equal recall move in different directions; CPT does not improve, or worsen, every recognition metric consistently.

The result suggests that **better domain language-model loss does not necessarily yield a consistent improvement in downstream extraction**. Language prediction, identifier recognition and structured output require separate measurement.

[Configuration and training records](evidence/primary/config.json) · [Frozen comparison](evidence/primary/INDEPENDENT_COMPARISON.json) · [Detailed results](docs/results.md)

### 3.3 Recognition and Output Delivery

Generative identifier extraction requires finding correct names and terminating the list normally. Across the 38 full-document generations in the two arms, 14 reach the output limit. Review identifies **eight clear repetition loops** and **six nonliteral identifier-sequence extensions**.

Some outputs contain correct identifiers before entering a loop. Treating every incomplete JSON response as zero recognition would conflate recognition errors with delivery failure.

The project therefore retains two measurement rules. Strict parsing accepts complete `entity_ids` JSON. Conservative prefix recovery reads complete strings sequentially from the beginning of the actual output list, stops at the first truncated or malformed item, and deduplicates within each article. It neither completes a partial identifier nor uses labels to select predictions.

Together, these rules distinguish recognized identifiers, complete-output usability and abnormal generation. Correct identifiers in a prefix do not establish completion of the extraction task.

[Output parsing](src/output_metrics.py) · [Truncation analysis](reports/OUTPUT_TRUNCATION_REVIEW_ZH.md) · [Audit counts](evidence/primary/OUTPUT_TRUNCATION_REVIEW.json)

### 3.4 Complete Documents and Structural Segmentation

A separate experiment holds an already-trained model fixed and divides the same 19 papers into 410 chunks. It aims to keep table headers, sample descriptions and data rows together, then merges and deduplicates predictions within each article.

The early analysis had table-boundary and labeling issues: some headers were separated from data, and 13 heating-stage values were mislabeled as spot identifiers. The corrected study adjusts the boundaries and removes these 13 confirmed label errors:

| Metric | Whole document | Structural chunks |
|---|---:|---:|
| Known identifiers recovered | 94/210 | 75/210 |
| Article-equal known recall | 57.44% | 38.77% |
| Predicted identifiers, summed after within-article deduplication | 481 | 678 |

Article-equal recall again uses the 11 nonempty known inventories. All 410 chunk outputs are parseable without truncation, and **the whole-document condition also has no output truncation**. Segmentation does not improve known-identifier recall and increases additional candidates that have not been individually reviewed.

For this model and document panel, **shorter input does not automatically improve recognition**. Context organization, task understanding and generation behavior warrant further study; the experiment does not isolate which is the principal cause.

This study uses a different fixed adapter from the CPT comparison. Of the 410 inputs, 324 identical inputs reuse historical outputs and 86 changed inputs were newly inferred in the original experiment. Call counts, total output budget and repeated context also differ from the whole-document condition.

[Structural segmentation](src/historical/evaluate_structural_chunks_tablefix_20261014.py) · [Corrected report](reports/STRUCTURAL_BOUNDARY_REVIEW_ZH.md) · [Article-level counts](evidence/structure/article_counts.json)

## 4. Current Findings and Next Steps

The experiments expand the initial question of whether a small model can extract identifiers into four research questions:

1. How do domain knowledge and context help distinguish real identifiers from text noise?
2. How does the order of recognition and type-assignment training affect false positives, omissions and type errors?
3. Why can continued pretraining improve language prediction without consistently improving downstream extraction?
4. How should entity omission, additional generation and output-delivery failure be measured separately?

The full-document evaluation uses incomplete known-identifier inventories, and the 19 papers were repeatedly examined during development. The results are development/regression comparisons. Additional predictions have not all been adjudicated, so they do not support full precision or F1 under an independent blind-test protocol.

Next steps are to build more complete annotations using geochronology databases and manual review, then evaluate how training strategies, document organization and output constraints affect recognition and delivery completeness.

## 5. Code and Reproduction

The repository retains training implementations, experiment configurations, output evaluation rules and statistical records with source text removed.

| Entry | Contents |
|---|---|
| [CPT/SFT comparison](src/historical/train_domain_pretraining_ab_20261014.py) | Common base, fresh LoRA initialization and aligned SFT stage |
| [Extraction strategies](src/historical/train_strategy_comparison.py) | Union-then-typed versus direct typed training |
| [Full-document routes](src/historical/train_real_full_routes.py) | Complete-paper training from historical parent adapters |
| [Document projection](src/historical/project_docx_with_math.py) | Prose, tables and Office Math text |
| [Structural segmentation](src/historical/evaluate_structural_chunks_tablefix_20261014.py) | Corrected table boundaries and chunk aggregation |
| [Result reproduction](scripts/reproduce_results.py) | Training schedules, logs, validation loss and per-article aggregate checks |

Python 3.11+ and the standard library are sufficient for offline verification. No GPU, original papers or model weights are required:

```bash
uv run --offline --no-project python -B scripts/reproduce_results.py
uv run --offline --no-project python -B scripts/demo.py
uv run --offline --no-project python -B -m unittest discover -s tests -v
uv run --offline --no-project python -B scripts/check_repository.py
```

The corresponding `python -B ...` commands can also be run directly. Reproduction recomputes statistics from released article-level counts and checks the training records. Re-parsing original generations requires the private archive. The parser demo uses handwritten fictional input and does not run a model.

Full training requires the original corpus, a matching base model and a GPU environment. Model weights, paper full texts and raw generations are excluded from the presentation version. See [Running](docs/running.md), [Data](docs/data.md) and [Provenance](docs/provenance.md) for historical script dependencies, workspace restoration and upstream attribution.

[Detailed results](docs/results.md) · [Training environment](requirements/TRAINING_ENVIRONMENT.json) · [Third-party notices](THIRD_PARTY_NOTICES.md)
