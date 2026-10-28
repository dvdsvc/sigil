<img src="assets/mark.svg" width="68" alt="Scientific document adaptation mark">

# Scientific Document Small-Model Adaptation

**LoRA-based continued pretraining and supervised fine-tuning for identifying geological sample, grain and analysis identifiers in noisy scientific documents.**

**English** · [简体中文](README.zh-CN.md)

Completed research experiments on Qwen3.5-0.8B: source-aware corpus preparation, controlled training, full-document regression evaluation and output-reliability analysis. This local showcase preserves research code and evidence while keeping article text, raw generations and model weights outside the candidate.

[Methods](docs/methods.md) · [Results](docs/results.md) · [Data](docs/data.md) · [Run and reproduce](docs/running.md) · [Provenance](docs/provenance.md) · [Local validation](docs/validation.md)

## Research question

Scientific identifiers must be copied from long, noisy papers and tables while distinguishing them from ages, analytical stages, instrument names and references. Can a small language model adapt to this domain, and how should recognition be measured when generated lists repeat or fail to terminate?

The primary experiment compares **direct SFT** with **CPT followed by the same SFT**. Both arms start from the untouched original base and identical fresh LoRA tensors. LoRA is the parameter-efficient adaptation technique; CPT and SFT use different training objectives on those parameters. They are not three competing model architectures.

## Research work

| Component | Implementation and evidence |
|---|---|
| Scientific-document preparation | Preserve body order, table rows and Office Math text; screen source aliases and exact/normalized document hashes; isolate known evaluation sources. [Projection](src/historical/project_docx_with_math.py), [preparation](src/historical/prepare_domain_pretraining_20261014.py) |
| Controlled parameter-efficient training | Fresh initialization hashes; seven projection targets; answer-only SFT loss; bf16 and gradient checkpointing; identical SFT schedules and optimizer reset. [Actual trainer](src/historical/train_domain_pretraining_ab_20261014.py), [configuration](evidence/primary/config.json) |
| CPT versus direct SFT | 128 CPT articles, 1,464 updates and 2,860,163 target tokens; both SFT arms use 64 complete articles, 72 updates and 22,659 answer tokens. [Training evidence](evidence/primary) |
| Recognition and delivery evaluation | Separate strict JSON delivery, conservative prefix recognition, repeated items, nonliteral tails and article-level aggregation. [Original pure parsers](src/output_metrics.py), [truncation audit](reports/OUTPUT_TRUNCATION_REVIEW_ZH.md) |
| Structure-aware inference analysis | A separate fixed-adapter study preserves every article character while keeping table headers, inserted sample descriptions and data together. [Corrected evaluator](src/historical/evaluate_structural_chunks_tablefix_20261014.py), [report](reports/STRUCTURAL_BOUNDARY_REVIEW_ZH.md) |

The implementation uses existing Transformers and PEFT capabilities. The research contribution lies in problem definition, data construction, controlled adaptation, full-document evaluation and error analysis. Upstream model architectures, LoRA and standard CPT/SFT objectives retain their original attribution.

## Observed results

The primary runs trained **3,194,880 LoRA parameters** on an RTX 3090. Fixed language-model validation loss decreased from **2.1534 to 1.9583 (−9.06%)** after CPT. Both SFT runs used complete inputs, with the longest training input reaching 59,864 tokens.

| Fixed 19-article regression panel | Direct SFT | CPT → same SFT |
|---|---:|---:|
| Complete JSON usable | 12/19 | 12/19 |
| Known identifiers recovered using complete JSON only | 23/210 | 18/210 |
| Known identifiers recovered including complete strings before truncation | 52/210 | 28/210 |
| Article-equal known recall under prefix recovery | 47.61% | 51.14% |

The micro and article-equal directions differ. The experiment shows domain language-loss improvement without a consistent extraction gain across these measures. Fourteen truncated outputs were audited as eight repetition loops and six nonliteral extension tails. An invalid complete JSON document can still contain recognized identifiers; these are measured separately.

The labels are incomplete and the 19 articles were repeatedly inspected development/regression documents. Counts measure recovery of the **known inventory**, not true full-document precision/F1 or a blind-test benchmark. CPT uses extra training compute. The structure study uses a different already-trained adapter and has its own corrected result (94/210 whole-document versus 75/210 segmented); it is not an additional CPT arm. [Result scopes and corrections](docs/results.md).

## Offline quick start

Python 3.11+ and the standard library suffice; no article corpus, weights, model credentials or GPU is needed:

```sh
uv run --offline --no-project python -B scripts/reproduce_results.py
uv run --offline --no-project python -B scripts/demo.py
uv run --offline --no-project python -B -m unittest discover -s tests -v
uv run --offline --no-project python -B scripts/check_repository.py
```

Use `--python /path/to/python3.12` before `python -B` if an explicit interpreter is needed. Direct `python -B ...` commands also work. The reproduction command checks training logs/schedules and recomputes released aggregates from article-level counts. The demo uses hand-written fictional examples to exercise the real parsing rules; it is not model inference or model-performance evidence.

## Repository map

| Directory | Contents |
|---|---|
| `src/historical/` | Preserved training, document projection and evaluation implementations; limited path relocation is recorded |
| `src/output_metrics.py` | Existing strict/prefix parsers extracted without the original publication automation |
| `evidence/primary/` | Config, initialization hashes, training logs/schedules, validation losses and no-text per-article counts |
| `evidence/structure/` | Separate table-boundary-corrected inference comparison |
| `reports/` | Selected research reports with source aliases and private locations generalized |
| `scripts/`, `tests/` | Safe offline verification, parser examples/tests and optional owner-restored workspace setup |
| `requirements/` | Actual historical training environment and selected recorded dependency pins |
| `provenance/` | Source/package hashes, transformations and private-audit receipt |

Fresh training requires separately restored private corpora and original model files. [Runtime instructions](docs/running.md) explain the historical directory dependency and provide an explicit new-workspace staging helper. Offline verification does not claim a new GPU training run. The original private archive remains unchanged. [License and attribution](THIRD_PARTY_NOTICES.md).

Configuration: [Environment variables](docs/security.md).
