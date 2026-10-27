# Research methods and implementation

[Home](../README.md) · [Results](results.md) · [Data](data.md)

## Task and representation

The primary target is the deduplicated union of geological sample, individual grain/spot and analysis identifiers literally present in a document. The model emits `{"entity_ids": [...]}` in first-appearance order. It is not trained here to extract complete age records, relations or scientific conclusions. Earlier route experiments emitted typed `sample_ids` and `spot_ids`; their supervision and initialization differ from the primary CPT experiment.

DOCX projection traverses body paragraphs and tables in order, retaining both ordinary `w:t` and Office Math `m:t` text. Tables serialize row cells with delimiters and a table-block marker. This is a plain-text projection; it does not recover equation layout, scan images or perform OCR. [Projection implementation](../src/historical/project_docx_with_math.py).

## Source-aware preparation

The preparation inspected 335 DOCX files, excluded 90 known validation/test sources and 15 duplicates, leaving 230 eligible sources. Screening uses existing source groups, filename/title/leading DOI aliases, DOCX hashes and exact/normalized projected-text hashes. Eight language-model validation sources are chosen independently of known SFT sources, then 128 eligible CPT sources in a fixed hash order. Selection rules do not prove exhaustive bibliographic identity resolution.

SFT retains 64 complete real documents: 60 positive-source documents once and four reviewed negatives three times per schedule, yielding 72 updates. Union labels are deduplicated, ordered by literal occurrence and checked against their source. The evaluation is the same 19 previously inspected articles; 13 confirmed heating-stage values were removed from the known inventory, leaving 210 known identifiers. Empty inventories remain unknown rather than negative gold.

## LoRA configuration and training objectives

| Parameter or control | Actual primary implementation |
|---|---|
| Base | Original Qwen3.5-0.8B; weight hash checked before training |
| Adaptation | PEFT LoRA, rank 8, alpha 16, dropout 0.05, bias none |
| Targets | `down_proj`, `gate_proj`, `q_proj`, `v_proj`, `k_proj`, `up_proj`, `o_proj` |
| Trainable parameters | 3,194,880 in the recorded runs |
| Precision / memory | bf16, batch 1, gradient accumulation 1, gradient checkpointing, SDPA |
| Optimizer | AdamW, weight decay 0, gradient norm clip 1; fresh optimizer for SFT |
| Initialization | Same seed, original base, identical freshly initialized trainable-tensor hash |
| Checkpoint policy | Fixed final CPT epoch and final 72-step SFT; no test-based checkpoint choice |

**CPT:** Causal next-token loss over raw article tokens and EOS, with 2,048-token sequences. The first sequence uses an EOS predecessor; later sequences carry the previous article token. Each article token is a target once, without cross-document packing, chat formatting or identifier labels. One epoch has 1,464 updates / 2,860,163 target tokens, learning rate 2e-5, warmup 32.

**SFT:** Answer-only cross entropy over the identifier JSON plus EOS. The code uses the answer suffix logits rather than making document prompt tokens targets. Both arms use the same 72 inputs/answers, order, seed reset, fresh optimizer, learning rate 5e-5 and warmup 5. Full document input is retained; maximum recorded training input is 59,864 tokens. The CPT arm receives extra updates before this common SFT stage, so total compute is not matched.

The [trainer](../src/historical/train_domain_pretraining_ab_20261014.py) is retained byte for byte. The [public log verifier](../scripts/reproduce_results.py) checks initialization, per-step IDs/text hashes/token counts, update totals and exact-text split intersections. Alias-level identity checks remain in the private preparation audit.

## Full-document evaluation and output audit

The primary arms use the same full inputs, greedy decoding, thinking disabled and 4,096 generated-token cap. The strict parser accepts exactly the `entity_ids` field containing strings and deduplicates them. Complete delivery is one outcome. Conservative prefix recognition is a second: read only sequential complete JSON strings at the beginning of the actual list, stop at the first malformed/non-string item or partial string, then deduplicate. It does not complete a cut identifier or select outputs using labels. Duplicate occurrences remain available for loop analysis.

Report micro recovery over known identifiers and an equal mean over the 11 articles with nonempty known inventories. Unknown empty inventories are included in delivery counts, but not recall denominators. Extra candidates are unreviewed, not automatically false positives. Literal presence is a coarse grounding flag, not semantic correctness. Frozen generation behavior was reviewed separately from strict JSON success.

## Separate structure-boundary study

The corrected [structure evaluator](../src/historical/evaluate_structural_chunks_tablefix_20261014.py) uses another fixed final adapter (`08921080…`) and no new training. It constructs contiguous original character slices, keeps headings and table captions/headers/inserted sample descriptions with data, repeats literal header context for long-table pieces, and checks full-body reconstruction. It does not select windows using known identifier labels.

The corrected corpus contains 410 chunks. Of these, 324 byte-identical inputs reuse previous outputs, while 86 changed inputs were newly inferred in the historical experiment. Article-level deduplication is applied after union. Extra model calls, total output budget and repeated context differ from whole-document inference, so this evaluates the pipeline rather than isolating a semantic-boundary effect. This remains generative inference; no BIO/BILOU token-classification model was trained by these scripts.
