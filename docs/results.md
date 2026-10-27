# Results and experiment lineage

[Home](../README.md) · [Reproduce](running.md)

## Primary: same-origin CPT versus direct SFT

The retained initialization records have identical original-base and fresh-LoRA hashes. Training completed in both arms. CPT uses 128 articles, 1,464 updates and 2,860,163 target tokens; both SFT stages use the same 64 articles / 72 updates / 22,659 answer tokens. Fixed LM validation uses 8 articles, 32 sequences and 54,960 target tokens. Its weighted loss decreases 2.1534167 → 1.9582802 (−9.06%). This is a language prediction result, separate from extraction.

| Full 19-article panel | Direct SFT | CPT → same SFT |
|---|---:|---:|
| Complete JSON usable | 12/19 | 12/19 |
| Output cap reached | 7 | 7 |
| Strict-delivery known hits | 23/210 (10.95%) | 18/210 (8.57%) |
| Strict-delivery article-equal known recall | 32.95% | 29.55% |
| Recoverable-prefix known hits | 52/210 (24.76%) | 28/210 (13.33%) |
| Recoverable-prefix article-equal known recall | 47.61% | 51.14% |

The strict common-success subset contains 9 articles, with only 15 known identifiers; both arms recover 6/15. It is a selected subset and does not replace the full 19-article denominator. Eight articles have unknown empty inventories; recall averages use the 11 with known identifiers.

The prefix correction changes interpretation of invalid delivery: completed identifier strings can still be recognized before the list ends. Fourteen capped outputs were reviewed: eight repetition loops, six nonliteral sequence-extension tails. The original delivery flags are retained. The recovery algorithm never completes a cut string or invents a missing identifier. Higher-output-budget inference was not performed.

These results establish completed training and a measurable domain language-loss improvement. Task findings depend on aggregation: prefix known-identifier micro recovery decreases, while article-equal prefix recall increases. They do not support a universal CPT success/failure conclusion, an equal-compute claim or a full-document precision/F1 score. The original report's strict-only interpretation was corrected by the [later truncation review](../reports/OUTPUT_TRUNCATION_REVIEW_ZH.md).

## Secondary: table-boundary-corrected generative inference

This study uses a different already-trained `B_v21_union` adapter, fixed across whole/segmented input. It is separate from the fresh-LoRA CPT arms. The completed corrected report supersedes an earlier 478-chunk analysis whose table-boundary/label issue inflated its apparent gain.

| Corrected full19 inventory | Whole document | Corrected segmentation |
|---|---:|---:|
| Known hits | 94/210 (44.76%) | 75/210 (35.71%) |
| Article-equal known recall | 57.44% | 38.77% |
| Predicted identifiers, deduplicated within article | 481 | 678 |
| Unreviewed additional outputs | 387 | 603 |

All 410 chunks are parseable without output truncation. Each article body is covered exactly, with literal contextual headings/table headers repeated where needed. Of the 410 inputs, 324 identical inputs reuse archived outputs; 86 changed inputs were re-inferred in the original experiment. The 20.77-minute full inference cost is an estimate combining new inference with cached-original timings, not a newly measured full wall-clock rerun. More calls and total output budget differ from whole-document inference.

The boundary correction improves experimental validity and removes misleading counts; it does not yield an extraction gain in this panel. Shorter input alone does not solve the recorded recognition omissions. [Completed report](../reports/STRUCTURAL_BOUNDARY_REVIEW_ZH.md).

## Historical adaptation routes

Earlier experiments compared union-then-typed and typed training. One continuing-SFT study starts from the same old v4 adapter, while the later real-full-text route study inherits its own separately DB-trained A/B parents and uses different total updates. These histories are retained as code and [a lineage report](../reports/HISTORICAL_ROUTE_LINEAGE_ZH.md), not pooled into the fresh-base CPT experiment. Historical diagnostic fragment/table metrics are not promoted to full-document benchmark scores.

Evidence: [primary frozen comparison](../evidence/primary/INDEPENDENT_COMPARISON.json), [output audit](../evidence/primary/OUTPUT_TRUNCATION_REVIEW.json), [secondary comparison](../evidence/structure/INDEPENDENT_COMPARISON.json). No new training, inference or scoring was performed for this showcase.
