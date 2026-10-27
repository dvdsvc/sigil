# Data and released evidence

[Home](../README.md) · [Methods](methods.md) · [Provenance](provenance.md)

| Stage | Recorded population | Included here |
|---|---|---|
| Original DOCX preparation | 335 documents examined, 230 eligible after known-source/duplicate exclusions | Source-processing code and aggregate manifest |
| CPT | 128 training documents; 8 independent LM-validation documents | Hashes, numeric schedules, 1,464-step log and validation records |
| SFT | 64 complete real articles, including 4 reviewed negatives | 72-step schedule and both training logs, configuration and initialization hashes |
| Primary regression evaluation | 19 articles / 210 known identifiers; 38 predictions across two arms | No-text per-article numeric counts, frozen summaries and truncation-case counts |
| Separate structure evaluation | Same 19 articles, different fixed adapter, 410 corrected chunks | Article-level counts, configuration and aggregate audits |

The original private archive contains article text and raw generations. Those files and model/tokenizer weights are intentionally absent from this candidate. It also excludes backup/upload scripts, server connection details and private model-repository identifiers. No model or data download occurs in the offline quick start.

`evidence/primary/article_counts.jsonl` contains arm, stable article ID/alias, text hash, known-reference count, strict/prefix hit counts, delivery flags, token counts and timings. It contains no actual identifier lists or article text. The original prediction files were checked against their source texts and both parser rules during local preparation; [the receipt](../provenance/private_audit_receipt.json) records this. Public reproduction recomputes aggregates from released counts and checks training logs. It cannot independently re-annotate entities or re-parse unpublished raw generations.

The eight empty inventories are unknown, not confirmed negative evaluation documents. The four reviewed negative SFT documents are a different set and serve a training role. Known labels are incomplete; additional predictions are not all false positives, so full precision/F1 is not reported. The 19 articles were repeatedly examined during earlier development and are not a new blind test.

The structure experiment originally used a 223-item inventory. Thirteen confirmed heating-stage numbers were removed to obtain the locally corrected 210-item inventory; this single correction is not a claim of comprehensive label cleaning. Retain raw/corrected and full19/historical17 scopes separately. No copyrighted article snippets are needed for the bundled parser demonstration: its `DEMO-*` names are explicitly fictional.
