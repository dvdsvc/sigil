# Local validation

Date: 2026-10-27. Scope: independent local showcase preparation using repository-polish in right-sized mode. No new GPU training, model inference, provider grading, remote publication or upload was performed.

| Check | Local evidence |
|---|---|
| Existing private archive | Archive hash and all 542 payload member hashes verified without copying its private content into the candidate |
| Primary source and data | Trainer matches server-frozen code and recorded runner hash; four private data-file hashes verified |
| Primary original outputs | 38 raw predictions checked against unchanged source text/hash, strict parser, prefix parser and recorded hit counts before generating no-text release evidence |
| Training provenance | Matching base/fresh-LoRA initialization; 1,464 CPT / 72 SFT schedules and logs checked; token targets, per-step IDs and exact-text split disjointness checked |
| Secondary original outputs | All 19 article bodies reconstructed from 410 chunk slices; 410 raw chunk predictions parsed, 324 cached-input markers checked and 75 corrected known hits recomputed locally |
| Public reproduction | Released count-level aggregates, fixed LM losses and training-log checks pass; this is distinct from re-annotation or replaying omitted raw generations |
| Parser contracts | Seven tests pass, covering unfinished strings, non-string tails, wrong fields, empty delivery, fenced output and repeated strings |
| Workspace helper | Synthetic integration fixture verifies new-workspace staging, existing/candidate-contained path rejection, mismatched-data rejection before writes, and preservation of fixture model bytes; no real model used |
| Source and guidance | Python files compiled in memory; local Markdown/image targets and copied-artifact hashes checked by the repository verifier |
| Sensitive payload | Candidate text scanned for high-confidence credentials and private filesystem locators; evidence JSON checked for disallowed raw-text/output/identifier-list fields; no weights, papers, raw generations, private model-store identifiers or backup/upload automation included |
| CI | Offline-only workflow added; equivalent checks run locally. Hosted CI execution is not asserted |

The root `SHA256SUMS` covers all candidate files except itself. Run [the repository verifier](../scripts/check_repository.py) after extracting the ZIP; it reports current counts and integrity. Intentional future README/code changes require refreshing that inventory. Original and packaged hashes are in [the source manifest](../provenance/source_manifest.json). The private owner review outside the candidate records the ZIP check and original-preservation audit.

The public payload supports aggregate reproduction without releasing copyrighted source papers. A complete scientific raw-output replay requires the owner's original private corpus, predictions and model artifacts. Historical GPU dependency restoration and a fresh training run were not attempted here. A release license remains the owner's choice.
