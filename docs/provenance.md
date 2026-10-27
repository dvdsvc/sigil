# Provenance, ownership and preparation changes

[Home](../README.md) · [Source-copy manifest](../provenance/source_manifest.json) · [Private audit receipt](../provenance/private_audit_receipt.json)

## Original research materials

The primary evidence comes from `domain_pretraining_ab_20261014`, its completed server snapshot and independently recomputed comparison. A pre-existing private experiment archive was found and verified against its receipt: SHA256 `2d41c16f9ac8502bbb725d5077acee548f9cfb3e36d04be068fc9df022548241`, 542 payload files plus its manifest. It contains private article text/raw predictions, so the archive itself is not copied here.

The main trainer matches the recorded `runner_sha256` and server-frozen trainer byte for byte. Original source files remain outside this candidate. Source manifests record original/package hashes and each transformation. These task-output source directories do not have their own Git histories; no invented commit or per-line authorship attribution is supplied. Stored hashes, configs, schedules and reports establish implementation states.

## Research and upstream boundaries

The work applies an existing small model and PEFT LoRA to a scientific identifier task. Research-specific work includes source-aware document construction, union/typed task formulations and lineage, controlled initialization and schedules, real-domain CPT/SFT execution, full-document evaluation, conservative incomplete-output recovery and table-boundary analysis. Transformers, PyTorch, PEFT, the original model architecture, LoRA and standard causal/answer-only cross-entropy training are existing methods and implementations.

The user's independent project attribution does not imply authorship of upstream libraries or a new CPT algorithm. The showcase adds front matter, safe offline entry points, no-text release evidence and integrity checks; these additions are distinguished from the historical research implementation.

## Exact adaptations

| Artifact | Change in the candidate |
|---|---|
| Primary CPT/SFT trainer | Preserved byte for byte, including original layout dependencies and hash guards |
| DOCX projector, typed-output helper, final primary audit, corrected structure evaluator | Preserved byte for byte |
| Primary data-preparation script | Explicit required legacy-root / DOCX-root / new-output arguments; no source-selection, label or projection algorithm change |
| Historical full-route trainer | Private parent-adapter repository/server location generalized; recorded hashes and training logic retained |
| Strict and prefix parsers | Original function bodies selected by AST; json/re imports only; original audit's publication tail excluded |
| Training evidence | Logs/summaries retained; CPT schedule/validation records omit paper/group names while retaining IDs, hashes and numeric coverage |
| Evaluation evidence | Per-article numeric counts derived from private source/label/output audit; no article text, raw generation or actual identifier list released |
| Reports | Article names use stable aliases; private backup locations generalized; numerical findings retained |

Original and packaged hashes cannot by themselves prove completeness or scientific correctness. The released counts support aggregate reproduction, and the logs support schedule checks. Re-annotation, full model re-execution or raw-generation replay require owner-controlled materials. No new training, inference or upload occurred during this preparation.

## Experiment versions

Keep the fresh-base CPT experiment, older inherited-adapter routes and later fixed-adapter structure inference separate. The completed table-boundary correction supersedes the earlier 478-chunk report's apparent gain. The prefix audit corrects a strict-delivery-only interpretation without replacing original JSON-valid flags. BIO/BILOU or encoder token classification is a possible future task formulation, not an implemented trained model in this package.

## Release scope

Original private archives, training/model files and results remain unchanged. The candidate contains only selected code, no-text metadata/counts, training traces and authored reports/examples. Framework implementations and pretrained weights are not vendored. Research-specific license selection remains with the owner; see [third-party notices](../THIRD_PARTY_NOTICES.md). No repository push, manuscript publication or website deployment is part of this local preparation.
