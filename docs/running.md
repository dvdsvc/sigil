# Running and reproduction

[Home](../README.md) · [Data requirements](data.md)

## Offline checks

Use Python 3.11+; Python 3.12.13 was used for the local showcase checks. There are no third-party dependencies for these commands:

```sh
uv run --offline --no-project python -B scripts/reproduce_results.py
uv run --offline --no-project python -B scripts/demo.py
uv run --offline --no-project python -B -m unittest discover -s tests -v
uv run --offline --no-project python -B scripts/check_repository.py
```

`--offline` prevents uv package retrieval, `--no-project` avoids GPU dependency installation, and `-B` avoids bytecode artifacts. Supply an installed interpreter with `--python /path/to/python3.12` before `python -B` if needed. Direct `python -B ...` is equivalent for the scripts. No command in this block trains, calls a provider, downloads a model or uploads artifacts.

The result script verifies matching initialization and 72-step SFT schedules, the 1,464-step CPT schedule/log and target-token coverage, exact-text split disjointness for released hashes, retained summaries and article-level aggregates. Prefix/strict aggregate reproduction starts from released numeric counts. Re-parsing original generations and auditing literal grounding require the private originals. The demo/tests exercise the original parsing algorithms on explicitly fictional inputs, not model performance.

## Optional GPU environment restoration

Actual training used Linux, Python 3.11.12, an RTX 3090 and the packages in [TRAINING_ENVIRONMENT.json](../requirements/TRAINING_ENVIRONMENT.json). Selected recorded pins are in [training-recorded.txt](../requirements/training-recorded.txt), including the recorded CUDA 12.6 Torch build. They are historical provenance, not a promise that every platform can install that environment unchanged.

```sh
uv venv --python 3.11 .venv-gpu
uv pip install --python .venv-gpu/bin/python -r requirements/training-recorded.txt
```

These optional commands install dependencies and require suitable package access/CUDA support. They were not run during showcase preparation. Review the recorded complete environment for any platform-specific acceleration packages required by the restored setup.

## Restore the primary training layout

Restore the owner's four private corpus files and matching `DATA_MANIFEST.json` to a separate data directory. Restore the original base model/tokenizer files with the exact expected weight hash. Each final adapter must load against the original base, not be stacked on a prior v4/B/CPT-only adapter.

The archived trainer reads its system prompt from a historical `runs/B_v21_union_20261014_01/config.json` location. The new helper supplies **only that prompt** from released config and links the owner-provided base into a **new** workspace. It does not inherit the old adapter, copy weights or change supplied model/corpus files:

```sh
uv run --offline --no-project python -B scripts/prepare_workspace.py \
  --workspace /path/to/new-training-workspace \
  --base-model /path/to/original-Qwen3.5-0.8B \
  --data /path/to/restored-private-corpus
```

All four corpus hashes, the manifest and original base weight hash must match. The workspace must not already exist and must be outside the public candidate. Once the separate GPU environment and data are ready:

```sh
uv run --no-project --python .venv-gpu/bin/python python -B \
  src/historical/train_domain_pretraining_ab_20261014.py \
  --workspace /path/to/new-training-workspace \
  --data /path/to/restored-private-corpus
```

This last command starts actual training and inference. The trainer has a fixed historical run name and refuses an existing training log. Its `--prepare-only` still loads the tokenizer and writes prepared schedules; it is distinct from the safe offline aggregate quick start.

## Data preparation and historical code

The preparation code requires private grouped-source metadata, prior SFT source records and the 19 evaluation source records in the legacy research layout. Only its location arguments were adapted:

```sh
python -B src/historical/prepare_domain_pretraining_20261014.py \
  --legacy-root /path/to/restored-legacy-research \
  --source-docx-root /path/to/authorized-docx-corpus \
  --output /path/to/new-prepared-corpus
```

The output directory must be new. Source selection/projection and supervision rules are retained. The standalone DOCX projector is also usable on an owner-provided document with `--docx` and a new `--output-directory`.

Other historical training/audit scripts preserve earlier workspace assumptions and optional prior adapters. They are source evidence, not safe importable modules or universal one-command reruns. The original truncation script's model-repository publishing tail, connection/backup utilities, article corpora, raw output files and model weights are not included. Pure parsing functions are extracted separately for offline use.
