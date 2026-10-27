"""Stage a new historical training workspace from owner-restored private data.

Creates only a new directory and a model-directory symlink. It neither trains,
downloads model files nor writes into the supplied corpus or base directory.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True, help="A directory that does not yet exist")
    parser.add_argument("--base-model", type=Path, required=True, help="Owner-restored untouched original base directory")
    parser.add_argument("--data", type=Path, required=True, help="Owner-restored four-file private corpus plus DATA_MANIFEST.json")
    args = parser.parse_args()
    workspace, base, data = args.workspace.resolve(), args.base_model.resolve(), args.data.resolve()
    if workspace.exists():
        parser.error("Workspace already exists; choose a new directory")
    if workspace == ROOT or workspace.is_relative_to(ROOT):
        parser.error("Keep training workspaces outside this public candidate")
    config = json.loads((ROOT / "evidence/primary/config.json").read_text())
    for name, expected in config["data_manifest"]["files_sha256"].items():
        path = data / name
        if not path.is_file() or file_hash(path) != expected:
            parser.error("Private data missing or hash mismatch: " + name)
    manifest = data / "DATA_MANIFEST.json"
    if not manifest.is_file() or json.loads(manifest.read_text()) != config["data_manifest"]:
        parser.error("Private DATA_MANIFEST.json does not match the archived protocol")
    weight = base / "model.safetensors-00001-of-00001.safetensors"
    if not weight.is_file() or file_hash(weight) != config["base_weight_sha256"]:
        parser.error("Original base weight missing or hash mismatch")
    if not (base / "tokenizer_config.json").is_file():
        parser.error("Base tokenizer files missing")
    (workspace / "models").mkdir(parents=True, exist_ok=False)
    (workspace / "models/Qwen3.5-0.8B").symlink_to(base, target_is_directory=True)
    legacy = workspace / "runs/B_v21_union_20261014_01"
    legacy.mkdir(parents=True)
    (legacy / "config.json").write_text(json.dumps({"system": config["system"]}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"workspace_prepared": True, "model_copied": False,
                      "new_training_or_inference": False,
                      "note": "The compatibility directory supplies only the historical system prompt; no prior adapter is inherited."}))


if __name__ == "__main__":
    main()
