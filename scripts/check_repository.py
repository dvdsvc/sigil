"""Check the release allowlist, copied-source hashes, syntax and new guidance.

Scans report filenames only when flagging a credential-like value; no matching
values are printed. This bounded scan is not proof of absence of every secret.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
KEY = re.compile(r"sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{25,}|ghp_[A-Za-z0-9]{25,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")
PRIVATE = re.compile(r"/" + r"Users/[^/\s]+/|/hy-" + r"tmp/|/private/" + r"var/folders/")
RAW_FIELDS = {"raw_output", "text", "body", "context_prefix", "prefix_names", "nonliteral_prefix_names", "known_reference", "original_known_reference", "gained_known_ids", "lost_known_ids"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_json_fields(obj, path, errors):
    if isinstance(obj, dict):
        forbidden = RAW_FIELDS & obj.keys()
        if forbidden:
            errors.append("Disallowed raw-data fields: " + path)
        for value in obj.values():
            check_json_fields(value, path, errors)
    elif isinstance(obj, list):
        for value in obj:
            check_json_fields(value, path, errors)


def main():
    errors = []
    records = json.loads((ROOT / "provenance/source_manifest.json").read_text())
    for record in records:
        path = ROOT / record["path"]
        if not path.is_file() or sha(path) != record["packaged_sha256"]:
            errors.append("Source-copy hash mismatch: " + record["path"])
    paths = sorted(p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts)
    code_count, links, text_count = 0, 0, 0
    for path in paths:
        rel = str(path.relative_to(ROOT))
        if path.is_symlink() or any(part in {"__pycache__", "node_modules", "models", "runs", ".venv", ".venv-gpu"} for part in path.relative_to(ROOT).parts):
            errors.append("Runtime/model/dependency payload: " + rel)
        if path.suffix.lower() in {".safetensors", ".pt", ".pth", ".bin", ".docx", ".pdf", ".zip", ".gz", ".tar", ".log"} or (path.name.startswith(".env") and path.name != ".env.example"):
            errors.append("Unapproved private/generated payload: " + rel)
        data = path.read_bytes()
        if b"\x00" in data:
            errors.append("Unexpected binary file: " + rel)
            continue
        text = data.decode("utf-8")
        text_count += 1
        if KEY.search(text) or PRIVATE.search(text):
            errors.append("Possible credential/private locator: " + rel)
        if path.suffix == ".py":
            code_count += 1
            try:
                compile(data, str(path), "exec")
            except (SyntaxError, ValueError):
                errors.append("Python syntax error: " + rel)
            tree = ast.parse(text)
            if rel == "src/output_metrics.py":
                functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
                assert {n.name for n in functions} == {"parse", "prefix"}
                assert all(isinstance(n, (ast.Expr, ast.Import, ast.FunctionDef)) for n in tree.body)
                imports = {a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names}
                assert imports == {"json", "re"}
        if path.suffix == ".md":
            targets = re.findall(r"!?\[[^\]]*\]\(([^\n)]+)\)", text)
            targets += re.findall(r'<img\b[^>]*\bsrc="([^"]+)"', text)
            for target in targets:
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                candidate = (path.parent / unquote(target.split("#")[0])).resolve()
                if not candidate.is_relative_to(ROOT) or not candidate.exists():
                    errors.append("Unresolved local Markdown link: " + rel)
                links += 1
        if path.suffix in {".json", ".jsonl"} and rel.startswith("evidence/"):
            objects = [json.loads(s) for s in text.splitlines() if s.strip()] if path.suffix == ".jsonl" else [json.loads(text)]
            for obj in objects:
                check_json_fields(obj, rel, errors)

    inventory = ROOT / "SHA256SUMS"
    verified = 0
    if not inventory.is_file():
        errors.append("Root SHA256SUMS missing")
    else:
        expected = set()
        for line in inventory.read_text().splitlines():
            digest, relative = line.split("  ", 1)
            expected.add(relative)
            path = ROOT / relative
            if not path.is_file() or sha(path) != digest:
                errors.append("Candidate inventory mismatch: " + relative)
        if expected != {str(p.relative_to(ROOT)) for p in paths if p != inventory}:
            errors.append("Candidate file inventory changed")
        verified = len(expected)
    print(json.dumps({"source_transformations_verified": len(records), "inventory_files_verified": verified,
                      "python_sources_compiled_in_memory": code_count, "local_markdown_links_checked": links,
                      "text_files_scanned": text_count, "files": len(paths),
                      "bytes": sum(p.stat().st_size for p in paths), "errors": errors,
                      "new_training_inference_or_upload": False, "passed": not errors}, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
