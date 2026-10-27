"""Exercise actual archived parsing rules with hand-written fictional examples."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from output_metrics import parse, prefix

examples = [
    ("complete", '{"entity_ids": ["DEMO-S01", "DEMO-S02"]}'),
    ("unfinished_repetition", '{"entity_ids": ["DEMO-S01", "DEMO-S01", "DEMO-S02'),
    ("malformed_tail", '{"entity_ids": ["DEMO-S01", 123, "DEMO-S03"]}'),
]
for name, raw in examples:
    strict_names, valid = parse(raw)
    complete = strict_names if valid else prefix(raw)
    print(json.dumps({"example": name, "synthetic_not_model_output": True,
                      "strict_json_usable": valid, "strict_names": strict_names,
                      "recoverable_complete_strings": list(dict.fromkeys(complete))}))
