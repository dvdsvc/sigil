"""Offline aggregate and training-log verification; no raw articles or model calls."""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "evidence/primary"


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def close(a, b):
    if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12):
        raise AssertionError(f"Aggregate mismatch: {a} != {b}")


def metrics(data, prefix=False):
    key = "prefix_known_recovered" if prefix else "strict_known_recovered"
    total = sum(r["known_reference_count"] for r in data)
    hits = sum(r[key] for r in data)
    return {
        "articles": len(data), "json_usable": sum(r["json_usable"] for r in data),
        "known_hits": hits, "known_reference_count": total,
        "known_micro_recall": hits / total,
        "article_equal_known_recall": statistics.mean(
            r[key] / r["known_reference_count"] for r in data if r["known_reference_count"] > 0
        ),
    }


def main():
    config = read(PRIMARY / "config.json")
    source = read(PRIMARY / "INDEPENDENT_COMPARISON.json")
    review = read(PRIMARY / "OUTPUT_TRUNCATION_REVIEW.json")
    count_rows = rows(PRIMARY / "article_counts.jsonl")
    arms = {a: [r for r in count_rows if r["arm"] == a] for a in ("direct_sft", "cpt_then_sft")}
    assert len(count_rows) == 38 and all(len(rs) == 19 for rs in arms.values())
    assert [r["article_id"] for r in arms["direct_sft"]] == [r["article_id"] for r in arms["cpt_then_sft"]]
    for a, b in zip(arms["direct_sft"], arms["cpt_then_sft"]):
        assert a["text_sha256"] == b["text_sha256"] and a["input_tokens"] == b["input_tokens"]
        assert not a["input_truncated"] and not b["input_truncated"]
        assert a["known_reference_count"] == b["known_reference_count"]

    initialization = [read(PRIMARY / a / "INITIALIZATION.json") for a in arms]
    assert initialization[0] == initialization[1]
    assert initialization[0]["fresh_common_base_sha256"] == config["base_weight_sha256"]
    assert initialization[0]["no_prior_task_adapter_loaded"] is True
    cpt_schedule = rows(PRIMARY / "cpt_schedule.jsonl")
    sft_schedule = rows(PRIMARY / "sft_schedule.jsonl")
    cpt_log = [r for r in rows(PRIMARY / "cpt_then_sft/training_steps.jsonl") if r["phase"] == "CPT"]
    assert len(cpt_schedule) == len(cpt_log) == config["CPT"]["optimizer_updates"] == 1464
    assert len({r["article_id"] for r in cpt_schedule}) == 128
    for i, (a, b) in enumerate(zip(cpt_schedule, cpt_log), 1):
        assert b["step"] == i and all(a[k] == b[k] for k in ("id", "text_sha256", "target_tokens"))
        assert a["token_end"] - a["token_start"] == a["target_tokens"]
        assert b["input_tokens"] == a["target_tokens"] + 1
    intervals = collections.defaultdict(list)
    for item in cpt_schedule:
        intervals[item["article_id"]].append((item["token_start"], item["token_end"]))
    for parts in intervals.values():
        ordered = sorted(parts)
        assert ordered[0][0] == 0
        assert all(left[1] == right[0] for left, right in zip(ordered, ordered[1:]))
    assert sum(r["target_tokens"] for r in cpt_log) == 2860163
    assert len(sft_schedule) == 72 and len({r["id"] for r in sft_schedule}) == 64
    assert max(r["input_tokens"] for r in sft_schedule) == 59864
    for arm in arms:
        log = [r for r in rows(PRIMARY / arm / "training_steps.jsonl") if r["phase"] == "SFT"]
        assert len(log) == 72
        for i, (a, b) in enumerate(zip(sft_schedule, log), 1):
            assert b["step"] == i and a["step"] == i
            assert all(a[k] == b[k] for k in ("id", "text_sha256", "input_tokens"))
            assert a["answer_tokens"] == b["target_tokens"]
        assert sum(r["target_tokens"] for r in log) == 22659
        summary = read(PRIMARY / arm / "SFT_TRAINING_SUMMARY.json")
        assert summary["complete"] and summary["steps"] == 72 and summary["trainable_parameters"] == 3194880
    eval_hashes = {r["text_sha256"] for r in count_rows}
    cpt_hashes = {r["text_sha256"] for r in cpt_schedule}
    val_hashes = {r["text_sha256"] for r in read(PRIMARY / "CPT_VALIDATION_MANIFEST.json")}
    sft_hashes = {r["text_sha256"] for r in sft_schedule}
    assert not eval_hashes & (cpt_hashes | val_hashes | sft_hashes)
    assert not val_hashes & (cpt_hashes | sft_hashes)

    output = {"study": "same-origin LoRA CPT versus direct SFT", "strict": {}, "prefix": {}}
    failed_ids = {r["article_id"] for r in count_rows if not r["json_usable"] or r["output_limit_reached"]}
    assert failed_ids == set(source["jointly_excluded_failed_ids"])
    for arm, rs in arms.items():
        strict, prefix = metrics(rs), metrics(rs, prefix=True)
        saved = source["full19"][arm]
        assert strict["known_hits"] == saved["known_recovered"] and strict["known_reference_count"] == 210
        assert strict["json_usable"] == saved["json_usable"] == 12
        close(strict["known_micro_recall"], saved["known_micro_recall"])
        close(strict["article_equal_known_recall"], saved["mean_per_article_known_recall"])
        saved_prefix = review["summary"]["arms"][arm]
        assert prefix["known_hits"] == saved_prefix["recoverable_prefix_known_hits"]
        close(prefix["known_micro_recall"], saved_prefix["recoverable_prefix_known_micro_recall"])
        close(prefix["article_equal_known_recall"], saved_prefix["recoverable_prefix_mean_per_article_known_recall"])
        joint = metrics([r for r in rs if r["article_id"] not in failed_ids])
        assert joint["articles"] == 9 and joint["known_hits"] == 6 and joint["known_reference_count"] == 15
        output["strict"][arm], output["prefix"][arm] = strict, prefix
    behaviors = collections.Counter(c["behavior"] for c in review["cases"])
    assert dict(behaviors) == review["summary"]["observed_behavior_counts"]
    assert sum(behaviors.values()) == 14 and behaviors["repetition_loop"] == 8
    before, after = [read(PRIMARY / "cpt_then_sft" / (phase + "_CPT_validation.json")) for phase in ("before", "after")]
    assert before["target_tokens"] == after["target_tokens"] == 54960
    assert before["sequences"] == after["sequences"] == 32
    output["LM_validation"] = {
        "before": before["target_token_weighted_loss"], "after": after["target_token_weighted_loss"],
        "relative_loss_decrease_percent": 100 * (1 - after["target_token_weighted_loss"] / before["target_token_weighted_loss"]),
        "is_extraction_accuracy": False,
    }
    secondary = read(ROOT / "evidence/structure/article_counts.json")
    secondary_source = read(ROOT / "evidence/structure/INDEPENDENT_COMPARISON.json")
    assert len(secondary) == 19 and sum(r["chunk_count"] for r in secondary) == 410
    assert sum(r["invalid_chunks"] + r["limited_chunks"] for r in secondary) == 0
    secondary_out = {}
    for arm in ("whole", "chunk"):
        hits = sum(r[arm + "_corrected_recovered"] for r in secondary)
        refs = sum(r["corrected_reference_count"] for r in secondary)
        macro = statistics.mean(r[arm + "_corrected_recovered"] / r["corrected_reference_count"] for r in secondary if r["corrected_reference_count"] > 0)
        saved = secondary_source["locally_corrected_labels"]["full19"][arm]
        assert hits == saved["known_recovered"] and refs == saved["known_reference_count"] == 210
        close(macro, saved["mean_per_article_known_recall"])
        secondary_out[arm] = {"known_hits": hits, "known_reference_count": refs, "known_micro_recall": hits / refs, "article_equal_known_recall": macro}
    assert secondary_source["reused_identical_inputs"] == 324 and secondary_source["new_inference_inputs"] == 86
    output["secondary_structure_study_different_adapter"] = secondary_out
    output["training_verified"] = {"CPT_updates": 1464, "CPT_target_tokens": 2860163, "SFT_updates_per_arm": 72, "SFT_answer_tokens_per_arm": 22659, "trainable_parameters": 3194880}
    output["limits"] = "Count-level reproduction plus training-log checks; raw-text/generation audit requires private originals. Incomplete weak labels, not blind-test full precision/F1; CPT uses extra compute."
    output["new_training_inference_or_upload"] = False
    output["passed"] = True
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
