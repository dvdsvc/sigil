"""Delivery validity and partial recognition must remain different outcomes."""
import importlib.util
from pathlib import Path
import unittest

source = Path(__file__).resolve().parents[1] / "src/output_metrics.py"
spec = importlib.util.spec_from_file_location("output_metrics", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class OutputContract(unittest.TestCase):
    def test_usable_json_deduplicates_in_original_order(self):
        self.assertEqual(module.parse('{"entity_ids":["DEMO-B","DEMO-A","DEMO-B"]}'), (["DEMO-B", "DEMO-A"], True))

    def test_cut_identifier_is_never_completed(self):
        raw = '{"entity_ids":["DEMO-A","DEMO-B'
        self.assertEqual(module.parse(raw), ([], False))
        self.assertEqual(module.prefix(raw), ["DEMO-A"])

    def test_numeric_tail_stops_without_skipping_ahead(self):
        raw = '{"entity_ids":["DEMO-A",123,"DEMO-B"]}'
        self.assertEqual(module.parse(raw), ([], False))
        self.assertEqual(module.prefix(raw), ["DEMO-A"])

    def test_wrong_field_is_not_a_recognized_list(self):
        self.assertEqual(module.parse('{"sample_ids":["DEMO-A"]}'), ([], False))
        self.assertEqual(module.prefix('{"sample_ids":["DEMO-A"]}'), [])

    def test_valid_empty_list_is_deliverable(self):
        self.assertEqual(module.parse('{"entity_ids":[]}'), ([], True))

    def test_existing_fence_support_is_preserved(self):
        self.assertEqual(module.parse('```json\n{"entity_ids":["DEMO-A"]}\n```'), (["DEMO-A"], True))

    def test_prefix_preserves_duplicate_occurrences_for_loop_audit(self):
        self.assertEqual(module.prefix('{"entity_ids":["DEMO-A","DEMO-A",'), ["DEMO-A", "DEMO-A"])


if __name__ == "__main__":
    unittest.main()
