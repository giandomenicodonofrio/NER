import unittest

from src.evaluation.metrics import (
    compute_entity_metrics,
    compute_nermud_metrics,
    normalize_nermud_label,
)


class NermudMetricsTest(unittest.TestCase):
    def test_normalizes_bio_prefix_and_pipe_suffix(self):
        self.assertEqual(normalize_nermud_label("I-PER|extra"), "PER")

    def test_computes_official_token_level_macro_and_micro_f1(self):
        gold = [["B-PER", "I-PER", "O", "B-LOC", "B-ORG"]]
        pred = [["B-PER", "O", "O", "B-ORG", "B-ORG"]]

        metrics = compute_nermud_metrics(gold, pred)

        self.assertAlmostEqual(metrics["nermud_macro_f1"], 4 / 9)
        self.assertAlmostEqual(metrics["nermud_micro_f1"], 4 / 7)
        self.assertNotEqual(
            metrics["nermud_macro_f1"],
            compute_entity_metrics(gold, pred)["entity_f1"],
        )


if __name__ == "__main__":
    unittest.main()
