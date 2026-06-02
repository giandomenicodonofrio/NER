import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.scripts.error_analysis import analyze_errors


class ErrorAnalysisTest(unittest.TestCase):
    def test_writes_confusion_and_token_summaries(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            predictions_path = root / "errors.tsv"
            output_dir = root / "out"

            pd.DataFrame(
                [
                    {"dataset": "WN", "token": "Roma", "gold": "B-LOC", "pred": "O"},
                    {"dataset": "WN", "token": "Roma", "gold": "B-LOC", "pred": "O"},
                    {"dataset": "WN", "token": "ACME", "gold": "B-ORG", "pred": "B-LOC"},
                ]
            ).to_csv(predictions_path, sep="\t", index=False)

            analyze_errors(predictions_path, output_dir)

            confusions = pd.read_csv(output_dir / "top_confusions.csv")
            tokens = pd.read_csv(output_dir / "top_error_tokens.csv")

            self.assertEqual(confusions.iloc[0].to_dict(), {
                "gold": "B-LOC",
                "pred": "O",
                "count": 2,
            })
            self.assertEqual(tokens.iloc[0].to_dict(), {
                "token": "Roma",
                "gold": "B-LOC",
                "pred": "O",
                "count": 2,
            })


if __name__ == "__main__":
    unittest.main()
