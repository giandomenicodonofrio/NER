"""Summarize the most frequent errors from an exported prediction TSV."""

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_PREDICTIONS = Path(
    "outputs/error_analysis/error_analysis_all_datasets/predictions/test_errors.tsv"
)
DEFAULT_OUTPUT_DIR = Path("outputs/analysis/error_analysis/error_analysis_all_datasets")
REQUIRED_COLUMNS = {"token", "gold", "pred"}


def analyze_errors(predictions_path: Path, output_dir: Path) -> None:
    """Write aggregate label confusions and the most frequent token errors."""
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    df = pd.read_csv(predictions_path, sep="\t")
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {predictions_path}: "
            f"{sorted(missing_columns)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    confusions = (
        df.groupby(["gold", "pred"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    confusions.to_csv(output_dir / "top_confusions.csv", index=False)

    tokens = (
        df.groupby(["token", "gold", "pred"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    tokens.to_csv(output_dir / "top_error_tokens.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS,
        help="TSV exported by the evaluation pipeline.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for error-analysis CSV files.",
    )
    args = parser.parse_args()

    analyze_errors(args.predictions, args.out_dir)
    print(f"Saved error analysis to: {args.out_dir}")


if __name__ == "__main__":
    main()
